from safetensors import safe_open
from safetensors.numpy import save_file as _save_file
from torchic.nn.models.test_myparser import build_from_repr, _parse_arg
from torchic import nn, Tensor  # usato in caso di eval
from torchic.settings import CUPY_AVAILABLE
import warnings
import numpy as np
import re

#import datetime
#now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# - assicurati che i parametri da trainare non vengano
#   portati su cpu quando chiami state_dict() (al momento SUCCEDE
#   sia per model che per optimizer). O reworko una funzione precedente,
#   o ne creo un'altra che passa il .data su cpu e lo returna. Stavo
#   pensando a .clone(), che crea una deep copy del tensore, o .detach()
# - commenta la funzione di load dell'optimizer
# - visto che ho dovuto salvare ogni ingresso dei vettori m e v
#   separatamente, la relativa logica di load e' da riscrivere


def _load_safetensors(filename, load_metadata=False) -> dict[str]:
    '''
    Function used internally to read data from .safetensors files.
    '''
    result = {"metadata":{}} if load_metadata else {}

    with safe_open(filename, framework="np") as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)

        if load_metadata:
            for k, v in f.metadata().items():
                result["metadata"][k] = v
    return result


# volendo si puo' modificare per fare in modo che salvi tutto su un unico file
def save(model: nn.Sequential, optim=None, filename: str="checkpoint", epoch: int=-1, info: dict=None) -> None:
    '''
    Creates two `.safetensors` file (if `optim` is given) containing
    the data necessary to reload the given model and optimizer.

    Args:
        model (nn.Sequential): model to save
        optim (nn.Optimizer): optimizer to save. Default is `None`
        filename (str): name of the files. The optimizer filename
            will be in the form `{filename}_optimizer.safetensors`

    This function serializes a dict of tensor into raw bytes in
    safetensors format with `safetensors.numpy.save_file()`.
    Metadata can be stored as a dict, but each value has to be
    convertible to a `PyString` object.

    The model's file metadata is a `dict` {'model_repr': `model.repr()`}
    to allow the reconstruction of the architecture.

    The optimizer's file metadata is a `dict` containing pairs
    {"`category`.`attr_name`": `str`|`int`|`float`}, which contains
    every simple attribute that doesn't need to be serialized,
    and {"optimizer_cls": `optim.__class__.__name__`}.

    For more info about the structure of the serialized dict, check
    `nn.Sequential.state_dict()` and `nn.Optimizer.state_dict()`.
    '''
    # NOTE: il metadata dict puo' SOLO contenere int, float o stringhe
    _save_file(model.state_dict(), f"{filename}.safetensors", metadata={"model_repr": repr(model), "epoch": str(epoch)})

    if optim:
        metadata = {"optimizer_cls": optim.__class__.__name__}
        internal_state = {}

        # divido tra cosa va salvato nei metadata e cosa nel body del file
        for key, value in optim.state_dict().items():
            if type(value) == np.ndarray:
                internal_state[key] = value
            else:
                metadata[key] = str(value)

        _save_file(internal_state, f"{filename}_optimizer.safetensors", metadata=metadata)


def __process_layer_keys(layer_keys):
    from collections import defaultdict

    grouped_layers = defaultdict(lambda: defaultdict(set))

    for key in layer_keys:
        # Estrarre informazioni utili dalla chiave
        parts = key.split(".")
        if len(parts) > 3:  # Verifica di avere abbastanza parti per raggruppare
            group = parts[0]  # encoder/decoder
            layer = parts[2]  # layer index
            attention_type = parts[3]  # cross_attention/self_attention
            grouped_layers[group][layer].add(attention_type)

    # Creare una rappresentazione compatta
    compact_representation = []
    for group, layers in grouped_layers.items():
        for layer, attention_types in layers.items():
            compact_representation.append(
                f"{group} layer {layer}: {', '.join(sorted(attention_types))}"
            )

    return compact_representation


def load_parameters(model: nn.Sequential, filename: str, _debug: bool=False) -> None:
    '''
    Given a model, loads the parameters from file.

    Args:
        model (nn.Sequential): eeeee

    Iterates through model's parameters with `parameters()`
    having `with_names=True` so that they can be used as a key
    to read the respective data in the state dict.
    This allow to big degrees of freedom with the model 
    architecture and naming: as long as the params' name
    on both the state dict and the model are the same,
    this function can work properly.
    '''
    # Itero sul modello, prendo i nomi dei parametri le sue chiavi e le uso come
    # chiavi nel dict letto da file per prendere la roba.
    # Il nome e le shape devono essere uguali
    model_state_dict: dict[str, np.ndarray]
    model_state_dict = _load_safetensors(filename)
    
    #assert len(model_state_dict) == len(model.state_dict())

    if _debug:
        curr_layers, saved_layers = set(model.state_dict().keys()), set(model_state_dict.keys())
        different_layers = curr_layers.symmetric_difference(saved_layers)

        processed_layers = __process_layer_keys(different_layers)
        if processed_layers:
            warnings.warn(
                f"Model and saved state have mismatched layers. Different layers:\n"
                + "\n".join(processed_layers) +
                "\nThis could be normal if the model uses dynamic configurations, "
                "like the MHA in transformers."
            )
    
    for name, param in model.parameters(with_names=True):
        try:
            if param.data.shape != model_state_dict[name].shape:
                warnings.warn(
                    f"\nParam {name} has mismatching shape. In the given "
                    f"model has shape {param.shape()}, but in the state "
                    f"dict is of shape {model_state_dict[name].shape}. "
                )
            param.data = model_state_dict[name]
            # per evitare incongruenze tra .device e type(.data)
            param.to(param.device, inplace=True)
        except KeyError as e:
            if _debug:
                print(
                    f"Parametro {name} non trovato all'interno dello state dict "
                    "letto da file. Forse era None nel modello originale."
                )
            continue
        

def load_architecture(filename: str, experimental_funct: bool=True, allow_eval: bool=False) -> str|nn.Sequential:
    '''
    Load a model architecture from a .safetensors' file metadata.
    Lazy load allows to read it without loading all the tensors
    contained in the file.

    Args:
        filename (str): file from whom load the architecture
        experimental_funct (bool): if True, allow to process the
            model repr string with `build_from_repr()`,
            a custom string parser built to recreate models from their
            repr without using `eval()` or other unsafe operations.
            Can contain bugs.
        allow_eval (bool): flag to allow the evaluation of the model
            repr. This solution is adopted only if `experimental_funct`
            is False. The string representing the architecture
            will be printend and confirmation will be required before
            evaluation. Then, a choice can be made to load the associated
            params or to return the randon initialized model.

    NOTE: the architecture string can be seen directly from the
    .safetensors file without executing or printing anything.
    '''

    with safe_open(f"{filename}.safetensors", framework="np") as f:
        try:
            model_repr = f.metadata()["model_repr"]
        except KeyError:
            # old repr
            model_repr = f.metadata()["architecture"]

    if experimental_funct:
        model = build_from_repr(model_repr)
        load_parameters(model, f"{filename}.safetensors")
        return model

    if not allow_eval:
        return model_repr
    
    print(f"Stringa dell'architettura:\n{model_repr}")
    choice = input("Confermi di voler eseguirla? [Y/n]\n").lower()

    if choice == 'y':
        model = eval(model_repr)
        choice = input("Carico gia' i parametri? [Y/n]\n").lower()

        if choice == "y":
            load_parameters(model, filename)
        return model
    else:
        return model_repr


def load_optimizer_state(filename: str, model_params: list):
    '''
    Returns an optimizer with the same state as the one saved
    in the `.safetensors` file.
    Args:
        filename (str): name of the `.safetensors` file containing the
            optimizer data (usually in the form of `{filename}_optimizer.safetensors`)
        model_params (list of Tensor): parameters of the model to optimize
    '''

    with safe_open(f"{filename}_optimizer.safetensors", framework="np") as f:
        non_array_attrs = f.metadata()

    optimizer_class = getattr(nn, non_array_attrs.pop("optimizer_cls"), None)

    if optimizer_class is None:
        raise RuntimeError(
            f"Optimizer {optimizer_class} not found. Maybe it was refactored?"
        )

    args = {}                   # parameters to give at __init__
    internal_state = {}         # internal variables state
    array_attrs = _load_safetensors(f"{filename}_optimizer.safetensors")

    attrs = {**non_array_attrs, **array_attrs}

    for key, value in attrs.items():
        category, attr_name = key.split(".", 1)
        
        match category:
            case "arg":
                args[attr_name] = _parse_arg(value)
            case "state":
                # prendo il nome della lista di appartenenza e l'index
                match = re.match(r"(.+)<(\d+)>", attr_name)

                if match:
                    list_name = match.group(1)  # Nome dell'attributo prima di `<`
                    index = int(match.group(2))  # Indice tra `<` e `>`

                    if list_name not in internal_state:
                        internal_state[list_name] = []

                    # Espandi la lista associata all'attributo se necessario
                    curr_len = len(internal_state[list_name])
                    if curr_len <= index:
                        internal_state[list_name].extend([None] * (index + 1 - curr_len))

                    # Posiziona il valore nella lista associata
                    internal_state[list_name][index] = value
                else:
                    # e' un attributo interno non composto da tensori
                    #internal_state[f"{attr_name}<simple>"] = _parse_arg(value)
                    internal_state[f"{attr_name}"] = _parse_arg(value)
                    #raise RuntimeError()
            case _:
                raise RuntimeError()

    optimizer = optimizer_class(model_params, **args)
    for name, value in internal_state.items():
        try:
            # se value e' una lista di np.ndarray
            if isinstance(value, list) and all(isinstance(item, np.ndarray) for item in value):
                # allora era una lista di Tensors
                for i in range(0, len(value)):
                    # questo cast mi porta anche value[i] su device
                    value[i] = Tensor(value[i])

            setattr(optimizer, name, value)
        except AttributeError:
            print(f"Class {optimizer.__class__} has no attribute named {name}.")
            exit()
    return optimizer
            

def load_checkpoint(filename: str, device: str="cpu"):
    model = load_architecture(filename)

    # non funziona. I parametri risultano tutti su device,
    # ma la maggior parte ha il .data che e' un np.array
    if device:
        model.to(device)

    #for name, param in model.parameters(trainable_only=False, with_names=True):
        #param.to(device)
        #print(name, param.device, type(param.data))

    optimizer = load_optimizer_state(filename, model.parameters())

    return model, optimizer


def preview_safetensor(filename: str) -> None:
    '''
    Given a safetensor file, safely shows its content.

    Args:
        filename (str): filename of model, extension included.

    Example::

        >>> preview_safetensor("model.safetensors")
        Tensors         Shape           Precision
        linear0.bias    (10,)           float32
        linear0.weights (10, 10)        float32
        linear2.bias    (20,)           float32
        linear2.weights (10, 20)        float32
    '''
    print("{: <15} {: <15} {: <15}".format("Tensors", "Shape", "Precision"))

    with safe_open(filename, framework="np") as f:
        for key in f.keys():
            print("{: <15} {: <15} {: <15}".format(str(key), str(f.get_tensor(key).shape), str(f.get_tensor(key).dtype)))