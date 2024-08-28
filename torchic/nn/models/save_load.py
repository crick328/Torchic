from safetensors import safe_open
from safetensors.numpy import save_file
from torchic.nn.models.test_myparser import build_from_repr
from torchic import nn  # usato in caso di eval



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


def save_model(model: nn.Sequential, filename: str="model") -> None:
    '''
    Il metodo .state_dict() restituisce un dizionario del tipo
    {
        'linear0.weights': [numpy array],
        'linear0.bias': [numpy array]
        ...
    }
    ed e' cio' che viene salvato nel core di save_file().
    In piu', i metadata salvano la repr() del modello
    in modo tale da poterlo ricostruire (senza parametri).
    '''
    # I metadata saranno usati per i modelli che sono checkpoint per optimizers e co.
    metadata = {
        # qua puoi mettere solo stringhe
        "architecture": repr(model),
        "optimizer": "optimizer.repr()",
        "metrics": "metrics.repr()"
        }
    save_file(model.state_dict(), f"./{filename}.safetensors", metadata=metadata)


def load_parameters(model: nn.Sequential, filename: str) -> None:
    '''
    Given a model, loads the parameters from file.
    It can be used indipendently from other functions.
    '''
    # popolo il modello che mi dai con la mia roba
    # Itero state_dict, prendo le sue chiavi e le uso come
    # chiavi nel dict letto da file per prendere la roba.
    # Il nome e le shape devono essere uguali
    
    model_state_dict = _load_safetensors(filename)
    
    assert len(model_state_dict) == len(model.state_dict())
    
    try:
        for param_name, model_param in model.parameters(with_names=True):
            model_param.data = model_state_dict[param_name]
    except AttributeError as e:
        print(f"{e}.\n"
              "Il modello salvato ha gli identificatori che non "
              "rispettano lo standard <ClassName>.lower()<N>. "
              "Il modello puo' ancora essere caricato in memoria, "
              "basta modificare il nome degli identificatori "
              "nel 'data' del file .safetensors.")
        
    


def load_architecture(filename: str, experimental_funct: bool=True, allow_eval: bool=False) -> str|nn.Sequential:
    '''
    Load a model architecture from a .safetensors' file metadata.
    Lazy load allows to read it without loading all the tensors
    contained in the file.
    If `allow_eval=True`, the string representing the architecture
    will be printend and confirmation will be required before
    evaluation. Then, a choice can be made to load the associated
    params or to return the randon initialized model.

    NOTE: the architecture string can be seen directly from the
    .safetensors file without executing or printing anything.
    '''
    with safe_open(f"./{filename}.safetensors", framework="np") as f:
        arch = f.metadata()["architecture"]

    if experimental_funct:
        model = build_from_repr(arch)
        load_parameters(build_from_repr(arch), f"./{filename}.safetensors")
        return model

    if not allow_eval:
        return arch
    
    print(f"Stringa dell'architettura:\n{arch}")
    choice = input("Confermi di voler eseguirla? [Y/n]\n").lower()

    if choice == 'y':
        model = eval(arch)
        choice = input("Carico gia' i parametri? [Y/n]\n").lower()

        if choice == "y":
            load_parameters(model, filename)
        return model
    else:
        return arch


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