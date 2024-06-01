from safetensors import safe_open
from safetensors.numpy import save_file
from torchic.nn.models.models import Sequential
from torchic import nn  # usato in caso di eval


# I metadata saranno usati per i modelli che sono checkpoint per optimizers e co.

def save_model(model, filename="model"):
    metadata = {
        "architecture": repr(model),
        "optimizer": "optimizer.repr()",
        "metrics": "metrics.repr()"
        }
    save_file(model.state_dict(), f"./{filename}.safetensors", metadata=metadata)


# conforme alla sintassi convenzionale in nome e funzione
def load_model(model, filename) -> None:
    # popolo il modello che mi dai con la mia roba
    # Itero il dict, prendo le sue chiavi e le uso come
    # chiavi nel dict letto da file per prendere la roba.
    # Il nome e le shape devono essere uguali
    dict_safetensors = load_safetensors(filename)

    for name in dict_safetensors.keys():
        # name sara' del tipo linear0.weights
        layer, param = name.split(".")
        try:
            if dict_safetensors[name].shape == getattr(model.__dict__[layer], param).shape():
                setattr(getattr(model.__dict__[layer], param), "data", dict_safetensors[name])
            else:
                raise RuntimeError(
                    f"Il parametro {name} ha shape diverse. " 
                    f"Safetensor file: {dict_safetensors[name].shape}"
                    f", modello: {getattr(model.__dict__[layer], param).shape()}"
                )
        except KeyError as e:
            print(f"Il modello dato non ha il parametro {model.__dict__[layer]}.{param}")


# mia variante
def load_architecture(filename, allow_eval=False) -> str|Sequential:
    '''
    Load a model architecture from a .safetensors' file metadata.
    Lazy load allows to read it without loading all the tensors
    contained in the file.
    If `allow_eval=True`, the string representing the architecture
    will be printend and confirmation will be required before
    evaluation. Then, a choice can be made to load the associated
    params or to return te randon initialized model.

    NOTE: the architecture string can be seen directly from the
    .safetensors file without executing or printing anything.
    '''
    with safe_open(filename, framework="np") as f:
        arch = f.metadata()["architecture"]

    if not allow_eval:
        return arch
    else:
        print(f"Stringa dell'architettura:\n{arch}")
        choice = input("Confermi di voler eseguirla? [Y/n]\n").lower()

        if choice == 'y':
            model = eval(arch)
            choice = input("Carico gia' i parametri? [Y/n]\n").lower()

            if choice == "y":
                load_model(model, filename)
            return model
        else:
            return arch


def load_safetensors(filename, load_metadata=False) -> dict[str]:
    result = {"metadata":{}} if load_metadata else {}

    with safe_open(filename, framework="np") as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)

        if load_metadata:
            for k, v in f.metadata().items():
                result["metadata"][k] = v
    return result


def preview_safetensor(filename):
    print("{: <15} {: <15} {: <15}".format("Tensors", "Shape", "Precision"))

    with safe_open(filename, framework="np") as f:
        for key in f.keys():
            print("{: <15} {: <15} {: <15}".format(str(key), str(f.get_tensor(key).shape), str(f.get_tensor(key).dtype)))