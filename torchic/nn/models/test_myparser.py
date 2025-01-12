import re
#import sys
#sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
from torchic import nn


def _parse_model_repr(model_str):
    # Rimuovi spazi bianchi superflui e le nuove linee
    model_str = model_str.replace('\n', '').replace(' ', '')
    
    # Trova il tipo del modello e il contenuto all'interno delle parentesi
    match = re.match(r'nn\.(\w+)\((.*)\)', model_str)
    if not match:
        raise RuntimeError("AAA")
    
    # prendo le stringhe dei layers e li divido. Ogni cella e' una stringa
    # completa del layer
    layer_matches = re.findall(r'nn\.\w+\([^)]*\)', match.group(2))
    layers = []

    for layer in layer_matches:
        layer_name_match = re.match(r'nn\.(\w+)\((.*)\)', layer)

        if layer_name_match:
            cls = layer_name_match.group(1)
            args = layer_name_match.group(2)
            
            # Crea un dizionario con il nome del layer e i suoi argomenti
            layers.append({'class': cls, 'args': args})
        else:
            raise RuntimeError("AAA")
    
    return {'model_type': match.group(1), 'layers': layers}


def _is_number(string):
    # tengo conto di segni e notazioni scientifiche
    # tolgo il - iniziale, le "e" della notaz. scientifica, la virgola
    # ed un eventuale segno - dell'esponente. Dopodioche', chiedo
    # se cio' che rimane e' un numero o no
    return string.lstrip('-').replace('e','',1).replace('-','',1).replace('.','',1).isdigit()


def _parse_arg(arg: str):
    if _is_number(arg):
        # se ha virgola o notazione esponenziale con esp. negativo
        if "." in arg or "e-" in arg:
            return float(arg)
        else:
            return int(arg)
    elif arg.isalpha():
        return arg
    else:
        raise ValueError(f"Arg type non riconosciuto: {arg}")


def create_layer(layer_dict):
    layer_class = getattr(nn, layer_dict['class'], None)

    if not layer_class:
        raise ValueError(f"Layer {layer_dict['class']} not found in torchic.nn"
                        " (hai dimenticato di importarlo dentro l'init di nn)")
    
    # se non ha args
    if not layer_dict["args"]:
        return layer_class()
    
    args_list = layer_dict["args"].split(",")
    positional_args = []
    keyword_args = {}

    for arg in args_list:
        keyword = None
        # se e' un arg con parametro
        if "=" in arg:
            keyword, arg = arg.split("=")
        parsed_arg = _parse_arg(arg)

        if keyword:
            keyword_args[keyword] = parsed_arg
        else:
            positional_args.append(parsed_arg)
        
    return layer_class(*positional_args, **keyword_args)


def build_from_repr(model_repr: str):
    '''
    Custom function to build models without using eval.

    Example::

        >>> model_str = """
        nn.Sequential(
            nn.Linear(10, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 20, bias=True),
            nn.Softmax(dim=1),
        )"""
        >>> model = build_model(model_str)
        >>> print(model)
        nn.Sequential(
            nn.Linear(10, 10, bias=True),
            nn.ReLU(),
            nn.Linear(10, 20, bias=True),
            nn.Softmax(dim=1)
        >>> print(model.linear0.weights)
        Tensor([
            [-0.2736,  0.0748, -0.0047],
            [-0.4336, -1.5017, -1.083 ]
        ], requires_grad=True)
    '''
    model_dict = _parse_model_repr(model_repr)
    model_class = getattr(nn, model_dict['model_type'], None)

    if not model_class:
        raise ValueError(f"Model {model_class['class']} not found in torchic.nn")
    
    if issubclass(model_class, nn.Sequential):
        # ha bisogno di ricevere i suoi layers al momento dell'init
        layers = [create_layer(layer) for layer in model_dict["layers"]]
        return model_class(*layers)
    elif issubclass(model_class, nn.Module):
        pattern = r"(\w+)=([\w\.\-]+)"
        matches = re.findall(pattern, model_repr)

        parameters = {key: _parse_arg(value) for key, value in matches}
        return model_class(**parameters)
    else:
        raise RuntimeError(
            "Ma che modello e'?"
        )
    
    


# Esempio di utilizzo
'''
model_str = """
nn.Sequential(
    nn.Linear(2, 3, bias=True),
    nn.ReLU(),
    nn.Linear(3, 2, bias=True),
    nn.Softmax(dim=1),
)
"""

model = build_model(model_str)
print(model)
print(model.linear0.weights)
'''