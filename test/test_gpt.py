import re
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 1.5B")
#import torchic
from torchic import Tensor
from torchic import nn


def parse_model_string(model_str):
    # Rimuovi spazi bianchi superflui e le nuove linee
    model_str = model_str.replace('\n', '').replace(' ', '')
    
    # Trova il tipo del modello e il contenuto all'interno delle parentesi
    match = re.match(r'nn\.(\w+)\((.*)\)', model_str)
    if not match:
        return None

    model_type = match.group(1)
    layers_str = match.group(2)
    
    # Trova i layer definiti all'interno della stringa
    layer_matches = re.findall(r'nn\.\w+\([^)]*\)', layers_str)
    
    layers = []
    for layer in layer_matches:
        # Estrai il nome del layer e i suoi argomenti
        layer_name_match = re.match(r'nn\.(\w+)\((.*)\)', layer)
        if layer_name_match:
            layer_name = layer_name_match.group(1)
            layer_args = layer_name_match.group(2)
            
            # Crea un dizionario con il nome del layer e i suoi argomenti
            layer_info = {'layer': layer_name, 'args': layer_args}
            layers.append(layer_info)
    
    return {'model_type': model_type, 'layers': layers}


# questo e' decisamente migliorabile, ma ha soluzioni interessanti
def create_layer(layer_dict):
    layer_name = layer_dict['layer']
    args_str = layer_dict['args']
    
    # Converti gli argomenti della stringa in un dizionario
    args_dict = {}
    if args_str:
        # Usa una regex per trovare le coppie chiave=valore e gli argomenti posizionali
        args_matches = re.findall(r'(\w+)=([^,]+)|([^,]+)', args_str)
        for match in args_matches:
            if match[0]:  # Se è una coppia chiave=valore
                key = match[0]
                value = match[1]
                try:
                    # Converti il valore in un tipo Python appropriato
                    # NOTE usa eval, ma non e' un problema. Se c'e' un . e' un
                    # float e uso float(), senno' int()
                    args_dict[key] = eval(value)
                except:
                    args_dict[key] = value
            elif match[2]:  # Se è un argomento posizionale
                pos_arg = match[2]
                try:
                    # Converti il valore in un tipo Python appropriato
                    args_dict[len(args_dict)] = eval(pos_arg)
                except:
                    args_dict[len(args_dict)] = pos_arg
    
    # NOTE: cosa MOLTO intelligente rispetto alla mia idea iniziale
    #n dello switch
    # Ottieni il costruttore del layer da torch.nn
    layer_class = getattr(nn, layer_name, None)
    if not layer_class:
        raise ValueError(f"Layer {layer_name} not found in torch.nn")

    # Costruisci l'oggetto layer
    # Separiamo gli argomenti posizionali da quelli a chiave
    positional_args = [args_dict[i] for i in args_dict if isinstance(i, int)]
    keyword_args = {k: v for k, v in args_dict.items() if isinstance(k, str)}
    #print(layer_class(*positional_args, **keyword_args))
    
    return layer_class(*positional_args, **keyword_args)

def build_model(model_str):
    model_dict = parse_model_string(model_str)
    if not model_dict:
        raise ValueError("Invalid model string")
    
    model_type = model_dict['model_type']
    layers_dict = model_dict['layers']
    layers = [create_layer(layer_dict) for layer_dict in layers_dict]
    print(layers[0])
    print(layers)
    print(*layers)
    
    if model_type == 'Sequential':
        return nn.Sequential(*layers)
    else:
        raise ValueError(f"Model type {model_type} not supported")

# Esempio di utilizzo
model_str = """
nn.Sequential(
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 20, bias=True),
    nn.Softmax(dim=1),
)
"""

model = build_model(model_str)
#print(model.linear0.weights)
layer = nn.Linear(weights = (10, 10), bias = (10,))