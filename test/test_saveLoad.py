import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 1.5B")
#import torchic
from torchic import Tensor
from torchic import nn
from torchic import set_defalt_dtype, float64, float32

set_defalt_dtype(float32)


linear = nn.Linear(10,10)

#from safetensors.numpy import save_file, load_file
#from safetensors.torch
#from safetensors import safe_open
from torchic.nn.models.utils import load_architecture, save_model


dd = nn.Sequential(
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,20),
    nn.Softmax(1)
)

#model=load_architecture("model.safetensors",allow_eval=True)
#print(repr(model))



import re

def parse_model_string(model_str):
    # Rimuovi spazi bianchi superflui e le nuove linee
    model_str = model_str.replace('\n', '').replace(' ', '')
    
    # Trova il contenuto all'interno di nn.Sequential(...)
    match = re.search(r'nn.Sequential\((.*)\)', model_str)
    if not match:
        return []

    layers_str = match.group(1)
    
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
    
    return layers

# Esempio di utilizzo
model_str = """
nn.Sequential(
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 20, bias=True),
    nn.Softmax(dim=1),
)
"""

parsed_layers = parse_model_string(model_str)
for layer in parsed_layers:
    print(layer)