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
#print(model.state_dict())
#save_model(model)

#preview_safetensor("model.safetensors")
model=load_architecture("model.safetensors",allow_eval=True)
#print(model.linear0.weights)
#d = load_safetensors("model.safetensors", True)

#print(d["metadata"]["architecture"])
#print(d)

#load_params_from_file(model, "model.safetensors")
