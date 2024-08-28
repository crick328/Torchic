import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.3")
from torchic import nn
from torchic.nn.models.save_load import load_architecture, save_model, preview_safetensor
from torchic.tools.mem import get_nbytes


model = nn.Transformer(2, 5, 4, 3, 0.1, 2, 2, 10)
#model = nn.Transformer(512, 15698, 22463, 310)
#print(model)
#print(model.state_dict())
nbytes = 0
for param in model.parameters():
    nbytes += get_nbytes(param)
print(nbytes /1024 )
save_model(model)
print("Modello salvato")


model=load_architecture("model", allow_eval=True)
print("Modello ricaricato con successo:")
preview_safetensor("model.safetensors")