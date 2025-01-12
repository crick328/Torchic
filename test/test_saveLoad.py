import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
from torchic import nn, Tensor, rand, set_default_device
from torchic.nn.models.save_load import load_architecture, save, preview_safetensor, load_checkpoint
from torchic.tools.code_analysis.mem import get_nbytes
import numpy as np

#import datetime
#now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#set_default_device("cpu")

def find_in_layer(layer, np_array, _name=''):
    if isinstance(layer, nn.ModuleList):
        for module in layer:
            find_in_layer(module, np_array, f"{_name}.ModuleList")
        return
    
    for l_name, l_param in layer:
        if l_name == "_init_params":
            continue

        if isinstance(l_param, nn.Module):
            find_in_layer(l_param, np_array, f"{_name}.{l_name}")

        elif isinstance(l_param, Tensor):
            if l_param.shape() == np_array.shape:
                if np.allclose(l_param.data, np_array):
                    print(f"TROVATO: {_name}.{l_name}")
                    print(l_param)


# TROVATO IL PROBLEMA: lo state_dict non salva separatamente i
# layers nei ModuleList, ma anzi sovrascrive con l'ultimo valore!
# Ecco perche' la funzione di load non ha avuto bisogno di fix
# ed ecco perche' i file dell'optimizer pesavano molto di piu'...
# Ma allora i file dell'optimizer non dovrebbero pesare ancora di piu'?
#model = nn.create_transformer(2, 5, 4, 3, 0.1, 4, 2, 10)
model = nn.TransformerV2(
    d_model=2,
    encoder_vocab_size=5,
    dec_vocab_size=4,
    max_len=3,
    drop_prob=0.1,
    n_layers=4,
    head_num=2,
    dim_feed_forward=10
    )
#model = nn.create_transformer(512, 15698, 22463, 310)
optimizer = nn.AdamV3(model.parameters())
print(repr(model))


#exit()
#print(model.state_dict())




'''
state_dict = model.state_dict()

# .parameters() e .state_dict() sembrano dare tensori diversi
# Il problema non e' la flag with_names
# Il Sequential di Mnist non riscontra questo problema
for (param_name, param), (state_name, state_tensor) in zip(
    model.parameters(with_names=True, trainable_only=True),
    model.state_dict().items()):

    found = False
    for k, v in state_dict.items():
        if v.shape == param.data.shape and np.allclose(v, param.data):
            print(f"Tensor trovato nello state dict: {k}\n")
            found = True
            break

    if not found:
        print("Warning: tensore NON trovato nello state dict!\n")
        print("Provo a richiamare .parameters() per vedere se riesce...")

        # il tensore riesce in model.parameters() anche se richiamo state_dict
        _ = model.state_dict()

        for p in model.parameters():
            if p.data.shape == param.data.shape and np.allclose(p.data, param.data):
                print(f"Tensor trovato in .parameters()\n")
                found = True
                break

        if not found:
            print("WARNING: tensore NON ritrovato in .parameters()!\n")
        else:
            print("Tensore ritrovato. Ora lo cerco nel modello")

            for layer in model:
                find_in_layer(layer, param.data, layer.__class__.__name__)

    for layer in model:
        find_in_layer(layer, state_tensor.data, layer.__class__.__name__)

    assert param_name == state_name, f"Nome diverso: {param_name} != {state_name}"
    assert np.allclose(param.data, state_tensor), (
        f"Valori diversi.\n.parameters():\n{param_name}: {param.data}\n.state_dict():\n{state_name}: {state_tensor}"
    )

exit()
'''

save(model, optimizer)
print("Modello salvato")



loaded_model, loaded_optimizer = load_checkpoint("checkpoint", device="cuda")

count = 0
for p, loaded_p in zip(model.parameters(with_names=True, trainable_only=False), loaded_model.parameters(with_names=True, trainable_only=False)):
    print(count, loaded_p[1].device)
    assert p[1]._device.allclose(p[1].data, loaded_p[1].data), (
        f"\n{p[0]}{p[1].data}\nLOADED:\n{loaded_p[0]}{loaded_p[1].data}"
        )
    count+=1

for slot in optimizer.__slots__:
    attr, loaded_attr = getattr(optimizer, slot), getattr(loaded_optimizer, slot)

    if isinstance(attr, list):
        for arr, loaded_arr in zip(attr, loaded_attr):
            assert np.allclose(arr.data, loaded_arr.data)
    else:
        assert attr == loaded_attr

preview_safetensor("checkpoint.safetensors")