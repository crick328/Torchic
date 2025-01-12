import torch
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
import torchic


my_tensor = torchic.rand((2,2), requires_grad=True)
torch_tensor = torch.tensor(my_tensor.data, requires_grad=True)

print("TEST TORCH")
torch_result = torch_tensor.sqrt()
torch_detached = torch_result.detach()

import numpy as np
# cambia anche l'originale
print(torch_result)
#torch_detached.requires_grad=True
#torch_detached += np.ones(torch_detached.shape)#torch.ones(torch_detached.shape, requires_grad=False)
torch_detached[0] += 2
print(torch_result)


print("\nTEST TORCHIC")
my_result = my_tensor.sqrt()
my_detached = my_result.detach()

print(my_result)
#my_detached += torchic.ones(my_detached.shape())
my_detached.data[0] += 2
print(my_result)