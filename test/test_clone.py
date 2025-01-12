import torch
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
from torchic import Tensor, rand
import numpy as np



my_tensor = rand((2,2), requires_grad=True)
torch_tensor = torch.tensor(my_tensor.data, requires_grad=True)

my_clone = my_tensor.clone()
torch_clone = torch_tensor.clone()


# dimostro che entrambi i tensori sono indipendenti, ma accumulano
# sullo stesso gradiente
torch_result = torch_clone * -1
torch_result2 = torch_tensor.sqrt()
torch_result.mean().backward()
torch_result2.mean().backward()

my_result = my_clone * Tensor(-1)
my_result2 = my_tensor.sqrt()
my_result.mean().backward()
my_result2.mean().backward()

print(my_result2)
print(torch_result2)


assert np.allclose(my_result.data, torch_result.detach().numpy())
assert np.allclose(my_result2.data, torch_result2.detach().numpy())
assert np.allclose(my_tensor.grad, torch_tensor.grad.numpy())