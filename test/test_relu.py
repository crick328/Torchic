import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
import torchic
import torch

my_ten = torchic.rand([2,3,2], requires_grad=True)
tor_ten = torch.tensor(my_ten.data, requires_grad=True)

my_relu = torchic.nn.activations.ReLU()
tor_relu = torch.nn.ReLU()

my_out = my_relu(my_ten)
tor_out = tor_relu(tor_ten)

print(my_out)
print(tor_out)

my_out.mean().backward()
tor_out.mean().backward()

print(my_ten.grad)
print(tor_ten.grad)