import torch
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
import torchic
from torchic import nn, Tensor

VOCAB_SIZE = 4
D_MODEL = 2


myTen = torchic.rand((VOCAB_SIZE, ), dtype=torchic.int64)
torTen = torch.tensor(myTen.data)

torEmb = torch.nn.Embedding(VOCAB_SIZE, D_MODEL)
myEmb = nn.Embedding(VOCAB_SIZE, D_MODEL, _weights = Tensor(torEmb.weight.detach().numpy(), requires_grad=True))

myOut = myEmb(myTen)
torOut = torEmb(torTen) 

myOut.mean().backward()
torOut.mean().backward()

print(myOut)
print(torOut)
print("---")
print(myEmb.weights.grad)
print(torEmb.weight.grad)
