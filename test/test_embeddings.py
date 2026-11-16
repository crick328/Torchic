import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torchic
from torchic import nn, Tensor
from torchic.nn.initializers import xavier_uniform

VOCAB_SIZE = 4
D_MODEL = 2


myTen = torchic.rand((VOCAB_SIZE, ), dtype=torchic.int64)
torTen = torch.tensor(myTen.data)

torEmb = torch.nn.Embedding(VOCAB_SIZE, D_MODEL)
myEmb = nn.Embedding(VOCAB_SIZE, D_MODEL)
myEmb.weights = Tensor(torEmb.weight.detach().numpy(), requires_grad=True)

myOut = myEmb(myTen)
torOut = torEmb(torTen) 

myOut.mean().backward()
torOut.mean().backward()

print(myOut)
print(torOut)
print("---")
print(myEmb.weights.grad)
print(torEmb.weight.grad)
