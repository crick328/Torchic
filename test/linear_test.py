import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torchic
from torchic import nn, Tensor
from torchic.nn.models.transformer import MultiHeadAttention
from torchic.tools.helpers import zeros_like, rand
import torch


my_ten = rand([5,], requires_grad=True)
tor_ten = torch.tensor(my_ten.data, requires_grad=True)

#my_lin = nn.Linear(5,5)
tor_lin = torch.nn.Linear(5,5)
my_lin = nn.Linear(5,5,
                   _weights = Tensor(tor_lin.weight.detach().numpy(), requires_grad=True),
                   _bias = Tensor(tor_lin.bias.detach().numpy(), requires_grad=True)
                   )

my_out = my_lin(my_ten)
tor_out = tor_lin(tor_ten)

print(my_out)
print(tor_out)

my_out.mean().backward()
tor_out.mean().backward()

print(my_ten.grad)
print(tor_ten.grad)
