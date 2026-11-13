import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic import nn, Tensor
from torchic.tools.helpers import rand

import torch


my_x = rand([5,], requires_grad=True)
my_w = rand([5,4], requires_grad=True)
my_b = rand([4,], requires_grad=True)

tor_x = torch.tensor(my_x.data, requires_grad=True)
tor_w = torch.tensor(my_w.data, requires_grad=True)
tor_b = torch.tensor(my_b.data, requires_grad=True)


my_out = my_x.matmul(my_w) + my_b
tor_out = tor_x.matmul(tor_w) + tor_b

#print(my_out)
#print(tor_out)

my_out.mean().backward()
tor_out.mean().backward()

print(my_x.grad)
print(tor_x.grad)
