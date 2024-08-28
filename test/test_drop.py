import torch
from torch import nn
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.3")
import torchic
import numpy as np


data = np.random.rand(2, 2, 2)

mydrop = torchic.nn.Dropout(0.5)
tdrop = nn.Dropout(0.5)

myt = torchic.Tensor(data, True)
tt = torch.tensor(data, requires_grad=True)

#print(f"myt: {myt}")
#print(f"t: {tt}\n")

myres = mydrop(myt)
tres = tdrop(tt)

print(f"myres: {myres}")
print(f"tres: {tres}\n")

myres.mean().backward()
tres.mean().backward()

print(f"myt grad: {myt.grad}")
print(f"tt grad: {tt.grad}\n")