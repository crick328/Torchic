import numpy as np
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.0")
from torchic import Tensor
from torchic import nn
import torch


r = torch.tensor([1,2])
print(r.dtype)


a = np.array([1.,2.,0.,1.], dtype=np.float32)
print(a.dtype)

print( (np.where(a > 0, 1, 0) * a).dtype )

s = torch.tensor([1.,2.,0.,1.], dtype=torch.float64)
b = torch.tensor(s.data)
print(b.dtype)
print(torch.where(s > 0, 1, 0).dtype)
print( (torch.where(s > 0, 1, 0) * s).dtype )