import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
import torchic
import torch

ea = torchic.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, dtype=torchic.float32)
ta = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, dtype=torch.float32)

for x in ta:
    print(x)