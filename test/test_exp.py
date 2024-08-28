import torch
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.3")
import torchic

myten = torchic.rand([2,3], requires_grad=True)
tten = torch.tensor(myten.data, requires_grad=True, dtype=torch.float32)
tten.retain_grad()

mysqrt = myten.exp()
tsqrt = tten.exp()

mysqrt.mean().backward()
tsqrt.mean().backward()

print(mysqrt)
print(tsqrt)
print("-----")
print(myten.is_leaf)
print(mysqrt.is_leaf)
print(myten.grad)
print(tten.grad)
