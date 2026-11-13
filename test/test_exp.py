import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
