import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torchic

myten = torchic.rand([2,3,4], requires_grad=True)
tten = torch.tensor(myten.data, requires_grad=True, dtype=torch.float32)
tten.retain_grad()

mysqrt = myten.sqrt()
tsqrt = tten.sqrt()

print(myten is mysqrt.grad_fn.ctx[0])

mysqrt.mean().backward()
tsqrt.mean().backward()

#print(mysqrt)
#print(tsqrt)
#print("-----")
#print(myten.grad)
#print(tten.grad)


assert torch.allclose(torch.tensor(mysqrt.data), tsqrt)
assert torch.allclose(torch.tensor(myten.grad), tten.grad)
