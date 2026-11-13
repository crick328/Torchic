import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic.cuda_tensor import Tensor
import torch
from torch import tensor
import numpy as np


MIN = -1
MAX = 1
torch.set_default_dtype(torch.float32)



data1 = np.random.uniform(MIN, MAX, (10,20)).astype(np.float32)
data2 = np.random.uniform(MIN, MAX, (20,10)).astype(np.float32)
data3 = np.random.uniform(MIN, MAX, 10).astype(np.float32)
data4 = np.random.uniform(MIN, MAX, 10).astype(np.float32)
data5 = np.random.uniform(MIN, MAX, 10).astype(np.float32)
data6 = np.random.uniform(MIN, MAX, 10).astype(np.float32)


my_tensor1 = Tensor(data1, requires_grad=True)
my_tensor2 = Tensor(data2, requires_grad=True)
my_tensor3 = Tensor(data3, requires_grad=True)
my_tensor4 = Tensor(data4, requires_grad=True)
my_tensor5 = Tensor(data5, requires_grad=True)
my_tensor6 = Tensor(data6, requires_grad=True)


torch_tensor1 = tensor(data1, requires_grad=True)
torch_tensor2 = tensor(data2, requires_grad=True)
torch_tensor3 = tensor(data3, requires_grad=True)
torch_tensor4 = tensor(data4, requires_grad=True)
torch_tensor5 = tensor(data5, requires_grad=True)
torch_tensor6 = tensor(data6, requires_grad=True)



my_a = my_tensor1.matmul(my_tensor2).mean(0)
my_b = my_a + my_tensor3
my_c = my_b / my_tensor4
my_d = - my_c
my_e = my_d.__pow__(3)
my_f = my_e - my_tensor5
my_g = my_f * my_tensor6
my_h = my_g.std(-1)
#my_i = my_h.std()


torch_a = torch_tensor1.matmul(torch_tensor2).mean(0)
torch_b = torch_a + torch_tensor3
torch_c = torch_b / torch_tensor4
torch_d = -torch_c
torch_e = torch_d.pow(3)
torch_f = torch_e - torch_tensor5
torch_g = torch_f * torch_tensor6
torch_h = torch_g.std(-1)
#torch_i = torch_h.std()

my_h.mean().backward()
torch_h.mean().backward()


assert torch.allclose(tensor(my_a.data), torch_a)
assert torch.allclose(tensor(my_b.data), torch_b)
assert torch.allclose(tensor(my_c.data), torch_c)
assert torch.allclose(tensor(my_d.data), torch_d)
assert torch.allclose(tensor(my_e.data), torch_e)
assert torch.allclose(tensor(my_f.data), torch_f)
assert torch.allclose(tensor(my_g.data), torch_g)
assert torch.allclose(tensor(my_h.data), torch_h)
#assert torch.allclose(tensor(my_i.data), torch_i)

assert torch.allclose(tensor(my_tensor6.grad), torch_tensor6.grad)
assert torch.allclose(tensor(my_tensor5.grad), torch_tensor5.grad)
assert torch.allclose(tensor(my_tensor4.grad), torch_tensor4.grad)
assert torch.allclose(tensor(my_tensor3.grad), torch_tensor3.grad)
assert torch.allclose(tensor(my_tensor2.grad), torch_tensor2.grad)
assert torch.allclose(tensor(my_tensor1.grad), torch_tensor1.grad)