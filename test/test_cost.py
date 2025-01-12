import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
from torchic.autograd.cuda.funct_cost import branch_time_estimator
from torchic import Tensor, rand, nn
from cupy import random

a = rand((1000, 1000), requires_grad=True, device="cuda")
b = rand((1000, 1000), requires_grad=True, device="cuda")

c = a.matmul(b)
#act = nn.ReLU()
#c = act(a)
print(a.device)
print(c.device)

print(branch_time_estimator(c))