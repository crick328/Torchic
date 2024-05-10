# ha ragione il bro di stackoverlow: non sta alla classe Tensor il cambio dei device
# di tutte le istanze, ma a chi li contiene (layer e modelli!)
import torchic
from torchic import nn
import timeit
import cupy as cp
import numpy as np
import torch
import time


a = torchic.Tensor([1,2], requires_grad=True)
b = torchic.Tensor([4,1], requires_grad=True)
d = torchic.Tensor([0,7], requires_grad=True)

c = a + b
e = c + d

c.requires_grad = False
e.mean().backward()

print(a.grad)











'''
t1 = timeit.timeit('a@b', number=1, setup="import cupy as cp; a=cp.random.normal(loc=0.0, scale=1, size=(10000, 10000)); b=cp.random.normal(loc=0.0, scale=1, size=(10000, 10000))")
print(t1)

t1 = timeit.timeit('a@b', number=1, setup="import numpy as np; a=np.random.normal(loc=0.0, scale=1, size=(10000, 10000)); b=np.random.normal(loc=0.0, scale=1, size=(10000, 10000))")
print(t1)
'''



"""
dummy = timeit.timeit('test3=test1.matmul(test2); test5=test3+test4; test5.mean().backward()', number=1, setup="import test_cuda; test1=test_cuda.Tensor([1,2,3], True, device='cpu'); test2=test_cuda.Tensor([[1,2,3],[1,2,3],[1,2,3]], True, device='cpu'); test4=test_cuda.Tensor([2,2,2], True, device='cpu');")
print(dummy) # 
dummy = timeit.timeit('test3=test1.matmul(test2); test5=test3+test4; test5.mean().backward()', number=1, setup="import test_cuda; test1=test_cuda.Tensor([1,2,3], True, device='cuda'); test2=test_cuda.Tensor([[1,2,3],[1,2,3],[1,2,3]], True, device='cuda'); test4=test_cuda.Tensor([2,2,2], True, device='cuda');")
print(dummy) # 

t1 = timeit.timeit('test3=test1.matmul(test2); test5=test3+test4; test5.mean().backward()', number=10000, setup="import test_cuda; import numpy as np; device='cpu'; test1=test_cuda.Tensor(np.random.random((1,3000)), True, device=device); test2=test_cuda.Tensor(np.random.random((3000,1000)), True, device=device); test4=test_cuda.Tensor(np.random.random((1,1000)), True, device=device);")
print(t1) #

t0 = timeit.timeit('og3=og1.matmul(og2); og5=og3+og4; og5.mean().backward()', number=10000, setup="import torchic; import numpy as np; device='cpu'; og1=torchic.cuda_tensor.Tensor(np.random.random((1,3000)), True, device=device); og2=torchic.cuda_tensor.Tensor(np.random.random((3000,1000)), True, device=device); og4=torchic.cuda_tensor.Tensor(np.random.random((1,1000)), True, device=device)")
print(t0) #
"""

#t2 = timeit.timeit('grad=grad+a', number=10000000, setup='import test_cuda; a=test_cuda.Tensor([1,2,3]); grad=None')
#print(t2)
#t3 = timeit.timeit('grad=grad+a if grad is not None else a', number=10000000, setup='import torchic; a=torchic.Tensor([1,2,3]); grad=None')
#print(t3)
