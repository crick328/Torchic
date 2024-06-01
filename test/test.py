# ha ragione il bro di stackoverlow: non sta alla classe Tensor il cambio dei device
# di tutte le istanze, ma a chi li contiene (layer e modelli!)
import torchic
import timeit
import cupy as cp
import numpy as np
import torch
import time

from torchic import Tensor
from torchic import nn
from torchic.tools.visualization.graphs import draw_graph


model = nn.Sequential(
    nn.Linear(2, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 100),
    nn.ReLU(),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Softmax(0)
)
res = model(Tensor([1,2]))
draw_graph(model, res, rankdir="TB")

#print(res.grad_fn.ctx[2] is res.data)



res.mean().backward()
#draw_dot(res)
#draw_model_dot(model)

'''
import test_cuda

a = test_cuda.Tensor([[1,2],[1,2]], requires_grad=True)
b = test_cuda.Tensor([[1,2],[1,2]], requires_grad=True)

#a = torch.tensor([[1.,2.],[1.,2.]], requires_grad=True)
#b = torch.tensor([[1.,2.],[1.,2.]], requires_grad=True)

c = a.matmul(b)

#a = a*a
a.data = a.data @ a.data


c.mean().backward()

print(a.grad)
print(b.grad)
'''








#t1 = timeit.timeit('a@b', number=1, setup="import cupy as cp; a=cp.random.normal(loc=0.0, scale=1, size=(10000, 10000)); b=cp.random.normal(loc=0.0, scale=1, size=(10000, 10000))")
#print(t1)

#t1 = timeit.timeit('a@b', number=1, setup="import numpy as np; a=np.random.normal(loc=0.0, scale=1, size=(10000, 10000)); b=np.random.normal(loc=0.0, scale=1, size=(10000, 10000))")
#print(t1)



'''
device = "cpu"

dummy = timeit.timeit('test3=test1.matmul(test2); test5=test3+test4; test5.mean().backward()', number=1, setup="import test_cuda; test1=test_cuda.Tensor([1,2,3], True, device='cpu'); test2=test_cuda.Tensor([[1,2,3],[1,2,3],[1,2,3]], True, device='cpu'); test4=test_cuda.Tensor([2,2,2], True, device='cpu');")
print(dummy) # 
dummy = timeit.timeit('test3=test1.matmul(test2); test5=test3+test4; test5.mean().backward()', number=1, setup="import test_cuda; test1=test_cuda.Tensor([1,2,3], True, device='cuda'); test2=test_cuda.Tensor([[1,2,3],[1,2,3],[1,2,3]], True, device='cuda'); test4=test_cuda.Tensor([2,2,2], True, device='cuda');")
print(dummy) # 

t1 = timeit.timeit('test3=test1.matmul(test2); test5=test3+test4; test5.mean().backward()', number=10000, setup=f"import test_cuda; import numpy as np; device='cpu'; test1=test_cuda.Tensor(np.random.random((1,3000)), True, device=device); test2=test_cuda.Tensor(np.random.random((3000,1000)), True, device=device); test4=test_cuda.Tensor(np.random.random((1,1000)), True, device=device);")
print(t1) #

t0 = timeit.timeit('og3=og1.matmul(og2); og5=og3+og4; og5.mean().backward()', number=10000, setup="import torchic; import numpy as np; device='cpu'; og1=torchic.cuda_tensor.Tensor(np.random.random((1,3000)), True, device=device); og2=torchic.cuda_tensor.Tensor(np.random.random((3000,1000)), True, device=device); og4=torchic.cuda_tensor.Tensor(np.random.random((1,1000)), True, device=device)")
print(t0) #
'''



'''
t1 = timeit.timeit('uno, due, tre, quattro = a["1"], a["2"], a["3"], a["4"]', number=100000000, setup='import numpy as np; a={"1": np.random.random((2000,2000)), "2": np.random.random((2000,2000)),"3": np.random.random((2000,2000)),"4": np.random.random((2000,2000))}')
print(t1)
t2 = timeit.timeit('uno=a["1"]; due=a["2"]; tre=a["3"]; quattro=a["4"]', number=100000000, setup='import numpy as np; a={"1": np.random.random((2000,2000)), "2": np.random.random((2000,2000)),"3": np.random.random((2000,2000)),"4": np.random.random((2000,2000))}')
print(t2)
t3 = timeit.timeit('uno=a[0]; due=a[1]; tre=a[2]; quattro = a[3]', number=100000000, setup='import numpy as np; a=[np.random.random((2000,2000)),np.random.random((2000,2000)),np.random.random((2000,2000)),np.random.random((2000,2000))]')
print(t3)
t4 = timeit.timeit('uno=a[0]; due=a[1]; tre=a[2]; quattro = a[3]', number=100000000, setup='import numpy as np; a=(np.random.random((2000,2000)),np.random.random((2000,2000)),np.random.random((2000,2000)),np.random.random((2000,2000)))')
print(t4)

t4 = timeit.timeit('uno: np.ndarray; due: np.ndarray; tre: np.ndarray; quattro: np.ndarray; uno, due, tre, quattro = a', number=100000000, setup='import numpy as np; a=(np.random.random((2000,2000)),np.random.random((2000,2000)),np.random.random((2000,2000)),np.random.random((2000,2000)))')
print(t4)
'''

