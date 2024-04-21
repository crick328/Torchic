# ha ragione il bro di stackoverlow: non sta alla classe Tensor il cambio dei device
# di tutte le istanze, ma a chi li contiene (layer e modelli!)
from replica import CudaTensor, nn
import timeit
import cupy as cp
import numpy as np
from types import ModuleType, NoneType
import torch

cazzo = [
        [
            [1,2],
            [1,2],
            [1,2],
            [1,2]
        ],
        [
            [4,4],
            [4,4],
            [4,4],
            [4,4]
        ]
    ]

ass = CudaTensor(cazzo, requires_grad=True)
b = torch.tensor(cazzo, dtype=torch.float32, requires_grad=True)

mia_soft = nn.Softmax(0)
torch_soft = torch.nn.Softmax(0)
c = mia_soft(ass)
d = torch_soft(b)
d.retain_grad()


if c.data.shape == d.shape:
    print("Le due softmax danno un risultato che ha le stesse shape! Tutto regolare!")
else:
    raise RuntimeError("MERDA LE SHAPE NON COINCIDONO")

d.mean().backward()
#print(f"Grad risultato softmax di torch: {d.grad.shape}")
c.mean().backward()


#print("Mean grad")
#print(c.grad)
#print("----")
#print(d.grad)

print("Softmax grad")
print(ass.grad)
print("----")
print(b.grad)





'''
# a quanto pare cupy deve rodarsi e timeit lo cronometra
dummy = timeit.timeit('og3=og1.matmul(og2); og5=og3+og4; og5.mean().backward()', number=1, setup="import replica; og1=replica.cuda_tensor.CudaTensor([1,2,3], True, device='cuda'); og2=replica.cuda_tensor.CudaTensor([[1,2,3],[1,2,3],[1,2,3]], True, device='cuda'); og4=replica.cuda_tensor.CudaTensor([2,2,2], True, device='cuda')")
print("DUMMYYYYYYYY") # 412s su 1M
'''


'''
t1 = timeit.timeit('a@b', number=1, setup="import cupy as cp; a=cp.random.normal(loc=0.0, scale=1, size=(10000, 10000)); b=cp.random.normal(loc=0.0, scale=1, size=(10000, 10000))")
print(t1)

t1 = timeit.timeit('a@b', number=1, setup="import numpy as np; a=np.random.normal(loc=0.0, scale=1, size=(10000, 10000)); b=np.random.normal(loc=0.0, scale=1, size=(10000, 10000))")
print(t1)
'''


'''
t1 = timeit.timeit('test3=test1.matmul(test2); test5=test3+test4; test5.mean().backward()', number=100000, setup="import test_cuda; test1=test_cuda.CudaTensor([1,2,3], True, device='cuda'); test2=test_cuda.CudaTensor([[1,2,3],[1,2,3],[1,2,3]], True, device='cuda'); test4=test_cuda.CudaTensor([2,2,2], True, device='cuda');")
print(t1) # 

t0 = timeit.timeit('og3=og1.matmul(og2); og5=og3+og4; og5.mean().backward()', number=100000, setup="import replica; og1=replica.cuda_tensor.CudaTensor([1,2,3], True, device='cuda'); og2=replica.cuda_tensor.CudaTensor([[1,2,3],[1,2,3],[1,2,3]], True, device='cuda'); og4=replica.cuda_tensor.CudaTensor([2,2,2], True, device='cuda')")
print(t0) #
'''

