from test2_NNet import NNetwork, LLayer, Sigmoid, ReLU, SELoss, save_model_type1, load_model_type1
#from test12_tensor_backup import Tensor
from optim import SGD
import numpy as np
import time

#t = time.time()
#a = Tensor([np.ones((2300,5400))])
#print(f"Tempo impiegato: {((time.time() - t)):.10f} secondi")

from sys import getsizeof

#print(f"{getsizeof(Tensor([]))+getsizeof(Tensor([]).__dict__)} bytes")
from test13_tensor import Tensor

model = NNetwork(
    LLayer(2, 30),
    ReLU(),
    LLayer(30, 1),
    Sigmoid()
)

params, grads = model.parameters()
loss = SELoss(model(Tensor([1,1])), Tensor([3]))
loss.mean().backward()
print(model.llayer0.weights.grad)
print(params[0][1])



#print(model(Tensor([1,4])))
#save_model_type1(model, "test")
#model = load_model_type1("test")
#print("ok")

#print(model(Tensor([1,4])))


'''
from test13_tensor import Tensor
#from test12_tensor_backup import Tensor as TensorOG
a = Tensor(np.ones(20), requires_grad=True)
b = TensorOG(np.ones(20))


from time import perf_counter

t = perf_counter()
a.data
a.data
a.data
a.data
a.data
print(f"Tempo impiegato: {((perf_counter() - t)):.10f} secondi")

t = perf_counter()
b.data
b.data
b.data
b.data
b.data
print(f"Tempo impiegato: {((perf_counter() - t)):.10f} secondi")

print(f"{getsizeof(Tensor([]))+getsizeof(Tensor([]).__slots__)} bytes")
print(f"{getsizeof(TensorOG([]))+getsizeof(TensorOG([]).__dict__)} bytes")

#print(f"{getsizeof(a)+getsizeof(a.)} bytes")
'''
import sys
print(sys.prefix)