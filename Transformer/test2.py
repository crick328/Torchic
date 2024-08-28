import torch
from torch import nn
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.1")
import torchic
from transformer import PositionalEncoding
from torchic.tools.visualization.graphs import draw_graph
import numpy as np


'''
Per far si' che si verifichi l'errore, devo avere:
- Dropout layer (NO, basta che non sia leaf! Allora c'entra il release!)
- un tensore usato in due operazioni che si riunisce in un unico tensore

Il riciclo degli identificatori e' un'invariante sia nella mha che nel drop
'''



drop = torchic.nn.Dropout(0.5)
x = torchic.Tensor(np.random.rand(10, 10), requires_grad=True)
#print(f"x: {x.shape()}, {x.requires_grad}, {x.is_leaf}, {x.grad_fn}")
#x = drop(x)
print(sys.getrefcount(x))
x += x
#print(f"drop(x): {x.shape()}, {x.requires_grad}, {x.is_leaf}, {x.grad_fn}")

print(sys.getrefcount(x))

def multi_ha(a, b):
    l1 = torchic.nn.Linear(10, 10)
    l2 = torchic.nn.Linear(10, 10)

    #print(f"a: {a.shape()}, {a.requires_grad}, {a.is_leaf}, {a.grad_fn}")
    #print(f"b: {b.shape()}, {b.requires_grad}, {b.is_leaf}, {b.grad_fn}")

    a = l1(a)
    b = l2(b)

    #print(f"l1(a): {a.shape()}, {a.requires_grad}, {a.is_leaf}, {a.grad_fn}")
    #print(f"l2(b): {b.shape()}, {b.requires_grad}, {b.is_leaf}, {b.grad_fn}")

    c = a + b

    #print(f"a + b: {c.shape()}, {c.requires_grad}, {c.is_leaf}, {c.grad_fn}")

    return c

res = multi_ha(x, x)
print(sys.getrefcount(x))
print("\n----------\n")
draw_graph(root_tensor=res)
res.mean().backward()

