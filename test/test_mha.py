import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torchic
from torchic.tools.visualization.graphs import draw_graph


# c'e' qualcosa che non va...
# Posso assumere che, se entrambi i parent hanno grad_fn,
# allora sto andando incontro a due rami momentaneamente paralleli?
# Se poi vedo che qualche grad_fn ha una ref_count maggiore di uno,
# aspetto il calcolo dei gradienti.
# Se ho due parent con grad_fn non nulle, apro CudaStreams e
# faccio subito partire il calcolo (ricorsivamente? O streammo Backward?)

def multi_ha(a, b, c):
    l1 = torchic.nn.Linear(10, 10, bias=False)
    l2 = torchic.nn.Linear(10, 10, bias=False)
    l3 = torchic.nn.Linear(10, 10, bias=False)

    a = l1(a)
    b = l2(b)
    c = l3(c)

    y = a + b + c

    #print(f"a + b: {c.shape()}, {c.requires_grad}, {c.is_leaf}, {c.grad_fn}")
    return y


input = torchic.rand((10, ))
bias = torchic.rand((10, ), requires_grad=True)
x = input + bias


res = multi_ha(x, x, x)
#draw_graph(root_tensor=res)
res.mean().backward()