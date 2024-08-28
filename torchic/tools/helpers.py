from torchic.cuda_tensor import Tensor
from torchic.settings import DEFAULT
import numpy as np


# where (occhio, questa chiederebbe backward. Ha senso metterla qua?
# Avrebbe anche bisogno di sapere il device e non e' un ars che posso
# passargli in input...)
# ones
# zeros
# random.rand
# random.uniform
# ranodm.normal (tutte funzioni degl initializers)
# ecc.


# solo per compatibilita' con PyTorch, non mi piace e non lo uso
def tensor(data, requires_grad, device, dtype):
    return Tensor(data=data, requires_grad=requires_grad, device=device, dtype=dtype)


def rand(shape: tuple, requires_grad=False, device=None, dtype=None):
    '''
    NOTE: Se si vuole ottenere un vettore, usare shape=(n,)
    '''
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad, device=device, dtype=dtype)

def randint(shape: tuple, low: int=0, high: int=100, requires_grad=False, device=None, dtype=None):
    return Tensor(np.random.randint(low=low, high=high, size=shape), requires_grad=requires_grad, device=device, dtype=dtype)

def zeros(shape: tuple, requires_grad=False, device=None, dtype=None):
    return Tensor(np.zeros(shape), requires_grad=requires_grad, device=device, dtype=dtype)

def zeros_like(tensor: Tensor, requires_grad=False, device=None, dtype=None):
    return Tensor(np.zeros(tensor.shape()), requires_grad=requires_grad, device=device, dtype=dtype)

def ones(shape: tuple, requires_grad=False, device=None, dtype=None):
    return Tensor(np.ones(shape), requires_grad=requires_grad, device=device, dtype=dtype)

def arange(start: int|float, stop: int|float, step: int=1, requires_grad=False, device=None, dtype=None):
    return Tensor(np.arange(start=start, stop=stop, step=step), requires_grad=requires_grad, device=device, dtype=dtype)

def log(x, requires_grad=False, device=None, dtype=None):
    return Tensor(np.log(x), requires_grad=requires_grad, device=device, dtype=dtype)

def sin(x, requires_grad=False, device=None, dtype=None):
    return Tensor(np.sin(x), requires_grad=requires_grad, device=device, dtype=dtype)

def cos(x, requires_grad=False, device=None, dtype=None):
    return Tensor(np.cos(x), requires_grad=requires_grad, device=device, dtype=dtype)

def cat(tensor_list: list[np.ndarray], dim: int=0, device=None, dtype=None) -> Tensor:
    # questa funzione ha bisogno di un rework. Deve prendere in input una
    # lista di tensors (anziche' di np)
    return Tensor(np.concatenate(tensor_list, axis=dim), device=device, dtype=dtype)

def stack(tensor_list: list[np.ndarray], dim: int=0, device=None, dtype=None) -> Tensor:
    '''
    Come cat, ma crea la dimensione su cui effettua il join
    '''
    return Tensor(np.stack(tensor_list, axis=dim), device=device, dtype=dtype)