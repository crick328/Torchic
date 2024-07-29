from torchic.cuda_tensor import Tensor
import numpy as np


# where (occhio, questa chiederebbe backward. Ha senso metterla qua?
# Avrebbe anche bisogno di sapere il device e non e' un ars che posso
# passargli in input...)
# ones
# zeros
# ecc.


# solo per compatibilita' con PyTorch, non mi piace e non lo uso
def tensor(data, requires_grad, device, dtype):
    return Tensor(data=data, requires_grad=requires_grad, device=device, dtype=dtype)


def cat(tensors: list[Tensor], dim: int=0, device=None, dtype=None) -> Tensor:
    # questa funzione ha bisogno di un rework. Deve prendere in input una
    # lista di tensors (anziche' di np), deve controllare che tutto sia
    # sullo stesso device e chiamare np o cp da un ._device di un tensor
    # per chiamare opportunamente np o cp

    return Tensor(np.concatenate(tensors, axis=dim), device=device, dtype=dtype)