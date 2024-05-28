from torchic.cuda_tensor import Tensor
import numpy as np


def cat(tensors: list, dim: int=0, device=None) -> Tensor:
    return Tensor(np.concatenate(tensors, axis=dim), device)