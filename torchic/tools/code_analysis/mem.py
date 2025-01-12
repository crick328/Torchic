from torchic.nn.module import Module, ModuleList
from torchic.nn.models.sequential import Sequential
from torchic.cuda_tensor import Tensor
#import numpy as np
#import cupy as cp
import sys



def memory_estimator(cls, data_manager, dim):
    '''
    cls: classe (np.array, cp.array, ecc.)
    data_manager: float32, float16, np.float32, ecc.
    dim: 0, (1,1), (4,1), (30,30)

    all'interno devo avere un dict in cui è segnato il peso
    di base di ogni classe e tipo

    output: memoria occupata
    '''
    pass


# NOTE: Non serve supportare le misure delle BackwardFunction perche'
# i leaf tensor (cioe' quelli che passano qui) non hanno grad_fn
def get_nbytes(obj: Tensor | Module | ModuleList | Sequential) -> int:
    # overhead base e size __slots__
    size = sys.getsizeof(obj)

    if isinstance(obj, Sequential):
        for module in obj:
            size += get_nbytes(module)
        return size
    
    for attr_name in obj.__slots__:
        attr = getattr(obj, attr_name)
        #print(f"{attr_name}, {type(attr)}")

        if isinstance(attr, ModuleList):
            for module in attr:
                size += get_nbytes(module)
        elif isinstance(attr, (Tensor, Module)):
            size += get_nbytes(attr)
        # necessario o da' valori senza senso
        else:
            try:
                size += attr.nbytes
            except AttributeError:
                size += sys.getsizeof(attr)
        '''
        elif isinstance(attr, (np.ndarray, cp.ndarray)):
            size += attr.nbytes
        else:
            size += sys.getsizeof(attr)
        '''
    return size


# non so dove metterlo per ora
def main_tree_len(tensor: Tensor) -> int:
    if tensor.grad_fn is not None:
        for next_ten in tensor.grad_fn.next_elems():
            pass
        