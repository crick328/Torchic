from torchic.nn.module import Module, ModuleList
from torchic.nn.models.sequential import Sequential
from torchic.autograd.functions import BackwardFunction
from torchic.autograd.backprop import ArrayType
from torchic.cuda_tensor import Tensor
import numpy as np
import cupy as cp
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


def total_memory_usage(model, root_tensor, optimizer):
    # voglio misurare e rappresentare in percentuale il memory usage
    # di optimizer, modello, tensori e tensori intermedi
    # => devo aggiornare get_nbytes

    def navigate_graph():
        pass

    
    pass


def model_memory_usage(model):
    pass

def graph_memory_usage(root_tensor):
    pass

def optimizer_memory_usage(optim, ignore_params: bool=True):
    pass



def get_nbytes_V2(obj: ArrayType | Tensor | Module | ModuleList | Sequential | BackwardFunction) -> int:
    size = 0

    return size



# NOTE: Non serve supportare le misure delle BackwardFunction perche'
# i leaf tensor (cioe' quelli che passano qui) non hanno grad_fn
def get_nbytes(obj: Tensor | Module | ModuleList | Sequential, _visited: set=None) -> int:
    if not _visited:
        _visited = set()
        
    # non conto due volte gli oggetti se sono puntati da piu' refernces
    if id(obj) in _visited:
        # short-circuit
        return 0
    
    _visited.add(id(obj))

    if isinstance(obj, (np.ndarray, cp.ndarray)):
        return obj.nbytes
    
    # overhead base e size __slots__
    size = sys.getsizeof(obj)

    if isinstance(obj, (tuple, list, set)):
        print("LISTA")
        size += sum(get_nbytes(value, _visited) for value in obj)
    elif isinstance(obj, dict):
        # considero sia chiavi che valori nei dizionari
        size += sum(get_nbytes(key, _visited) + get_nbytes(value, _visited) for key, value in obj.items())
    elif isinstance(obj, Sequential):
        size += sum(get_nbytes(module, _visited) for module in obj)
    else:
        # allora si tratta di un Tensor, un Module (o qualcosa che eredita da lui)
        # o una BackwardFunction. Tutti e tre hanno sempre gli slots
        for attr_name in obj.__slots__:
            attr = getattr(obj, attr_name)
            print(f"{attr_name}, {type(attr)}")

            if isinstance(attr, ModuleList):
                #size += sum(get_nbytes(module, _visited) for module in attr)
                size += get_nbytes(attr.modules)
            elif isinstance(attr, (Tensor, Module)):
                size += get_nbytes(attr, _visited)
    return size


# non so dove metterlo per ora
def main_tree_len(tensor: Tensor) -> int:
    if tensor.grad_fn is not None:
        for next_ten in tensor.grad_fn.next_elems():
            pass
        