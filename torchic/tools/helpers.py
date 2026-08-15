from torchic.cuda_tensor import Tensor
from torchic.settings import get_current_backend
import numpy as np
from torchic.settings import CUPY_AVAILABLE
if CUPY_AVAILABLE:
    import cupy as cp


# zeros
# random.rand
# random.uniform
# ranodm.normal (tutte funzioni degl initializers)
# ecc.


# solo per compatibilita' con PyTorch, non mi piace e non lo uso
def tensor(data, requires_grad=False, device=None, dtype=None):
    return Tensor(data=data, requires_grad=requires_grad, device=device, dtype=dtype)


# ===== Functions that creates tensors =====

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
    return Tensor(tensor._device.zeros(tensor.shape()), requires_grad=requires_grad, device=device, dtype=dtype)

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

def full(shape: tuple, fill_value: Tensor, requires_grad=False, device=None, dtype=None):
    backend = fill_value._device
    return Tensor(backend.full(shape, fill_value.data, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)


# ===== Functions that works on existing tensors =====

def max(tensor, dim=None) -> tuple[Tensor, int]:
    '''
    Returns a tuple containing the max value on dim and it's index aving same dims
    as the given tensor
    '''
    return Tensor(tensor._device.max(tensor.data, dim)), tensor._device.argmax(tensor.data, dim)

def cat(tensor_list: list[Tensor], dim: int=0, device=None, dtype=None) -> Tensor:
    #backend = tensor_list[0]._device
    backend = get_current_backend()
    #output = backend.empty([t.data for t in tensor_list], dtype=tensor_list[0].dtype)
    # alcune volte da' errore se uso le tonde per il generatore...
    return Tensor(backend.concatenate([t.data for t in tensor_list], axis=dim), device=device, dtype=dtype)

# operazione che pare essere ok, ma da verificare se e' allineata
# con PyTorch, deve avere una backward e ha i for (si puo' migliorare?)
# --!-- Operazione differenziabile --!--
def __chunk(tensor: Tensor, num_chunks: int, dim=0) -> list['Tensor']:
        '''
        Tries to divide self in `num_chunks`. If the split cannot
        be exact, the first `total_length % num_chunks` sub-`Tensor`s
        will have an additional element.

        Args:
            num_chunks (int): number of desired chunks
            dim (int): the dimension where the chunking is done. Default: 0.

        Returns:
            list[Tensor]

        Notes:
            - the specified dimension has to be valid
            - `num_chunks` has to be inferior to `self.size(dim)`
        '''
        # Lunghezza totale dell'array
        total_length = tensor.size(dim)
        # Dimensione base di ciascuna parte
        base_size = total_length // num_chunks
        # Numero di parti con un elemento in più
        extra = total_length % num_chunks

        # Indici di inizio e fine per ciascun split
        chunks = []
        start = 0
        for i in range(num_chunks):
            # 1. calcolo la dimensione del sub-tensor
            part_size = base_size + (1 if i < extra else 0)

            # 2.
            slc = [slice(None)] * tensor.data.ndim  # Placeholder per tutte le dimensioni
            slc[dim] = slice(start, start + part_size)  # Slice lungo l'asse specificato

            # 3. operazione vera e propria. Uso lo slice per prendere il subtensor.
            # Quest'operazione e' differenziata da IndexBackward
            indices = tuple(slc)
            subtensor = tensor[indices]
            chunks.append(subtensor)

            # 4. Aggiorna `start` per il prossimo sotto-array
            start += part_size

        return chunks

#def _OGcat(tensor_list: list[np.ndarray], dim: int=0, device=None, dtype=None) -> Tensor:
    # questa funzione ha bisogno di un rework. Deve prendere in input una
    # lista di tensors (anziche' di np)
    #return Tensor(np.concatenate(tensor_list, axis=dim), device=device, dtype=dtype)

def __cat(tensor_list: list[Tensor], dim: int=0, device=None, dtype=None) -> Tensor:
    '''
    NOTE: Questa funzione e' pensata per operare su tensori che NON hanno
    bisogno del gradiente in quanto NON ha una BackwardFunction e non e'
    differenziabile.
    '''
    # Determina la forma dell'array di output
    shapes = [t.shape() for t in tensor_list]
    
    # Controlla che le forme siano compatibili
    # la dimensione scelta puo' avere valore arbitrario
    if not all(shape[:dim] == shapes[0][:dim] and shape[dim+1:] == shapes[0][dim+1:] for shape in shapes):
        raise ValueError("Incompatible shapes for concatenation")

    # Calcola la dimensione dell'asse specificato
    total_size = sum(shape[dim] for shape in shapes)
    # casto a lista per renderlo mutabile
    output_shape = list(shapes[0])
    output_shape[dim] = total_size

    # Crea l'array di output
    backend = tensor_list[0]._device
    output = backend.empty(output_shape, dtype=tensor_list[0].dtype)

    # Concatenazione
    current_index = 0
    for t in tensor_list:
        slice_ = [slice(None)] * output.ndim  # Crea uno slice per tutte le dimensioni
        slice_[dim] = slice(current_index, current_index + t.shape[dim])
        output[tuple(slice_)] = t.data
        current_index += t.shape[dim]

    return Tensor(output, device=device, dtype=dtype)

def stack(tensor_list: list[Tensor], dim: int=0, device=None, dtype=None) -> Tensor:
    '''
    Come cat, ma crea la dimensione su cui effettua il join
    '''
    #tensors = [t.unsqueeze(dim) for t in tensor_list]
    #return cat(tensors, device, dtype)
    backend = get_current_backend()
    #backend = tensor_list[0]._device
    #data_list = [t.data for t in tensor_list]
    return Tensor(backend.stack([t.data for t in tensor_list], axis=dim), device=device, dtype=dtype)


def hierarchical_state_dict(state_dict):
    """
    Converts a flat state dictionary (e.g., {'layer1.weight': ...}) 
    into a hierarchical one with nested dictionaries.

    Args:
        state_dict (dict): A flat state dictionary where keys are strings 
                           with dot-separated paths (e.g., 'layer1.weight').

    Returns:
        dict: A hierarchical state dictionary with nested dictionaries.
    
    Example:
        Input:
            {
                'layer1.weight': tensor,
                'layer1.bias': tensor,
                'layer2.submodule.weight': tensor
            }
        Output:
            {
                'layer1': {
                    'weight': tensor,
                    'bias': tensor
                },
                'layer2': {
                    'submodule': {
                        'weight': tensor
                    }
                }
            }
    """
    hierarchical_dict = {}

    for flat_key, value in state_dict.items():
        # Split the key into parts by '.'
        keys = flat_key.split('.')
        current_dict = hierarchical_dict

        # Traverse the hierarchy and create nested dictionaries
        for key in keys[:-1]:
            # Convert numeric keys to integers, if applicable
            key = int(key) if key.isdigit() else key

            # Create a nested dictionary if the key doesn't exist
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

        # Assign the final value to the deepest key
        final_key = int(keys[-1]) if keys[-1].isdigit() else keys[-1]
        current_dict[final_key] = value

    return hierarchical_dict


def gpu_time(funct, funct_args: tuple, n_repeat, n_warmup):
    import cupyx as cpx
    
    elapsed_time = cpx.profiler.benchmark(cp.matmul, funct_args, n_repeat=n_repeat, n_warmup=10)
    #cpx.time.repeat()
    elapsed_time = elapsed_time.gpu_times.mean()
    elapsed_time *= 1000    # converto in ms

    return elapsed_time