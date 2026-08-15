from numpy import int8, int16, int32, int64
from numpy import float16, float32, float64
from numpy import complex64
from numpy import uint8, uint16, uint32, uint64
from numpy import bool_ as np_bool
from numpy import dtype
from torchic.settings import CONFIG, bfloat16, is_mixed_precision_enabled, get_mixed_precision_dtype
import re


dtype_map = {
    'int': {
        8: int8,
        16: int16,
        32: int32,
        64: int64,
    },
    'uint': {
        8: uint8,
        16: uint16,
        32: uint32,
        64: uint64,
    },
    'float': {
        16: float16,
        32: float32,
        64: float64,
    },
    'complex': {
        64: complex64
        #128: np.complex128,
    },
    'bool': {
        16: int8,
        32: int8,
        64: int8,
    }
}

DTYPE_MAP_V2 = {
    # float
    'f': {
        16: float16,
        32: float32,
        64: float64,
    },
    # int
    'i': {
        8: int8,
        16: int16,
        32: int32,
        64: int64,
    },
    # unsigned int
    'u': {
        8: uint8,
        16: uint16,
        32: uint32,
        64: uint64,
    },
    # bool
    'b': {
        16: int8,
        32: int8,
        64: int8,
    },
    # complex
    'c': {
        64: complex64
        #128: np.complex128,
    },
    # (in theory is never accessed here, but better to have it mapped)
    # Void dtype -> custom -> brain floats
    'V': {
        16: bfloat16
    }
}


# deprecated
def choose_dtype(data_type: dtype) -> dtype:
    '''
    Sets to the library's CONFIG precision, maintaining the given dtype, for every
    Tensor instance.

    If in mixed precision mode, precision won't be modified if already
    matching the mixed precision dtype (which will be the case for tensors
    obtained via operations). If it's not the same, then will still be
    casted to CONFIG precision. This is intentional because only non leaf
    tensors should be in mixed precision, not every tensor instanciated
    inside an autocast context.
    '''
    if is_mixed_precision_enabled() and data_type == get_mixed_precision_dtype():
        return data_type

    name, precision = re.findall(r'([a-zA-Z]+)(\d*)', data_type.name)[0]
    
    if name.startswith('b') or int(precision) <= CONFIG["precision"]:
        return data_type
    else:
        return dtype_map[name][CONFIG["precision"]]