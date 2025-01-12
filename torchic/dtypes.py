from numpy import int8, int16, int32, int64
from numpy import float16, float32, float64
from numpy import complex64
from numpy import uint8, uint16, uint32, uint64
from numpy import bool_ as np_bool
from numpy import ndarray
from numpy import dtype
from torchic.settings import CONFIG, is_mixed_precision_enabled, get_mixed_precision_dtype
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


def choose_dtype(data_type: dtype) -> dtype:
    # in teoria supporta il confronto tra cp.dtypes e np.dtypes
    # Al massimo confronta i .names
    if is_mixed_precision_enabled() and data_type == get_mixed_precision_dtype():
        return data_type

    name, precision = re.findall(r'([a-zA-Z]+)(\d*)', data_type.name)[0]
    
    if name.startswith('b') or int(precision) <= CONFIG["precision"]:
        return data_type
    else:
        return dtype_map[name][CONFIG["precision"]]