import numpy as np


dtype_map = {
    'int': {
        8: np.int8,
        16: np.int16,
        32: np.int32,
        64: np.int64,
    },
    'float': {
        16: np.float16,
        32: np.float32,
        64: np.float64,
    },
    'complex': {
        64: np.complex64
        #128: np.complex128,
    },
    'bool': {
        32: np.int32,
        64: np.int64,
    }
}


def get_type(data: np.ndarray):
    match(data.dtype.name[0]):
        case 'i':
            return "int"
        case "f":
            return "float"
        case 'c':
            return "complex"
        case 'b':
            return "bool"