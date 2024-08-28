from numpy import int8, int16, int32, int64
from numpy import float16, float32, float64
from numpy import complex64
from numpy import ndarray

dtype_map = {
    'int': {
        8: int8,
        16: int16,
        32: int32,
        64: int64,
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
        32: int32,
        64: int64,
    }
}


def get_type(data: ndarray):
    match data.dtype.name[0]:
        case 'i':
            return "int"
        case "f":
            return "float"
        case 'c':
            return "complex"
        case 'b':
            return "bool"