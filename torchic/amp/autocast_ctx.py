# Automatic Mixed Precision (AMP)
# autocast to float16 with cupy in order to use Tensor Cores

# NOTE: sembra una funzione da usare con parsimonia. I Tensor Cores non si attivano
# sempre
# NOTE: un forward in fp16 richiede comunque l'uso del loss scaling, anche se la
# backward e' tutta fp32. Il loss scaling, cosi' come la mixed precision, richiedono
# controllo sull'optimizer

from torchic.settings import CUPY_AVAILABLE, get_default_device, get_mixed_precision_dtype, is_mixed_precision_enabled, set_mixed_precision_enabled

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor



# non ho capito se devo scartare solo le operazioni sensibili o le loro intere funzioni


# decorator
def autocast_operation(func):
    """
    Decoratore per applicare il casting agli argomenti delle operazioni.
    """

    def wrapper(*args, **kwargs):
        if is_mixed_precision_enabled():
            target_dtype = get_mixed_precision_dtype()
            new_args = []
            #print(target_dtype)

            # Cast degli argomenti posizionali
            new_args = [
                arg.clone().to(target_dtype) if hasattr(arg, "data") else arg
                for arg in args
            ]

            # Cast degli argomenti keyword
            new_kwargs = {
                key: value.clone().to(target_dtype) if hasattr(value, "data") else value
                for key, value in kwargs.items()
            }
            return func(*new_args, **new_kwargs)
        return func(*args, **kwargs)
    return wrapper


# per standard, le classi context manager vanno scritte minuscole
class autocast:
    '''
    EEEE
    '''
    def __init__(self):
        pass

    def __enter__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError()
        
        set_mixed_precision_enabled(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_mixed_precision_enabled(False)


