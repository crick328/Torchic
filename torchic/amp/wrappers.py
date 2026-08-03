from torchic.dtypes import float32, dtype
import torchic.autograd.functions as back_op

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor


amp_cache = {}

def wrapper_single_param(func, safeness):
    def cast_if_needed(x: 'Tensor'):
        x = _manage_arg_dtype(x, safeness)
        return func(x)
    return cast_if_needed

def wrapper_double_param(func, safeness):
    def cast_if_needed(self, other):
        self = _manage_arg_dtype(self, safeness)
        other = _manage_arg_dtype(other, safeness)
        return func(self, other)
    return cast_if_needed

def wrapper_act_funct(func, safeness):
    def cast_if_needed(self, other):
        other = _manage_arg_dtype(other, safeness)
        return func(self, other)
    return cast_if_needed

def generic_wrapper(func, safeness):
    pass


'''
Wrapper for `funct` (`Tensor`'s operation) that casts the args tensor(s)
to mixed precision.

Args:
    arg1 (Tensor): first arg of the function. It could be the first
        operand, the `self` arg of a class or don't care
    arg2 (Tensor): second arg of the function. It could be the second
        operand, the input of an `ActivationFunction`.__call__()
        or don't care

It converts to float16 the tensors that:
- are on a cuda device (otherwise there's no point because you can't
    use CUDA optimizations)
- have a precision higher than fp16
If the tensor does not require grad, an inplace conversion is made, otherwise it's first cloned
and then converted. This is done in order to avoid any variation to the master weights (original
parameters of the network in fp32) because their update have to be done in fp32.
'''


from torchic.settings import CONFIG, CUPY_AVAILABLE
from torchic.dtypes import DTYPE_MAP_V2

def _manage_arg_dtype(arg: 'Tensor', safeness: int) -> 'Tensor':
    # NOTE: non posso importare Tensor perche' andrei in import circolare,
    # quindi devo verificare la classe via stringa
    # NOTE: questa casistica gestisce anche arg2=None perche' e' di
    # classe NoneType
    # (se c'e') l'arg e' un Tensor ed e' su cuda
    #if arg.__class__.__name__ != "Tensor" or arg.device != "cuda":
    if not arg.device == "cuda":
        return arg
    
    match safeness:
        case -1:
            # funzione in blacklist: devo accertarmi che sia in fp32
            casted = arg.to(float32)

            #if arg.dtype.itemsize*8 < CONFIG["precision"]:
                # mantengo il tipo di dato e casto alla precisione richiesta
                #dtype = DTYPE_MAP_V2[arg.dtype.kind][CONFIG["precision"]]
                #arg.data = arg._to_dtype(arg.data, dtype)

        # per il momento non faccio nulla con le funzioni neutre
        case 0: pass
        case 1: pass

        case 2:
            # devo lavorare in fp16

            # non ha senso salvare in cache tensori intermedi
            # che sono da buttare al prossimo forward, percio'
            # mi assicuro che siano leaf
            if arg.is_leaf:
                tensor_id = id(arg)

                # questo dict va pulito...
                # Non e' un grosso problema: in teoria posso
                # modificare la parent class di Optimizer per fare
                # in modo che l'update parta sempre con un check di
                # questo dizionario e si pulisca... O lo faccio
                # in zero_grad
                # NOTE: ASPETTA, chi si prendera' questi gradienti?
                # Eh, i pesi-copia in fp16... L'optimizer non vedra'
                # nessun gradiente nei master
                if tensor_id not in amp_cache.keys():
                    # cache miss
                    # mantengo i parametri originali in fp32 e faccio il calcolo
                    # con una copia in fp16
                    casted = arg.clone().to(CONFIG["mixed_precision"]["dtype"])
                    amp_cache[tensor_id] = casted
                else:
                    # cache hit

                    # gli assegno la backward (altrimenti i gradienti
                    # rimarrebbero nei tensori fp16, verrebbero accumulati
                    # in fp16 e dovrei copiarli uno ad uno nei master prima
                    # di chiamare .step()) e restituisco il tensore
                    # in cache
                    casted = amp_cache[tensor_id]
                    casted.register_backward_if_needed(back_op.ToCopyBackward, arg, arg.device, arg.dtype)
                    
            else:
                # converto direttamente
                casted = arg.to(CONFIG["mixed_precision"]["dtype"])
        
        case _:
            raise RuntimeError(
                f"I received a safeness level of {safeness}, which "
                "isn't inside of the accepted range [-1, 2]."
            )

            # cosi' non mantengo il tipo originale (int, float, ecc),
            # ma comunque dovrei vedere solo float perche' solo
            # quelli possono richiedere un gradiente.
            # Potrei altrimenti mettere un metodo .to_precision()
            #arg = arg.to(CONFIG["mixed_precision"]["dtype"])
    return casted