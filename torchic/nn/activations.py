# RULE OF THUMB: se devi solo fare operazioni sui .data e non
# ti serve la backprop, chiama solo quelli anziché tirare sempre in ballo i Tensor!
# metodo interessante per mantenere nella cache i risultati: https://docs.python.org/3/library/functools.html#functools.lru_cache

from torchic.cuda_tensor import Tensor
import torchic.autograd.functions as back_op
from torchic.autograd.functions import register_backward_if_needed
from torchic.amp import autocast_operation
from torchic.nn.module import ForceSlots
#import numpy as np
#import cupy as cp


# perche' mi serve capire se una classe e' cosa.
# In questo modo, se faccio isistance(layer, ActivationFunction)
# le spotto a prescindere dal nome
class ActivationFunction(metaclass=ForceSlots):
    def __repr__(self):
        output = f"{self.__class__.__name__}("
        return output + f"{self.dim})" if hasattr(self, "dim") else output + ")"


# esperimento, chiamabili all'interno di Tensor() in modo che le ops
# ricevano kernel fusion e tu possa comunque usare la tua logica
'''
import cupy as cp
@cp.fuse()
def _fused_sigmoid_forward(x_data):
    return 1 / (1 + cp.exp(-x_data))

@cp.fuse()
def _fused_sigmoid_backward(sigmoid_data, cr_grad):
    return sigmoid_data * (1 - sigmoid_data) * cr_grad
'''
class Sigmoid(ActivationFunction):
    # ho bisogno che sia una classe per usare __repr__
    '''
    Input: 1 dimensional Tensor
    Output: 1 dimensional Tensor having same shape as input
    '''
    def __call__(self, x: Tensor) -> Tensor:
        sigmoid = Tensor(1/(1 + x._device.exp(-x.data)))

        def _sigmoid_backward(cr_grad) -> None:
            # d/dσ = σ(x) * (1 - σ(x)); dL/dσ = d/dσ * prodotto derivate parziali delle operazioni precedenti
            x._run_backward( sigmoid.data * (1 - sigmoid.data) * cr_grad )

        # da qua in avanti, requires_grad non sarà più modificabile perché
        # non è più considerato leaf tensor
        register_backward_if_needed(sigmoid, back_op.SigmoidBackward(x, sigmoid.data))
        return sigmoid
    
    def __repr__(self):
        return "Sigmoid()"
    

class Tanh(ActivationFunction):
    @autocast_operation
    def __call__(self, x: Tensor) -> Tensor:
        tanh = Tensor(x._device.tanh(x.data))

        register_backward_if_needed(tanh, back_op.TanhBackward(x, tanh.data))
        return tanh


class ReLU(ActivationFunction):
    @autocast_operation
    def __call__(self, x: Tensor) -> Tensor:
        output = Tensor(x._device.maximum(0, x.data))

        register_backward_if_needed(output, back_op.ReLUBackward(x))
        return output
    
    def __repr__(self):
        return "ReLU()"


class Softmax(ActivationFunction):
    __slots__ = ("dim",)

    # Può essere calcolata solo su una dimensione
    def __init__(self, dim=0) -> None:
        self.dim = dim

    @autocast_operation
    def __call__(self, x: Tensor) -> Tensor:
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)
        #exps = x._device.exp(shiftx)
        #softmax = exps / x._device.sum(exps, axis=self.dim, keepdims=True)
        softmax = x._device.exp(shiftx)
        softmax /= x._device.sum(softmax, axis=self.dim, keepdims=True)
        output = Tensor(softmax)
        
        register_backward_if_needed(output, back_op.SoftmaxBackward(x, self.dim, softmax))
        return output
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"
    

class LogSoftmax(ActivationFunction):
    __slots__ = ("dim",)

    # Può essere calcolata solo su una dimensione
    def __init__(self, dim=0) -> None:
        self.dim = dim

    @autocast_operation
    def __call__(self, x: Tensor) -> Tensor:
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient
        shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)
        shiftx -= x._device.log(x._device.exp(shiftx).sum(axis=self.dim, keepdims=True))
        softmax = Tensor(shiftx)

        register_backward_if_needed(softmax, back_op.LogSoftmaxBackward(x, self.dim, softmax.data))
        return softmax
    
    def __repr__(self):
        return f"LogSoftmax(dim={self.dim})"