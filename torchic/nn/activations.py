# RULE OF THUMB: se devi solo fare operazioni sui .data e non
# ti serve la backprop, chiama solo quelli anziché tirare sempre in ballo i Tensor!
# metodo interessante per mantenere nella cache i risultati: https://docs.python.org/3/library/functools.html#functools.lru_cache

from torchic.cuda_tensor import Tensor
import torchic.backward as back_op
import numpy as np
import cupy as cp


# perche' mi serve capire se una classe e' cosa.
# In questo modo, se faccio isistance(layer, ActivationFunction)
# le spotto a prescindere dal nome
class ActivationFunction:
    def __repr__(self):
        output = f"{self.__class__.__name__}("
        return output + f"{self.dim})" if hasattr(self, "dim") else output + ")"


class Sigmoid(ActivationFunction):
    # ho bisogno che sia una classe per usare __repr__
    '''
    Input: 1 dimensional Tensor
    Output: 1 dimensional Tensor having same shape as input
    '''
    def __call__(self, x: Tensor) -> Tensor:
        sigmoid = Tensor(1/(1 + x._device.exp(-x)))

        def _sigmoid_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # d/dσ = σ(x) * (1 - σ(x)); dL/dσ = d/dσ * prodotto derivate parziali delle operazioni precedenti
            x._run_backward( sigmoid.data * (1 - sigmoid.data) * cr_grad )

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            sigmoid.grad_fn = _sigmoid_backward
        return sigmoid
    
    def __repr__(self):
        return "Sigmoid()"


class ReLU(ActivationFunction):
    def __call__(self, x: Tensor) -> Tensor:
        output = Tensor(x._device.maximum(0, x.data))

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = back_op.ReLUBackward(x)
        return output
    
    def __repr__(self):
        return "ReLU()"


class Softmax(ActivationFunction):
    __slots__ = ("dim")

    # Può essere calcolata solo su una dimensione
    def __init__(self, dim=0) -> None:
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)
        exps = x._device.exp(shiftx)
        softmax = exps / x._device.sum(exps, axis=self.dim, keepdims=True)
        output = Tensor(softmax)
        
        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = back_op.SoftmaxBackward(x, self.dim, softmax)
        return output
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"
    

class LogSoftmax(ActivationFunction):
    __slots__ = ("dim")

    # Può essere calcolata solo su una dimensione
    def __init__(self, dim=0) -> None:
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        #shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)
        #logsumexp = x._device.log(x._device.exp(shiftx).sum(axis=self.dim, keepdims=True))
        #softmax = Tensor(x.data - shiftx - logsumexp)

        shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)  # Sottrai il massimo per evitare l'overflow
        softmax = Tensor(shiftx - x._device.log(x._device.exp(shiftx).sum(axis=self.dim, keepdims=True)))  # Calcola il log del softmax

        # NOTE: riciclare il calcolo della softmax è geniale! Fallo ovunque!
        def _log_softmax_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            '''
            la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
            derivata completa w.r.t. un input
            '''
            # Con bruteforce di CHADGPT e match di logica della softmax, ecco una
            # derivata di cui al momento non so assolutamente nulla sulla
            # sua derivazione matematica!
            x._run_backward( cr_grad - x._device.exp(softmax.data) * cr_grad.sum(axis=self.dim, keepdims=True) )

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            softmax.grad_fn = _log_softmax_backward
        return softmax
    
    def __repr__(self):
        return f"LogSoftmax(dim={self.dim})"