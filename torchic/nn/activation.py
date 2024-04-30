# RULE OF THUMB: se devi solo fare operazioni sui .data e non
# ti serve la backprop, chiama solo quelli anziché tirare sempre in ballo i Tensor!
# metodo interessante per mantenere nella cache i risultati: https://docs.python.org/3/library/functools.html#functools.lru_cache

from torchic.cuda_tensor import Tensor
import numpy as np
import cupy as cp


class Sigmoid:
    # ho bisogno che sia una classe per usare __repr__
    '''
    Input: 1 dimensional Tensor
    Output: 1 dimensional Tensor having same shape as input
    '''
    def __call__(self, x: Tensor) -> Tensor:
        sigmoid = Tensor(1/(1 + x._device.exp(-x)))

        def _sigmoid_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            x._backward(
                # d/dσ = σ(x) * (1 - σ(x)); dL/dσ = d/dσ * prodotto derivate parziali delle operazioni precedenti
                self_grad=sigmoid.data * (1 - sigmoid.data) * cr_grad
            )

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            sigmoid.grad_fn = _sigmoid_backward
        return sigmoid
    
    def __repr__(self):
        return "Sigmoid()"


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        output = Tensor(x._device.maximum(0, x.data))

        def _ReLU_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            x._backward(
                self_grad=(x._device.where(x.data > 0, 1, 0) * cr_grad).astype(x._device.float32)
            )

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _ReLU_backward
        return output
    
    def __repr__(self):
        return "ReLU()"


class Softmax:
    __slots__ = ("dim")

    # Può essere calcolata solo su una dimensione
    def __init__(self, dim=0) -> None:
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)
        exps = x._device.exp(shiftx)
        softmax = Tensor(exps / x._device.sum(exps, axis=self.dim, keepdims=True))

        # NOTE: riciclare il calcolo della softmax è geniale! Fallo ovunque!
        def _softmax_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            '''
            la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
            derivata completa w.r.t. un input
            '''
            # CHADGPT. Calcolo introvabile. ChatGPT l'ha scritto versione Torch
            # e io l'ho confrontato alla softmax di PyTorch (ed era PERFETTA al
            # PRIMO TENTATIVO) e riadattato
            x._backward(
                self_grad=softmax.data * (cr_grad - x._device.sum(cr_grad * softmax.data, axis=self.dim, keepdims=True))
            )

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            softmax.grad_fn = _softmax_backward
        return softmax
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"