import numpy as np
import cupy as cp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor


# Questo approccio permette di:
# - separare il codice;
# - evitare di "compilare" la backward function ogni volta che si chiama una
#   funzione (cosa che cambia le performance in inference)
# - selezionare cosa salvare per la backward e cosa no anziche' avere Python
#   che salva tutto (args inclusi)
# - creare un vero grafo;
#
# introduzione di _version in modo da evitare casini? Si potrebbe introdurre
# facendo si che .data abbia un setter. Ogni volta che viene triggerato,
# aumenta la version di uno.


class BackwardFunction:
    '''
    Parent class of all backward functions.

    Attributes
        `count` (int): class attributes that is used to create
        a meaningful id to attach to the class name when printing.
        It's very useful for debugging and economic computation-wise,
        but it doesn't follow the PyTorch numeration system (for personal
        preference).
        `ctx` (tuple): 
    '''
    count = 0

    def __init__(self, *ctx: tuple) -> None:
        self.ctx = ctx
        self.id = BackwardFunction.count
        BackwardFunction.count += 1

    def next_elems(self):
        '''
        Allows to trace the backward graph.
        Yields the parent tensor(s).
        '''
        for elem in self.ctx:
            if hasattr(elem, "grad_fn"):
                yield elem
    
    def __str__(self):
        return f"{self.__class__.__name__}{self.id}"
    
    def __hash__(self) -> int:
        return hash(self.ctx) + hash(self.__str__())
    
    


# mettile semplicemente in ordine alfabetico
# perche' non fare cosi' anche in Tensor (solo per le operazioni)?

class AddBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        
        cr_grad = cr_grad.reshape(x.data.shape)

        if x.requires_grad:
            # derivata parziale w.r.t. self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
            x._run_backward(cr_grad)

        if y.requires_grad:
            # derivata parziale w.r.t. other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
            y._run_backward(cr_grad)


class _CELossBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        softmax_x: "Tensor"
        target: "Tensor"
        x, softmax_x, target = self.ctx

        cr_grad = x._device.expand_dims(cr_grad, -1)
        #print(f"crgrad {cr_grad}")
        #print(f"x {x}")
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        x._run_backward((softmax_x - target).data * cr_grad)


class DivBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx

        if x.requires_grad:
            # d/dx( x/y ) = d/dx(x) * 1/y = 1 * 1/y = 1/y
            # usando la funzione evito errori div by zero
            x._run_backward(x._device.reciprocal(y.data)  *cr_grad)

        if y.requires_grad:
            # d/dy( x/y ) = d/dy(1/y)*x = -1/y^2 * x = -x/y^2
            # per il momento non gestisco div by zero
            y._run_backward(-x.data / y._device.power(y.data, 2) * cr_grad)


class EmbeddingBackward(BackwardFunction):
    # sparse gradient
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        weights: "Tensor"
        x: "Tensor"
        weights, x = self.ctx
        
        if weights.requires_grad:
            # equivalente custom di weights._run_backward()
            cr_grad = cr_grad.astype(weights.dtype, copy=False)

            if weights._grad is None:
                weights._grad = weights._device.zeros(weights.data.shape, dtype=weights.dtype)
            # NOTE: che differenza c'e' tra le due versioni?
            #weights._grad[x.data] += cr_grad
            weights._device.add.at(weights._grad, x.data, cr_grad)   

        if x.requires_grad:
            # pytorch ha della logica per x?
            pass      


class MaskedFillBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        x: "Tensor"
        masked: "Tensor"
        value: int|float
        #x, masked, value = self.ctx
        x, mask, masked, value = self.ctx

        # NOTE: where(mask) =/= where(mask==0)!
        x._run_backward(x._device.where(mask.data, cr_grad, 0))


class MatmulBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        # NOTE: l'ordine in cui si moltiplica il gradient con la derivata
        # parziale e' importante, ma non ho capito perche' cambia da self a other
        # PEP 585
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx

        #print(f"SONO IO: {self.__str__()}")
        
        if x.requires_grad:
            if y.data.ndim > 1:
                # caso X @ Y (operazione tra le ultime due dims)
                # d/dX = Y.T;   dL/dX = crt @ Y.T
                x._run_backward(cr_grad @ y.data.swapaxes(-1, -2))
            else:
                # caso X @ y
                # d/dX = y.T;   dL/dX = crt ⊗ y.T
                x._run_backward(x._device.outer(cr_grad, y.data.T))

        if y.requires_grad:
            if x.data.ndim > 1:
                # caso X @ Y (operazione tra le ultime due dims)
                # d/dY = X.T;   dL/dY = X.T @ crt
                y._run_backward(x.data.swapaxes(-1, -2) @ cr_grad)
            else:
                # caso x @ Y
                # d/dY = x.T;   dL/dY = x.T ⊗ crt
                y._run_backward(y._device.outer(x.data.T, cr_grad))


class __MatmulBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        # NOTE: l'ordine in cui si moltiplica il gradient con la derivata
        # parziale e' importante, ma non ho capito perche' cambia da self a other
        # PEP 585
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        

        match([x.data.ndim, y.data.ndim]):
            # x @ A
            #if x.data.ndim == 1 and y.data.ndim == 2:
            case [1, 2]:
                if x.requires_grad:
                    # d/dx = A.T;   dL/dx = crt @ A.T
                    x._run_backward(cr_grad @ y.data.T)
                if y.requires_grad:
                    # d/dA = x.T;   dL/dA = x.T ⊗ crt
                    y._run_backward(y._device.outer(x.data.T, cr_grad))

            # A @ B
            #elif x.data.ndim == y.data.ndim == 2:
            case [2, 2]:
                if x.requires_grad:
                    # d/dA = (A @ B)' = (A)' @ B = B.T = d/dx
                    # dL/dA = crt @ B.T
                    x._run_backward(cr_grad @ y.data.T)
                if y.requires_grad:
                    # d/dB  = (A @ B)' = A @ (B)' = A.T
                    # dL/dB  = A.T @ crt
                    y._run_backward(x.data.T @ cr_grad)
            # A @ x oppure matmul con dim >= 3
            #else:
            case _:
                raise NotImplementedError(
                    'Per ora hai definito la matmul backward nel caso vettore @ matrice '
                    'e matrice @ matrice, ma l\'operazione che stai facendo '
                    'non rientra in questa casistica. '
                    f'dim self={x.data.ndim}, dim other={y.data.ndim}'
                )  


class MeanBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        # NOTE: la funzione x.expand_as(y) di PyTorch equivale a
        # np.broadcast_to(np.expand_dims(x, dim), y.data.shape) perche'
        # prima devo aggiungere la dimensione e poi devo far si' che i
        # suoi valori siano uguali a quelle delle altre dims.
        # Se x ha gia' lo stesso numero di dims, si omette .epxand_dims().
        # Con la mia classe Tensor, l'equivalente di x.expand_as(y) sarebbe
        # x.unsqueeze(dim).expand(y.shape())
        x: "Tensor"
        dim: int
        x, dim = self.ctx

        
        # NOTE: con questo non funziona il transformer...
        # Che abbia fatto male i test di mean()?
        #if dim is not None:
            #cr_grad = x._device.expand_dims(cr_grad, dim)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        # d/dx( (x + y + ... + z) / N) = 1/N * [d/dx(x) + d/dx(y) + ... + d/dx(z)]
        # sto derivando w.r.t. x => le altre le tratto come costanti
        # = 1/N * [1 + 0 + ... + 0] = 1/N
        # dL/dx = grad * 1/N (dove N = size(self) = n ingressi. totali di self o dim)
        x._run_backward(cr_grad / x._device.size(x.data, dim))


class MulBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx

        if x.requires_grad:
            # d/dx( x*y ) => d/dx(x)*y = 1*y = y
            x._run_backward(y.data * cr_grad)

        if y.requires_grad:
            # d/dy( x*y ) => d/dx(y)*x = 1*x = x
            y._run_backward(x.data * cr_grad)


class NegBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        #x = self.ctx[0]
        x, = self.ctx
        
        # d/dx(x * -1) = d/dx(x) * -1 = ones(len(x)) * -1
        x._run_backward(-cr_grad.reshape(x.data.shape))


class PowBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        exponent: int
        x, exponent = self.ctx

        x._run_backward(exponent * x._device.power(x.data, exponent-1) * cr_grad)


class ReLUBackward(BackwardFunction):
    # numpy nativamente fa (int32 * float32 = float64)
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        x, = self.ctx

        x._run_backward( x._device.where(x.data > 0, 1, 0) * cr_grad )


class SoftmaxBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        softmax: np.ndarray|cp.ndarray
        x, dim, softmax = self.ctx

        '''
        la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
        derivata completa w.r.t. un input
        '''
        # CHADGPT. Calcolo introvabile. ChatGPT l'ha scritto versione Torch
        # e io l'ho confrontato alla softmax di PyTorch (ed era PERFETTA al
        # PRIMO TENTATIVO) e riadattato
        x._run_backward( softmax * (cr_grad - x._device.sum(cr_grad * softmax, axis=dim, keepdims=True)) )


class SubBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        cr_grad = cr_grad.reshape(x.data.shape)

        if x.requires_grad:
            # derivata parziale w.r.t self, cioè x. d/dx(x-a)=1 ==> d/dx(x-y)=1-0, cioè self con i valori a 1.
            x._run_backward(cr_grad)

        if y.requires_grad:
            # derivata parziale w.r.t other, cioè y. d/dy(a-y)=-1 ==> d/dy(x-y)=0-1, cioè other con i valori a -1.
            y._run_backward(-cr_grad)


class SumBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        '''
        x: "Tensor"
        x, = self.ctx
        x._run_backward(cr_grad.reshape(x.data.shape))
        '''

        x: "Tensor"
        dim: int
        x, dim = self.ctx

        if dim is not None:
            cr_grad = x._device.expand_dims(cr_grad, dim)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        # d/dx( (x + y + ... + z)) = [d/dx(x) + d/dx(y) + ... + d/dx(z)]
        # sto derivando w.r.t. x => le altre le tratto come costanti
        # = [1 + 0 + ... + 0] = 1
        # dL/dx = grad * 1
        x._run_backward(cr_grad)


class StdBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        mean: np.ndarray|cp.ndarray
        denom: np.ndarray|cp.ndarray
        x, dim, mean, denom = self.ctx

        if dim is not None:
            cr_grad = x._device.expand_dims(cr_grad, dim)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        # d/dx( sqrt((x-x.mean)^2 / (N)) )
        # h(x) = (x-x.mean)^2 / (N); f(y) = sqrt(h(x))
        # chain rule => d/dx = df/dh * dh/dx
        # => 1/( 2*sqrt(h(x)) ) * 2(x-x.mean)/N
        # = 1/( N*sqrt(h(x)) ) * (x-x.mean)  =  (x-x.mean) / ( N*sqrt(h(x)) );
        # sqrt(h(x)) e' la formula della std, =>
        # (x - x.mean) / (N * std)

        # evito una possibile division by zero esplicando che voglio la divisione
        # solo dove i valori sono != 0
        x._run_backward(x._device.divide(mean, denom, where= denom!=0) * cr_grad)


class TransposeBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim1: int
        dim2: int
        x, dim1, dim2 = self.ctx

        # rifaccio la trasposta al contrario
        #x._run_backward(cr_grad.transpose(dim2, dim1))
        x._run_backward(cr_grad.swapaxes(dim2, dim1))


class UnsqueezeBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        x, dim = self.ctx

        # converto cr_grad da output.shape a self.data.shape
        x._run_backward(cr_grad.squeeze(dim))

    
class VarBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        mean: np.ndarray|cp.ndarray
        n: int
        x, dim, mean, n = self.ctx

        if dim is not None:
            cr_grad = x._device.expand_dims(cr_grad, dim)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        # d/dx((x-x.mean)^2 / (N)) = 2(x-x.mean) / N
        x._run_backward(2*mean/n * cr_grad)


class ViewBackward(BackwardFunction):
    def __call__(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        x = self.ctx[0]

        # converto cr_grad da output.shape a self.data.shape
        x._run_backward(cr_grad.reshape(x.data.shape))