import numpy as np
import cupy as cp
# illegale perche' zeros usa Tensor e Tensor usa backward.py
#from torchic.tools.helper import zeros

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
    __slots__ = ("ctx", "ref_count", "acc_grad", "id")

    def __init__(self, *ctx: tuple) -> None:
        self.ctx = ctx
        self.ref_count = 0
        self.acc_grad = None
        self.id = BackwardFunction.count
        BackwardFunction.count += 1

        for tensor in self.next_elems():
            if tensor.grad_fn is not None:
                tensor.grad_fn.ref_count += 1
    
    def _update_acc_grad(self, cr_grad):
        if self.acc_grad is not None:
            self.acc_grad += cr_grad
        else:
            self.acc_grad = cr_grad

    def _next_grad_fn(self):
        for tensor in self.next_elems():
            if tensor.grad_fn is not None:
                yield tensor.grad_fn

    def next_elems(self):
        '''
        Allows to trace the backward graph.
        Yields the parent tensor(s).
        '''
        for elem in self.ctx:
            if hasattr(elem, "grad_fn"):
                # alcune backward hanno il tensore che le ha come grad_fn,
                # ma ovviamente sono usate solo per riciclare i calcoli
                # e non sono i prossimi tensori del grafo
                if elem.grad_fn is not None and elem.grad_fn.id == self.id:
                    continue
                yield elem

    def compute_grad(self, cr_grad):
        raise NotImplementedError

    def __call__(self, cr_grad):
        # se self fa partire la backward, non ha reference
        if self.ref_count <= 1:
            if self.acc_grad is not None:
                cr_grad += self.acc_grad
                # non mi serve piu' quindi lo resetto subito
                # per fare spazio
                self.acc_grad = None
            
            self.compute_grad(cr_grad)
            #del self
            self.ref_count -= 1
        else:
            self._update_acc_grad(cr_grad)
            self.ref_count -= 1
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.id}"
    
    def __hash__(self) -> int:
        return hash(self.ctx) + hash(self.__str__())
    
    def __del__(self):
        '''
        Called when the instance is about to be destroyed.
        This is also called a finalizer or (improperly) a destructor.
        '''
        try:
            if BackwardFunction.count is not None:
                BackwardFunction.count -= 1
        except AttributeError as e:
            #print(f"Qualcosa sta cancellando le backfun prima del dovuto: {e}")
            print("NONE")
    
    


# mettile semplicemente in ordine alfabetico
# perche' non fare cosi' anche in Tensor (solo per le operazioni)?

class AddBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        
        #cr_grad = cr_grad.reshape(x.data.shape)

        if x.requires_grad:
            # derivata parziale w.r.t. self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0,
            # cioè self con i valori a 1.
            grad_x = cr_grad

            if x.data.shape != cr_grad.shape:
                # Ridurre il gradiente alla shape originale di x
                grad_x = grad_x.sum(axis=tuple(range(cr_grad.ndim - x.data.ndim)))
            x._run_backward(grad_x)

        if y.requires_grad:
            # derivata parziale w.r.t. other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1,
            # cioè other con i valori a 1.
            grad_y = cr_grad

            if y.data.shape != cr_grad.shape:
                # Ridurre il gradiente alla shape originale di y
                grad_y = grad_y.sum(axis=tuple(range(cr_grad.ndim - y.data.ndim)))
            y._run_backward(grad_y)


class _CELossBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
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
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx

        #print(x._device.reciprocal(y.data)  *cr_grad)
        #print(x._device.reciprocal(y.data + 1e-8)  *cr_grad)
        #print(1/y.data *cr_grad)
        if x.requires_grad:
            # d/dx( x/y ) = d/dx(x) * 1/y = 1 * 1/y = 1/y
            # usando la funzione evito errori div by zero
            x._run_backward(x._device.reciprocal(y.data + 1e-8)  *cr_grad)

        if y.requires_grad:
            # d/dy( x/y ) = d/dy(1/y)*x = -1/y^2 * x = -x/y^2
            # per il momento non gestisco div by zero
            y._run_backward(-x.data / y._device.power(y.data, 2) * cr_grad)


class EmbeddingBackward(BackwardFunction):
    # sparse gradient
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        weights: "Tensor"
        x: "Tensor"
        weights, x = self.ctx
        
        if weights.requires_grad:
            # equivalente custom di weights._run_backward()
            cr_grad = cr_grad.astype(weights.dtype, copy=False)

            if weights._grad is None:
                # NOTE: si puo' aggiornare usanto torchic.zeros
                weights._grad = weights._device.zeros(weights.data.shape, dtype=weights.dtype)
            # NOTE: che differenza c'e' tra le due versioni?
            #weights._grad[x.data] += cr_grad
            weights._device.add.at(weights._grad, x.data, cr_grad)   

        if x.requires_grad:
            # pytorch ha della logica per x?
            # NO, x e' int!
            pass     


class ExpBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        x: "Tensor"
        x_exp: "Tensor"
        x, x_exp = self.ctx

        # d/dx( e^x ) = e^x
        # NOTE: l'esponente non si tratta come una composta
        x._run_backward(cr_grad * x_exp.data)


class IndexBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        x: "Tensor"
        index: int|slice
        x, index = self.ctx
        
        # porto cr_grad alle dimensioni di self mettendo degli 0 di padding
        grad = x._device.zeros(x.data.shape)
        grad[index] = cr_grad
        x._run_backward(grad)


class LogSoftmaxBackward(BackwardFunction):
    # Con bruteforce di CHADGPT e match di logica della softmax, ecco una
    # derivata di cui al momento non so assolutamente nulla sulla
    # sua derivazione matematica!
    def compute_grad(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        softmax: "Tensor"
        x, dim, softmax = self.ctx

        x._run_backward( cr_grad - x._device.exp(softmax.data) * cr_grad.sum(axis=dim, keepdims=True) )


class MaskedFillBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        x: "Tensor"
        mask: "Tensor"
        x, mask = self.ctx

        # NOTE: where(mask) =/= where(mask==0)!
        x._run_backward(x._device.where(mask.data, cr_grad, 0))


class MatmulBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        
        if x.requires_grad:
            if y.data.ndim > 1:
                # caso X @ Y
                grad_x = cr_grad @ y.data.swapaxes(-1, -2)
            else:
                # caso X @ y
                grad_x = x._device.outer(cr_grad, y.data.T)

            # Riduci grad_x se necessario
            if grad_x.shape != x.data.shape:
                grad_x = grad_x.sum(axis=tuple(range(grad_x.ndim - x.data.ndim)))
            x._run_backward(grad_x)

        if y.requires_grad:
            if x.data.ndim > 1:
                # caso X @ Y
                grad_y = x.data.swapaxes(-1, -2) @ cr_grad
            else:
                # caso x @ Y
                grad_y = y._device.outer(x.data.T, cr_grad)

            # Riduci grad_y se necessario
            if grad_y.shape != y.data.shape:
                grad_y = grad_y.sum(axis=tuple(range(grad_y.ndim - y.data.ndim)))
            y._run_backward(grad_y)


class _MatmulBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray | cp.ndarray) -> None:
        # NOTE: l'ordine in cui si moltiplica il gradient con la derivata
        # parziale e' importante, ma non ho capito perche' cambia da self a other
        # PEP 585
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        
        
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
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
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
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        # NOTE: la funzione x.expand_as(y) di PyTorch equivale a
        # np.broadcast_to(np.expand_dims(x, dim), y.data.shape) perche'
        # prima devo aggiungere la dimensione e poi devo far si' che i
        # suoi valori siano uguali a quelle delle altre dims.
        # Se x ha gia' lo stesso numero di dims, si omette .epxand_dims().
        # Con la mia classe Tensor, l'equivalente di x.expand_as(y) sarebbe
        # x.unsqueeze(dim).expand(y.shape())
        x: "Tensor"
        dim: int
        keepdim: bool
        x, dim, keepdim = self.ctx

        
        # NOTE: con questo non funziona il transformer...
        # Che abbia fatto male i test di mean()?
        if not keepdim and dim is not None:
            cr_grad = x._device.expand_dims(cr_grad, dim)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        # d/dx( (x + y + ... + z) / N) = 1/N * [d/dx(x) + d/dx(y) + ... + d/dx(z)]
        # sto derivando w.r.t. x => le altre le tratto come costanti
        # = 1/N * [1 + 0 + ... + 0] = 1/N
        # dL/dx = grad * 1/N (dove N = size(self) = n ingressi. totali di self o dim)
        x._run_backward(cr_grad / x._device.size(x.data, dim))


class MulBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx

        #if x.requires_grad:
            # d/dx( x*y ) => d/dx(x)*y = 1*y = y
            #x._run_backward(y.data * cr_grad)

        if x.requires_grad:
            # d/dy( x*y ) => d/dy(y)*x = 1*x = x
            grad_x = y.data * cr_grad

            # Se x è stato broadcastato, somma lungo gli assi corretti
            if x.data.ndim != grad_x.ndim:
                # Identifica gli assi da sommare
                axis = tuple(range(grad_x.ndim - x.data.ndim))
                grad_x = grad_x.sum(axis=axis)

            x._run_backward(grad_x)

        #if y.requires_grad:
            # d/dy( x*y ) => d/dx(y)*x = 1*x = x
            #y._run_backward(x.data * cr_grad)

        if y.requires_grad:
            # d/dy( x*y ) => d/dy(y)*x = 1*x = x
            grad_y = x.data * cr_grad

            # Se y è stato broadcastato, somma lungo gli assi corretti
            if y.data.ndim != grad_y.ndim:
                # Identifica gli assi da sommare
                axis = tuple(range(grad_y.ndim - y.data.ndim))
                grad_y = grad_y.sum(axis=axis)

            y._run_backward(grad_y)


class NegBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        #x = self.ctx[0]
        x, = self.ctx
        
        # d/dx(x * -1) = d/dx(x) * -1 = ones(len(x)) * -1
        x._run_backward(-cr_grad.reshape(x.data.shape))


class PowBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        exponent: int
        x, exponent = self.ctx

        x._run_backward(exponent * x._device.power(x.data, exponent-1) * cr_grad)


class ReLUBackward(BackwardFunction):
    # numpy nativamente fa (int32 * float32 = float64)
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        x, = self.ctx

        x._run_backward( x._device.where(x.data > 0, 1, 0) * cr_grad )


class SELossBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        target: "Tensor"
        x, target = self.ctx

        x._run_backward(2*(x.data - target.data) * cr_grad)

class SoftmaxBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
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


class SqrtBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        x_sqrt: "Tensor"
        x, x_sqrt = self.ctx

        # d/dx = 1/(2*sqrt(x))
        #x._run_backward(cr_grad * x_sqrt._device.reciprocal(2 * x_sqrt.data))
        x._run_backward(cr_grad * 1/(2 * x_sqrt.data))


class SubBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
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
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        '''
        x: "Tensor"
        x, = self.ctx
        x._run_backward(cr_grad.reshape(x.data.shape))
        '''

        x: "Tensor"
        dim: int
        keepdim: bool
        x, dim, keepdim = self.ctx

        if not keepdim and dim is not None:
            cr_grad = x._device.expand_dims(cr_grad, dim)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        # d/dx( (x + y + ... + z)) = [d/dx(x) + d/dx(y) + ... + d/dx(z)]
        # sto derivando w.r.t. x => le altre le tratto come costanti
        # = [1 + 0 + ... + 0] = 1
        # dL/dx = grad * 1
        x._run_backward(cr_grad)


class StdBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        mean: np.ndarray|cp.ndarray
        denom: np.ndarray|cp.ndarray
        keepdim: bool
        x, dim, mean, denom, keepdim = self.ctx

        if not keepdim and dim is not None:
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
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim1: int
        dim2: int
        x, dim1, dim2 = self.ctx

        # rifaccio la trasposta al contrario
        #x._run_backward(cr_grad.transpose(dim2, dim1))
        x._run_backward(cr_grad.swapaxes(dim2, dim1))


class UnsqueezeBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        x, dim = self.ctx

        # converto cr_grad da output.shape a self.data.shape
        x._run_backward(cr_grad.squeeze(dim))

    
class VarBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        dim: int
        mean: np.ndarray|cp.ndarray
        n: int
        keepdim: bool
        x, dim, mean, n, keepdim = self.ctx

        if not keepdim and dim is not None:
            cr_grad = x._device.expand_dims(cr_grad, dim)
            print(cr_grad.shape)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)

        # d/dx((x-x.mean)^2 / (N)) = 2(x-x.mean) / N
        x._run_backward(2*mean/n * cr_grad)


class ViewBackward(BackwardFunction):
    def compute_grad(self, cr_grad: np.ndarray|cp.ndarray) -> None:
        x: "Tensor"
        x = self.ctx[0]

        # converto cr_grad da output.shape a self.data.shape
        x._run_backward(cr_grad.reshape(x.data.shape))