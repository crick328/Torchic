from torchic.autograd.backprop import ArrayType
from torchic.autograd.backprop import pool
from collections.abc import Generator
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor
from torchic.settings import CUPY_AVAILABLE, CONFIG
if CUPY_AVAILABLE:
    import cupy as cp
import random
import numpy as np

# RULES OF THUMB:
# 1) i parent devono sempre occupare i primi due posti in ctx (per backpropagation)
# 2) i parent devono essere passati come tensors (per grad e next_functions)
# 3) tutto il resto si puo' passare come array e non ha un ordine
#    specifico, devi solo spacchettarli con il giusto ordine
# Di base, solo i parent dovrebbero essere salvati come tensor perche'
# tutti gli altri elementi basta il .data ed e' molto meglio puntare
# solo quello

# NOTE: bisogna aggiornare tutti i typehint dei return delle backward functions
# NOTE (!): l'errore "output array is read-only" deriva dall'uso di .broadcast_to()! 
# Infatti, l'output e' una view read-only NON CONTIGUA! La prima cosa ti impedisce
# di usare operazioni in-place, la seconda potrebbe creare rallentamenti in memoria.
# Se puoi, sostituiscila ASAP!
# NOTE: la tua backward si blocca se la fai partire da un ramo che converge
# in un nodo di accumulazione perche' gli altri non saranno mai calcolati, il
# ref_count non si abbassera' e quindi la backward si ferma al nodo di accumulazione.
# Pytorch riesce a gestirlo, ma non so come



class BackwardFunction:
    '''
    Base class for all backward functions.

    Attributes:
        ctx (tuple): context variable that contains the parent(s) `Tensor`(s) and other
            useful data for the backward pass such as reusable tensor (in order to
            do less computation as possible). Parents are always present even if they
            are not useful for the computation because they are used to navigate the graph.
        ref_count (int): counts the number of non-leaf tensors (nodes) that have this
            node as parent. I call them "critical tensor" if `ref_count` is greater than 1
            because they need special treatment during the backward pass.
        event_register (dict of cp.cuda.Stream: cp.cuda.Event): a dictionary containing
            cuda events recorded on `stream` after partial grad contribution.
            Used to fine-synching all streams involved in the computation of the
            gradient of a critical tensor.
        grad_buffer (dict of cp.cuda.Stream: ArrayType): dictionary containing all the
            partial gradients computed by each involved stream. If a stream computes
            multiple parts of it, a new entry on the dict is created in order to avoid
            instabilities with partial or incomputed gradients. A dict is used instead
            of a list in order to avoid concurrent access and allow for a safe accumulation.
            For more details, see `self.accumulate_grad()`.
        id (int): number of backward functions created at the time of the istance.
            It's used to identify every function and starts from 0 at every backpropagation.
    '''

    count = 0
    __slots__ = ("ctx", "ref_count", "event_register", "grad_buffer", "id")

    def __init__(self, *ctx: tuple) -> None:
        self.ctx = ctx
        self.ref_count = 0
        self.event_register = {}
        self.grad_buffer = {}
        self.id = BackwardFunction.count
        BackwardFunction.count += 1

        self.__update_ref_count()

    
    def __update_ref_count(self) -> None:
        '''
        Updates the `ref_count` (reference count) of the parent(s).
        '''
        grad_fn1 = self.ctx[0].grad_fn
        if grad_fn1:
            grad_fn1.ref_count += 1
        
        if len(self.ctx) > 1:
            grad_fn2 = getattr(self.ctx[1], "grad_fn", None)

            if grad_fn2:
                grad_fn2.ref_count += 1
                
    def parents(self) -> tuple['Tensor']:
        '''
        Returns the parent(s) `Tensor`(s).

        They are stored at the beginning of `self.ctx` because they are used
        during the backward process to trace back the graph.
        They are not stored in an apposite attribute of `self` because they
        would occupy memory (two pointers plus the overhead of a tuple/list)
        and should still be present in `self.ctx` for backward computations.
        Instead, they are extracted anytime they are needed (lazy approach).
        '''
        if len(self.ctx) > 1:
            parent2 = self.ctx[1]
            # devo sincerarmi che sia un tensore
            #if parent2.__class__.__name__ == "Tensor":
            if type(parent2) == type(self.ctx[0]):
                return self.ctx[0], parent2
        return self.ctx[0], None

    def accumulate_grad(self, grad: ArrayType, stream: cp.cuda.Stream, using_cuda: bool) -> None:
        """
        Safely accumulates the given gradient contribution.

        Args:
            grad (ArrayType): the gradient to accumulate
            stream (cp.cuda.Stream): tye CUDA stream that is working on it.
                `None` is used if on CPU.

        Gradient contributions are stored in the attribute `grad_buffer`, which is
        a dict containing key-value pairs `{stream_id: respective_gradient}`.
        Everytime a stream add a contribution, a new entry on the dict is created.
        This approah allows to avoid data corruption (for example when trying
        to add a gradient with a partially compute one) and race conditions.

        An in-place sum is used only if not using stream or if it is the
        null stream because its implicitly synchronized with the CPU and
        all the other streams.
        When training on cpu, then `grad_buffer` will be `{None: accumulated_grad}`
        because no stream is used (`stream=None`) and so the accumulation can
        be in-place.
        """
        # se sono su CPU, stream e' None
        # prima metto il grad, poi registro l'evento

        # se è la prima volta che questa stream fornisce un contributo
        if stream not in self.grad_buffer:
            # aggiungo il primo contributo
            self.grad_buffer[stream] = grad
        # se ho una stream, non e' la null e ha gia' dato un contributo
        elif using_cuda and stream is not pool.null_stream:
            # aggiungo separatamente gli altri contributi per
            # evitare somme in-place su dati non ancora pronti
            self.grad_buffer[f"{stream}_{random.random()}"] = grad
        # se non sto usando stream o sto usando la null
        else:
            # posso sommare in-place
            self.grad_buffer[stream] += grad
            # non ho bisogno di eventi per la null o per CPU
            return

        # aggiorno l'evento per il nuovo gradiente parziale
        # (aspettare il nuovo gia' implica aspettare il vecchio).
        # La somma del gradiente accumulato avverra' sulla default_stream,
        # ma le stream non possono aspettare i loro stessi eventi,
        # quindi non ne creo uno per lei
        if using_cuda and stream is not pool.default_stream:
            new_event = cp.cuda.Event()
            stream.record(new_event)
            self.event_register[stream] = new_event

    def next_elems(self) -> Generator['Tensor']:
        '''
        Allows to trace the backward graph.
        Yields the parent tensor(s).
        '''
        print("Sono next_elems(), depecrami!")
        yield self.ctx[0]
        
        if len(self.ctx) > 1:
            parent2 = self.ctx[1]
            # devo sincerarmi che sia un tensore
            #if getattr(parent2, "grad_fn", None):
            if parent2.__class__.__name__ == "Tensor":
                yield parent2
        
    def __call__(self, cr_grad: ArrayType) -> tuple[ArrayType, ArrayType]:
        raise NotImplementedError
    
    def __eq__(self, grad_fn) -> bool:
        if isinstance(grad_fn, BackwardFunction):
            return self.id == grad_fn.id
        else:
            return False
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.id}"
    
    def __hash__(self) -> int:
        return hash(self.ctx) + hash(self.__str__())
    
    def __del__(self):
        '''
        Called when the instance is about to be destroyed.
        This is also called a finalizer or (improperly) a destructor.
        '''
        #print(f"Cancello {self.__class__.__name__}{self.id}")
        try:
            if BackwardFunction.count is not None:
                BackwardFunction.count -= 1
        except AttributeError as e:
            #print(f"Qualcosa sta cancellando le backfun prima del dovuto: {e}")
            pass
    


def register_backward_if_needed(output: 'Tensor',
                                grad_fn: BackwardFunction) -> None:
    '''
    Function that registers the backward function if some conditions are met, such as
    autograd is on and at least one parent requires grad.
    Args:
        output (Tensor): result of the operation
        grad_fn (BackwardFunction): instance of the backward function to register (which
           already has received the parents as args) 
    '''
    if not CONFIG["grad_enabled"]:
        return
    #if grad_fn is not None:
    parent1, parent2 = grad_fn.parents()
    #else:
        #raise RuntimeError(
            #"Mi stai chiedendo di sovrascrivere una grad function non nulla "
            #"con un valore nullo... Perche'?"
            #)

    
    #print(f"PRINT IF: {CONFIG['grad_enabled']} and {parent1.requires_grad} or {parent2.requires_grad if parent2 is not None else False}, che fa {CONFIG['grad_enabled'] and (parent1.requires_grad or (parent2.requires_grad if parent2 is not None else False))}")
    # if autograd is on and at least one of the two parents (if there are two) requires grad
    #if CONFIG["grad_enabled"] and
    if parent1._requires_grad or (parent2._requires_grad if parent2 is not None else False):
        if output._grad_fn is None:
            # aggiornato a False perché si assegna una grad_fn
            output._is_leaf = False
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output._requires_grad = True
            # assegno la grad_fn
            output._grad_fn = grad_fn
        else:
            raise RuntimeError(
                "Mi stai chiedendo di sovrascrivere una grad function non nulla "
                "con un valore non nullo, ma questo romperebbe la backpropagation."
                )
        


# Ordine alfabetico

class AddBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        grad_x, grad_y = None, None

        if x.requires_grad:
            # derivata parziale w.r.t. self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0,
            # cioè self con i valori a 1.
            grad_x = cr_grad

            if x.data.shape != cr_grad.shape:
                # Ridurre il gradiente alla shape originale di x
                grad_x = grad_x.sum(axis=tuple(range(cr_grad.ndim - x.data.ndim)))

        if y.requires_grad:
            # derivata parziale w.r.t. other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1,
            # cioè other con i valori a 1.
            grad_y = cr_grad

            if y.data.shape != cr_grad.shape:
                # Ridurre il gradiente alla shape originale di y
                grad_y = grad_y.sum(axis=tuple(range(cr_grad.ndim - y.data.ndim)))
        
        return grad_x, grad_y
    

class CatBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: 'Tensor'
        chunk_shape: tuple
        indices: tuple
        x, chunk_shape, indices = self.ctx
    

class __ChunckBackward(BackwardFunction):
    # IDENTICA ad IndexBackward... Non troppo stonks
    def __call__(self, cr_grad: ArrayType) -> None:
        x: 'Tensor'
        chunk_shape: tuple
        indices: tuple
        x, chunk_shape, indices = self.ctx

        out_grad = x._device.zeros(x.shape())
        out_grad[indices] = cr_grad

        return out_grad, None


class ClampBackward(BackwardFunction):
    '''
    Propagates the gradient only to the entries which
    where not clipped during forward
    '''
    def __call__(self, cr_grad: ArrayType) -> None:
        x: 'Tensor'
        x, min_value, max_value = self.ctx

        cr_grad *= ((x.data >= min_value) & (x.data <= max_value))
        return cr_grad, None


class CloneBackward(BackwardFunction):
    '''
    This backward's only job is to pass the gradient to
    the original tensor. No computation is needed.
    '''
    def __call__(self, cr_grad: ArrayType) -> None:
        return cr_grad, None


class DivBackward(BackwardFunction):
    '''
    output = x/y
    d/dx(x/y) = d/dx(x) * 1/y = 1/y
    d/dy(x/y) = d/dy(1/y) * x = -x/y^2
    '''
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        grad_x, grad_y = None, None

        if x.requires_grad:
            # d/dx
            # usando la funzione evito errori div by zero
            grad_x = x._device.reciprocal(y.data + 1e-8) * cr_grad

        if y.requires_grad:
            # d/dy
            # per il momento non gestisco div by zero
            grad_y = -x.data / y._device.power(y.data, 2) * cr_grad

        return grad_x, grad_y


class EmbeddingBackward(BackwardFunction):
    '''
    NOTE: devo usare un add speciale perche' il gradiente e'
    particolare. x.data funge da lista di indici su cui sommare
    i corrispettivi ingressi del gradiente.
    Example::

        >>> arr = cp.array([1, 2, 3, 4, 5])
        >>> indices = cp.array([0, 2, 2])
        >>> values = cp.array([10, 20, 30])

        >>> cp.add.at(arr, indices, values)
        >>> arr
        [11, 2, 53, 4, 5]
    '''
    # sparse gradient
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        weights: "Tensor"
        x, weights = self.ctx
        
        if weights.requires_grad:
            if cr_grad.dtype != weights.dtype:
                cr_grad = cr_grad.astype(weights.dtype, copy=False)

            if weights._grad is None:
                weights._grad = weights._device.zeros(weights.data.shape, dtype=weights.dtype)
            # richiede un aggiornamento speciale
            #weights._grad[x.data] += cr_grad
            weights._device.add.at(weights._grad, x.data, cr_grad)   

        # pytorch ha della logica per x?
        # NO, x e' int!

        return None, None


class ExpBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        #x: "Tensor"
        x_exp: ArrayType
        x_exp = self.ctx[1]

        # d/dx( e^x ) = e^x
        # NOTE: l'esponente non si tratta come una composta
        cr_grad *= x_exp
        return cr_grad, None


class IndexBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        index: int|slice
        x, index = self.ctx
        
        # porto cr_grad alle dimensioni di self mettendo degli 0 di padding
        grad_x = x._device.zeros(x.data.shape, dtype=x.dtype)
        grad_x[index] = cr_grad
        return grad_x, None


class LogSoftmaxBackward(BackwardFunction):
    # Con bruteforce di CHADGPT e match di logica della softmax, ecco una
    # derivata di cui al momento non so assolutamente nulla sulla
    # sua derivazione matematica!
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        dim: int
        softmax: ArrayType
        x, dim, softmax = self.ctx

        cr_grad -= x._device.exp(softmax) * cr_grad.sum(axis=dim, keepdims=True)
        return cr_grad, None


class MaskedFillBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        #x: "Tensor"
        mask: ArrayType
        #x, mask = self.ctx
        mask = self.ctx[1]

        # NOTE: where(mask) =/= where(mask==0)!
        #return x._device.where(mask, 0, cr_grad), None
        cr_grad *= ~mask
        return cr_grad, None


class MatmulBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"     # tensor di shape (m, k)
        y: "Tensor"     # tensor di shape (k, n)
        x, y = self.ctx
        grad_x, grad_y = None, None

        #assert cr_grad.flags['C_CONTIGUOUS'] Fallisce
        # Penso che numpy/cupy ci pensi in automatico, ma e'
        # bene fare un gesto esplicito
        #if not cr_grad.flags['C_CONTIGUOUS']:
        #   cr_grad = x._device.ascontiguousarray(cr_grad)
        # NOTE: se vuoi metterci mani, fallo nella logica centralizzata,
        # ovvero in backpropagation()!
        
        # Gradiente rispetto ad x
        if x.requires_grad:
            if y.data.ndim > 1:
                # Caso X @ Y (vettore|matrice @ matrice)
                grad_x = cr_grad @ y.data.swapaxes(-1, -2)
            else:
                # Caso X @ y (matrice @ vettore)
                grad_x = x._device.outer(cr_grad, y.data.T)

            # Riduco grad_x se necessario
            if grad_x.shape != x.data.shape:
                grad_x = grad_x.sum(axis=tuple(range(grad_x.ndim - x.data.ndim)))

        # Gradiente rispetto ad y
        if y.requires_grad:
            if x.data.ndim > 1:
                # Caso X @ Y (matrice @ vettore|matrice)
                grad_y = x.data.swapaxes(-1, -2) @ cr_grad
            else:
                # Caso x @ Y (vettore @ matrice)
                grad_y = y._device.outer(x.data.T, cr_grad)
                # alternativa ad outer, fa la stessa cosa
                #grad_y = x.data.T[..., None] * cr_grad

            # Riduco grad_y se necessario
            if grad_y.shape != y.data.shape:
                grad_y = grad_y.sum(axis=tuple(range(grad_y.ndim - y.data.ndim)))

        #assert cr_grad.flags['C_CONTIGUOUS'] Fallisce ancora
        #assert grad_x.flags['C_CONTIGUOUS'] Non fallisce mai: i risultati
        # vengono giustamente messi in array contigui

        return grad_x, grad_y


class MeanBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
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

        #print(f"MEAN {cr_grad.shape}")

        # NOTE: con questo non funziona il transformer...
        # Che abbia fatto male i test di mean()?
        '''
        if not keepdim and dim is not None:
            cr_grad = x._device.expand_dims(cr_grad, dim)
        cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)
        '''
        if not keepdim:
            if dim is not None:
                cr_grad = x._device.expand_dims(cr_grad, dim)
            cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)
        else:
            cr_grad = cr_grad.sum(dim, keepdims=True)

        #print(f"MEAN {cr_grad.shape}")

        # d/dx( (x + y + ... + z) / N) = 1/N * [d/dx(x) + d/dx(y) + ... + d/dx(z)]
        # sto derivando w.r.t. x => le altre le tratto come costanti
        # = 1/N * [1 + 0 + ... + 0] = 1/N
        # dL/dx = grad * 1/N (dove N = size(self) = n ingressi. totali di self o dim)

        # non posso usare la in-place
        return cr_grad / x._device.size(x.data, dim), None


class MulBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        grad_x, grad_y = None, None


        if x.requires_grad:
            # d/dy( x*y ) => d/dy(y)*x = 1*x = x
            grad_x = y.data * cr_grad

            # Se x è stato broadcastato, somma lungo gli assi corretti
            if x.data.ndim != grad_x.ndim:
                # Identifica gli assi da sommare
                axis = tuple(range(grad_x.ndim - x.data.ndim))
                grad_x = grad_x.sum(axis=axis)
                #axes = [i for i in range(grad_x.ndim) if i >= len(x.data.shape) or x.data.shape[i] == 1]
                #grad_x = grad_x.sum(axis=tuple(axes)).reshape(x.data.shape)

        if y.requires_grad:
            # d/dy( x*y ) => d/dy(y)*x = 1*x = x
            grad_y = x.data * cr_grad

            # Se y è stato broadcastato, somma lungo gli assi corretti
            if y.data.ndim != grad_y.ndim:
                # Identifica gli assi da sommare
                axis = tuple(range(grad_y.ndim - y.data.ndim))
                grad_y = grad_y.sum(axis=axis)
                #axes = [i for i in range(grad_y.ndim) if i >= len(y.data.shape) or y.data.shape[i] == 1]
                #grad_y = grad_y.sum(axis=tuple(axes)).reshape(y.data.shape)

        return grad_x, grad_y


class NegBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        x, = self.ctx
        
        # d/dx(x * -1) = d/dx(x) * -1 = ones(len(x)) * -1
        return -cr_grad.reshape(x.data.shape), None
    

class NLLLossBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        target: "Tensor"
        eps: int
        x, target, eps = self.ctx

        # d/dx(y*log(x + eps)) = y * d/dx(log(x + eps)) = y * 1/(x + eps)
        cr_grad *= -target.data/(x.data + eps)
        return cr_grad, None


class PermuteBackward(BackwardFunction):
     def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        dims: tuple[int]
        x, dims = self.ctx

        # proprietà permutazioni: l'ordinamento degli indici di una permutazione
        # genera esattamente la sua permutazione inversa
        # => argsort: ordina l'array e ti restituisce in output l'array ordinato,
        # ma al posto degli elementi hai il loro index originale
        # NOTE: non ha senso fare operazioni come queste su GPU!
        inv_dims = np.argsort(dims)
        
        return cr_grad.transpose(inv_dims), None


class PowBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        exponent: int
        x, exponent = self.ctx

        # exponent * x._device.power(x.data, exponent-1) * cr_grad
        cr_grad *= exponent
        cr_grad *= x._device.power(x.data, exponent-1)
        return cr_grad, None


class ReLUBackward(BackwardFunction):
    # numpy nativamente fa (int32 * float32 = float64)
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        x, = self.ctx

        #cr_grad *= x._device.where(x.data > 0, 1, 0)
        cr_grad *= x._device.heaviside(x.data, 0)
        return cr_grad, None


class SELossBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        target: ArrayType
        x, target = self.ctx

        # 2*(x.data - target) * cr_grad
        cr_grad *= 2
        cr_grad *= (x.data - target)
        return cr_grad, None
    

class SigmoidBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        sigmoid: ArrayType
        sigmoid = self.ctx[1]

        # d/dσ = σ(x) * (1 - σ(x));
        # sigmoid * (1 - sigmoid) * cr_grad
        cr_grad *= sigmoid
        cr_grad *= (1 - sigmoid)
        return cr_grad, None


class SoftmaxBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        dim: int
        softmax: ArrayType
        x, dim, softmax = self.ctx

        '''
        la derivata della softmax è una matrice. Ogni colonna rappresenta una sua
        derivata completa w.r.t. un input
        '''
        #grad_x = softmax * (cr_grad - x._device.sum(cr_grad * softmax, axis=dim, keepdims=True))
        cr_grad -= x._device.sum(cr_grad * softmax, axis=dim, keepdims=True)
        cr_grad *= softmax
        return cr_grad, None


class SqrtBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        #x: "Tensor"
        x_sqrt: ArrayType
        x_sqrt = self.ctx[1]

        # d/dx = 1/(2*sqrt(x))
        cr_grad *= 1/(2*x_sqrt)
        return cr_grad, None
    

class SqueezeBackward(BackwardFunction):
    '''
    Returns the grad unsqueezed on the given dim
    '''
    def __call__(self, cr_grad: ArrayType) -> None:
        x: Tensor
        dim: int
        x, dim = self.ctx[1]
        
        # forse ho bisogno di fare device.expand_dims
        #return cr_grad.expand_dims(dim), None
        return x._device.expand_dims(cr_grad, dim), None


class SubBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        y: "Tensor"
        x, y = self.ctx
        grad_x, grad_y = None, None
        cr_grad = cr_grad.reshape(x.data.shape)

        if x.requires_grad:
            # derivata parziale w.r.t self, cioè x. d/dx(x-a)=1 ==> d/dx(x-y)=1-0, cioè self con i valori a 1.
            grad_x = cr_grad

        if y.requires_grad:
            # derivata parziale w.r.t other, cioè y. d/dy(a-y)=-1 ==> d/dy(x-y)=0-1, cioè other con i valori a -1.
            grad_y = -cr_grad

        return grad_x, grad_y


class SumBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        dim: int
        keepdim: bool
        x, dim, keepdim = self.ctx

        if not keepdim:
            if dim is not None:
                cr_grad = x._device.expand_dims(cr_grad, dim)
            cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)
        else:
            cr_grad = cr_grad.sum(dim, keepdims=True)

        # d/dx( (x + y + ... + z)) = [d/dx(x) + d/dx(y) + ... + d/dx(z)]
        # sto derivando w.r.t. x => le altre le tratto come costanti
        # = [1 + 0 + ... + 0] = 1
        # dL/dx = grad * 1
        return cr_grad, None


class StdBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        dim: int
        numerator: ArrayType
        denom: ArrayType
        keepdim: bool
        x, dim, numerator, denom, keepdim = self.ctx

        if not keepdim:
            if dim is not None:
                cr_grad = x._device.expand_dims(cr_grad, dim)
            # questo cr_grad sara' read-only dopo il broadcast
            cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)
        else:
            cr_grad = cr_grad.sum(dim, keepdims=True)

        # Derivata della deviazione standard:
        # d/dx(sqrt((x - mean)^2 / N))
        # = (x - mean) / (N * std)

        #return cr_grad * numerator / (denom + 1e-8), None
        numerator /= (denom + 1e-8)
        return cr_grad * numerator, None


class TanhBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        tanh: ArrayType
        tanh = self.ctx[1]

        cr_grad *= 1-tanh**2
        return cr_grad, None
    

class ToCopyBackward(BackwardFunction):
    '''
    This BackwardFunction is created only if .to() has
    changed the Tensor's device and/or dtype.
    Hence, the term "Copy" is used to highlight that
    .to() needed to create another Tensor
    '''
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        device: str
        dtype: type
        x, device, dtype = self.ctx

        # hai cambiato dtype ma non device
        if dtype != x.dtype and device == x.device:
            # caso di training
            cr_grad = cr_grad.astype(x.dtype, copy=False)
        # hai cambiato sicuramente il device e forse anche il dtype
        else:
            # a prescindere che tu mi chieda solo il cambio di
            # device o sia cambio device che dtype, la funzione
            # e' sempre questa

            if CUPY_AVAILABLE and cr_grad.__class__ == cp.ndarray:
                cr_grad = cr_grad.get()
            cr_grad = x._device.asarray(cr_grad, dtype=x.dtype)
        
        return cr_grad, None



class TransposeBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        dim1: int
        dim2: int
        x, dim1, dim2 = self.ctx

        # rifaccio la trasposta al contrario
        return cr_grad.swapaxes(dim2, dim1), None


class UnsqueezeBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        #x: "Tensor"
        dim: int
        dim = self.ctx[1]

        # converto cr_grad da output.shape a self.data.shape
        return cr_grad.squeeze(dim), None

    
class VarBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        dim: int
        mean: ArrayType
        n: int
        keepdim: bool
        x, dim, mean, n, keepdim = self.ctx

        if not keepdim:
            if dim is not None:
                cr_grad = x._device.expand_dims(cr_grad, dim)
            cr_grad = x._device.broadcast_to(cr_grad, x.data.shape)
        else:
            cr_grad = cr_grad.sum(dim, keepdims=True)

        # d/dx((x-x.mean)^2 / (N)) = 2(x-x.mean) / N
        # 2*mean/n * cr_grad
        mean *= 2
        mean /= n
        # non posso farlo, l'output e' read-only
        #cr_grad *= mean
        return cr_grad*mean, None


class ViewBackward(BackwardFunction):
    def __call__(self, cr_grad: ArrayType) -> None:
        x: "Tensor"
        x, = self.ctx

        # converto cr_grad da output.shape a self.data.shape
        return cr_grad.reshape(x.data.shape), None
    