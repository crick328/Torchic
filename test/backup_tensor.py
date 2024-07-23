import numpy as np
import cupy as cp
from typing import Callable, Self
from types import ModuleType
import warnings

from torchic.settings import DEFAULT
import torchic.backward as back_op


# Controlla se la rete non si traina con certi init per gli exploding gradients
# Puoi velocizzare ulteriormente il codice chiamando direttamente le variabili _

# (!) Bisogna costruire una sorta di documentazione/spiegazione delle implementazioni!
# Crea un file apposito con funzioni per visualizzare i parametri/"modelli mentali"
# della rete e l'apprendimento durante il training


# NOTE: fare Tensor(a+b) è meglio di Tensor(a)+Tensor(b) (oppure b)
# perché ci sono meno chiamate e non ci sono if. perciò, conviene sempre la prima forma
# se non si necessita della backward
# NOTE: NON fare operazioni che modificano data inplace (senza un return di un nuovo
# tensor) perche' altrimenti rompi la backward! Ogni tensore ha gli attributi che ruotano
# attorno al proprio .data. Se cambi quello (anche con un banale cambio di segno) svalvoli!

# NOTE: Pytorch usa molta più cpu e gpu!

# NOTE:
# Per ora un modo abbastanza sicuro per contorllare se tutto e' convertito dovrebbe
# essere il check del dtype dei gradienti all'interno dell'optimizer. Quelli interagiscono
# indirettamente con tutti i tensori creati, quindi se sono giusti dovrebbe essere tutto
# regolare
# RISOLTO IL MISTERO DI .where(): restituisce un array di int32, percio' quando si ritrova
# a fare int32*float32, le librerie scelgono di dare il risultato in float64 (LOL), presumo
# perche' e' il loro tipo di default.
# => se  gli ndarray coinvolti non hanno tutti lo stesso tipo, loro danno float64
# (se dtype e' implicito)!

# RECAP: il dtype del forward e' garantito essere uguale al DEFAULT["dtype"] grazie
# a __data_manager() (cosa che accade sempre perche' gli ouutput sono castati a Tensor),
# mentre per la backward casto i singoli grad in _backward() con .astype(DEFAULT['dtype'], copy=False)

# non mi piace __data_manager(). Vorrei alleggerirla. I return non sembrano avere effetto,
# ma e' da indagare. Vedi se la logica puo' essere alleggerita perche' sull'init
# del Tensor c'e' in ballo tanta performance! 

# NOTE: le operazioni come __neg__() non so se sono implementate bene... Per fare il
# negativo di self perdi tutti i dati di grad, _is_leaf, ecc... Ci vorrebbe un
# metodo comodo per quando voglio creare uno stesso Tensor con gli stessi
# attributi (mi viene in ment il case self.__class di data manager)

# NOTE: si puo' gestire il dtype del child tensor sempre in modo sicuro tramite
# le mie funzioni, ma dinamicamente. Il dtype puo' essere passato come parametro
# quando si crea un tensor (default=None => settings). Ogni operazione della
# classe tensor deve chiamare una funzione (che puo' essere un class method)
# che prende in input il dtype dei parent e da' in output il dtype da castare.
# Si puo' anche chiamare nel metodo ._backward perche' tanto lavora direttamente
# sui .data

# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

class Tensor:
    '''
    Classe Tensor.

    Attributes:
        data: `ndarray` che funge da scalare, vettore, matrice o elementi a più dimensioni.
        _requires_grad: Può essere modificato solo se il tensore è leaf (`is_leaf = True`).
            Se si assegna una `grad_fn`, viene automaticamente settato a `True`.
        _device: contiene il modulo `np` o `cp`. Se si legge il suo valore, grazie al
            getter si riceve `"cpu"` per `np` e `"cuda"` per `cp`.
        _is_leaf: flag to mark if a tensor is child or created directly by the user.
            True per convenzione se `requires_grad = False`: Altrimenti,
            è `True` se il tensor non è frutto di operazioni.
            Automaticamente impostato a `False` se viene assegnata una `grad_fn`. 
        _grad_fn (`BackwardFunction`):
            La variabile è `None` se il tensor è leaf. Viene automaticamente reimpostata
            a None dopo ogni passaggio.
        grad: contiene il gradient della loss w.r.t self. Calcolato tramite
            la grad_fn se requires_grad è `True`.
            Viene automaticamente reimpostata a None se il tensor non è leaf.
    '''
    __slots__ = ("data", "_requires_grad", "_device", "_is_leaf", "_grad_fn", "_grad")

    def __init__(self, data, requires_grad: bool=False, device: str=None, dtype: np.dtype=None):
        '''
        Returns a `Tensor` on `device` with dtype `torch.settings.DEFAULT["dtype"]`.

        Args:
            data (ArrayLike): Tenno, skuuu!.
            requires_grad (bool, optional): flag request the update of the gradient.
                Default is False.
            device (str, optional): can be either `"cpu"` or `"cuda"`.
                Default is `torch.settings.DEFAULT["device"]`.
        '''
        #self.data, self._device = self.__data_manager(data, device)
        #self.data = self._to_dtype(data, DEFAULT['dtype'])
        self.data, self._device = self._to_device(data, device, dtype)
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._grad_fn = None
        self._grad = None


    # ===== PROPERTIES =====

    # NOTE: se rendi property anche grad puoi fare stampare l'alert quando
    # cerchi di leggere il grad di un leaf tensor (che logicamente è None)
    # ed evitarti infartini nel debugging

    @property
    def requires_grad(self) -> bool:
        """The setter doesn't allow to change `requires_grad` if `is_leaf=False`,
        which means that this Tensor is part of the backpropagation.
        """
        return self._requires_grad
    
    @property
    def device(self) -> str:
        """Under the hood, `_device` contains the module to operate on the
        requested device (so `_device=np|cp`). This property just returns
        the corrispondent device name as string (`cp`->`"cuda"`, `np`->`"cpu"`).
        
        The setter methods doesn't accept any value, making this attribute not
        writable. If the user wants to change it, they can use the `.to()` method.
        """
        if self._device is cp:
            return "cuda"
        else:
            return "cpu"
    
    @property
    def is_leaf(self) -> bool:
        """The setter doesn't accept any value, making this attribute not writable."""
        return self._is_leaf
    
    @property
    def grad_fn(self) -> Callable:
        """The setter method doesn't accept `None` as value. Setting a `grad_fn`
        will also set `is_leaf=False` and `requires_grad=True`.
        """
        return self._grad_fn
    
    @property
    def grad(self) -> bool:
        """Property created to alert the user when he tries to read
        the grad of a leaf tensor, which is None because it's not
        calculated."""
        if not self.is_leaf:
            warnings.warn("you are trying to read the gradient of a not leaf tensor, "
                          "but, by default, is not calculated. Call .retain_grad() "
                          "(stocazzo, non l'ho ancora implementato) if you want "
                          "to have it.")
        return self._grad
    
    @property
    def dtype(self) -> bool:
        """Sos"""
        return self.data.dtype
    

    # definisco i setter
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        # impedisco all'utente di cambiare la flag se il tensor fa parte della backprop
        if not self._is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables.")
        else:
            self._requires_grad = value

    @device.setter
    def device(self, value) -> None:
        # impedisco di cambiare la flag
        raise AttributeError (
            "attribute 'device' of Tensor object is not writable."
            "Use '.to()' if you want to change the current device of this tensor."
            )
    
    @is_leaf.setter
    def is_leaf(self, value) -> None:
        # impedisco di cambiare la flag
        raise AttributeError ("attribute 'is_leaf' of Tensor object is not writable.")

    @grad_fn.setter
    def grad_fn(self, value: bool) -> None:
        if value is not None:
            if self._grad_fn is None:
                # aggiornato a False perché si assegna una grad_fn
                self._is_leaf = False
                # da qua in avanti, requires_grad non sarà più modificabile perché
                # non è più considerato leaf tensor
                self._requires_grad = True
            else:
                raise RuntimeError(
                    "Mi stai chiedendo di sovrascrivere una grad function non nulla "
                    "con un valore non nullo, ma questo romperebbe la backpropagation."
                    )
        self._grad_fn = value

    @grad.setter
    def grad(self, value) -> None:
        self._grad = value

    @dtype.setter
    def dtype(self, value) -> None:
        # impedisco di cambiare la flag
        raise AttributeError ("attribute 'dtype' of Tensor object is not writable.")
    

    # ===== BASE FUNCTIONS =====

    # questa funzione non e' chiamata nel training
    def to(self, *args):
        match args:
            # inserito solo il device
            case (str(),):
                self.data, self._device = self._to_device(self.data, args[0])
            # unico caso di doppio input supportato da torch
            # device e dtype
            case (str(), np.dtype):
                self.data, self._device = self._to_device(self.data, args[0], args[1])
            # solo dtype
            case (np.dtype,):
                self.data = self._to_dtype(self.data, args[0])
            case _:
                print('no match')
        # per match dell'API di Torch
        return self


    @staticmethod
    def _to_device(data, device: str, dtype: np.dtype=None) -> tuple[np.ndarray|cp.ndarray, ModuleType]:
        """Function that returns a tensor on the requested device as
        dtype, if specified.
        """
        
        # di norma si verifica se la funzione e' chiamata
        # dall'init del Tensor
        if device is None:
            device = DEFAULT["device"]

        if dtype is None:
            dtype = DEFAULT["dtype"]

        match(device):
            case "cuda":
                if data.__class__ != cp.ndarray:
                    return cp.array(data, dtype), cp
            case "cpu":
                # se e' un np e non e' cp tiro subito dritto al return
                if data.__class__ != np.ndarray and data.__class__ != cp.ndarray:
                    return np.array(data, dtype), np
                # controllo se e' un CuPy perche' devo usare .get()
                # condizione non eseguita durante il training
                elif data.__class__ == cp.ndarray:
                    return np.array(data.get(), dtype), np
            case _:
                # questo caso non dovrebbe mai verificarsi a meno
                # che l'utente non setta un device di default
                # che non e' accettabile
                raise ValueError(
                    f"Expected 'cuda' or 'cpu', got {device} instead."
                    )
            
        # se mi viene chiesta una conversione allo stesso tipo
        # restituisco l'input con il giusto dtype
        return Tensor._to_dtype(data, dtype), cp if data.__class__ == cp.ndarray else np
                
    @staticmethod
    def _to_dtype(data: np.ndarray|cp.ndarray, dtype: np.dtype):
        return data.astype(dtype, copy=False)


    def expand(self, *shape) -> Self:
        '''
        Returns self with values copied in the new dim.
        
        Example:
        [1,2,3] -> broadcast_to([2,3]) -> [[1,2,3],[1,2,3]]
        '''
        output = Tensor(self._device.broadcast_to(shape))
        # non so che pesci prendere per la backward
        raise RuntimeError("Hai chiamato una funzione in beta.")

    def unsqueeze(self, dim: int) -> Self:
        '''
        Insert a new dimension in the given dim.

        Example:
        [1,2] -> unsqueeze(0) -> [[1,2]] (new shape is 1x2)
        [1,2] -> unsqueeze(1) -> [[1],[2]] (new shape is 2x1)
        '''
        # NOTE: np ha squeeze, pero' "unsqueeze" e' expand_dim
        output = Tensor(self._device.expand_dims(self.data, axis=dim))

        if self.requires_grad:
            output.grad_fn = back_op.UnsqueezeBackward(self, dim)
        return output

    def view(self, *shape) -> Self:
        '''
        Returns self with data rearranged to match the given shape.
        The desired shape has to have the same size (n of total elem)
        of self.
        '''
        output = Tensor(self.data.reshape(shape))

        if self.requires_grad:
            output.grad_fn = back_op.ViewBackward(self)
        return output
    
    def transpose(self, dim1: int, dim2: int) -> Self:
        output = Tensor(self.data.transpose(dim1, dim2))
        
        if self.requires_grad:
            output.grad_fn = back_op.TransposeBackward(self, dim1, dim2)
        return output
    
    def contiguous(self) -> Self:
        '''
        Riarrangia i dati in memoria. Questa operazione NON richiede backward.
        '''
        output = Tensor(self._device.ascontiguousarray(self.data))

        if self.grad_fn is not None:
            output.grad_fn = self.grad_fn
        return output
    
    def shape(self) -> tuple[int, int]:
        # (row, column)
        return self.data.shape
    
    def dim(self) -> int:
        '''n° of dim (scalar 0, vector 1, matrix 2, ecc.)'''
        return self.data.ndim
    
    def size(self, dim: int=None) -> int:
        # numero di ingressi
        if not dim:
            return self.data.size
        else:
            self._device.size(self.data, axis=dim)
    
    def backward(self, gradient: Self=None) -> None:
        '''
        gradient: array-like richiesto se self.data non è uno scalare (cioè size != 1).

        Funzione richiamata sull'ultimo tensor (solitamente loss) per far partire la backprop.
        Inizia il processo di richiamo a catena delle funzioni (grad_fn) contenenti le derivate
        parziali dei layer.
        La backprop non può essere fatta più volte di fila perché il grafo viene eliminato ad
        ogni passaggio per liberare memoria (le grad_fn vengono impostate a None) a meno che
        non si lavori con retain_graph o create_graph, che non ho implementato :) .
        '''
        if gradient is not None and gradient.__class__ is not Tensor:
            raise ValueError(
                f"Gradient has to be Tensor or None, received {type(gradient)} instead."
            )

        if self.grad_fn is not None:
            # se l'utente passa un gradient
            if gradient is not None:
                if gradient.device != self.device:
                    raise RuntimeError(
                        f"gradient has to be on the same device as receiving tensor ({self.device},"
                        f"got {gradient.device} instead.)"
                    )
                if self.data.shape != gradient.data.shape:
                    raise RuntimeError(
                        f"need gradient of shape {self.data.shape}, got {gradient.data.shape} instead."
                    )
                
                self.grad_fn(gradient.data)

            elif self.data.size == 1:
                # => posso calcolarlo implicitamente
                data, _ = Tensor._to_device(np.array(1.), device=self.device)
                self.grad_fn(data)
            else:
                raise RuntimeError(
                    f"grad can be implicitly created only for scalar outputs because is not ambiguous."
                    "Use .sum() or .mean() before calling .backward(), or give an all-ones tensor of shape {self.data.shape}."
                    )
            
            # elimino la reference in modo che il garbage collector possa cancellare le variabili inutili
            self.grad_fn = None

        elif self.is_leaf:
            raise RuntimeError(
                "You called .backward() on a leaf tensor which, by definition, doesn't have a grad_fn."
                )
        else:
            raise RuntimeError(
                "You called .backward() two times in a row, but can't be computed again because the computational graph was freed."
                )


    def _run_backward(self, grad: np.ndarray|cp.ndarray) -> None:
        # requires_grad: attiva il sistema di tracciamento delle backward.
        # Se requires_grad=False, is_leaf puo' essere solo True.
        # Se requires_grad=True, is_leaf e' settato a False.
        # requires_grad decide solo se salvare una backward_function e se
        # calcolare il suo gradiente.
        # is_leaf decide solo se aggiornare il ._grad del tensore o continuare
        # la backprop.
        grad = grad.astype(DEFAULT["dtype"], copy=False)
        
        if self.is_leaf:
            # aggiorno il grad e fine

            # caso che metto prima perche' e' il piu' frequente
            if self._grad is not None:
                self._grad += grad
            else:
                # questo puo' essere inutile perche' dopo
                # lo imposto a None se non e' leaf, ma cosi'
                # mi sara' piu' facile implementare retain_grad
                self._grad = grad
        else:
            # se i tensor hanno parent, continuo la backprop

            # NOTE FONDAMENTALE: self.grad e' diverso da grad!
            self.grad_fn(grad)
            # DA QUI IN POI il codice viene runnato DOPO la fine di tutte
            # le call partite da grad_fn()
            self.grad_fn = None

    
    # ===== EXPLICIT OPERATIONS =====

    # NOTE: non ho gestito il caso A @ x
    def matmul(self, other: Self) -> Self:
        # self: vector, other: matrix
        output = Tensor(self.data @ other.data)
        
        if self.requires_grad or other.requires_grad:
            output.grad_fn = back_op.MatmulBackward(self, other)
        return output
    
    # NOTE: la backward NON supporta piu' axis!
    def sum(self, dim: int=None, keepdim: bool=False) -> Self:
        """
        return: sum of data as scalar Tensor object
        """
        output = Tensor(self._device.sum(self.data, axis=dim, keepdims=keepdim))

        if self.requires_grad:
            #assert dim==None, ValueError(
            #    "mi hai chiesto di fare una sum su una dimensione, ma "
            #    "la mia backpropo NON supporta queta funzione."
            #)
            output.grad_fn = back_op.SumBackward(self, dim)
        return output
    
    def mean(self, dim: int=None, keepdim: bool=False) -> Self:
        """
        Returns the mean of all dims as scalar Tensor object if dim is None,
        else only of the selected dim.

        Args:
            dim (int): 
        """
        output = Tensor(self._device.mean(self.data, axis=dim, keepdims=keepdim))

        # in questi casi c'e' un po' di ridondanza
        if self.requires_grad:
            output.grad_fn = back_op.MeanBackward(self, dim)
        return output
    
    def var(self, dim: int=None, keepdim=False, unbiased=True):
        mean = self.data - self.data.mean(axis=dim, keepdims=True)
        n = self._device.size(self.data, dim)

        if unbiased:
            n -= 1

        var = self._device.power(mean, 2).sum(axis=dim, keepdims=True) / n
        output = Tensor(var.squeeze()) if not keepdim else Tensor(var)
        
        if self.requires_grad:
            output.grad_fn = back_op.VarBackward(self, dim, mean, n)
        return output

    
    def std(self, dim: int=None, keepdim=False, unbiased=True):
        '''
        Returns standard deviation calculated with Bessel's correction
        (which is the -1 at denominator) if unbiased=True.

        std = sqrt((x-x.mean)^2 / (N-1))
        std = sqrt(var(x))
        '''
        #x = self.data - self.data.mean(axis=dim, keepdims=keepdim)
        mean = self.data - self.data.mean(axis=dim, keepdims=True)
        n = self._device.size(self.data, dim)

        if unbiased:
            n -= 1

        std = self._device.sqrt( self._device.power(mean, 2).sum(axis=dim, keepdims=True) / n )
        output = Tensor(std.squeeze()) if not keepdim else Tensor(std)
        
        if self.requires_grad:
            output.grad_fn = back_op.StdBackward(self, dim, mean, n*std)
        return output
        
    
    # ===== IMPLICIT CLASS FUNCTIONS ===== 
    
    #def __matmul__(self, other):
        #return self.matmul(other)

    def __and__(self, other: Self) -> Self:
        # Nota: questa operazione non ha backward anche in
        # PyTorch perche' e' definita solo per gli int
        return Tensor(self.data & other.data, dtype=self.dtype)

    def __add__(self, other: Self) -> Self:
        output = Tensor(self.data + other.data)
        
        if self.requires_grad or other.requires_grad:
            output.grad_fn = back_op.AddBackward(self, other)
        return output
    
    def __sub__(self, other: Self) -> Self:
        output = Tensor(self.data - other.data)   

        if self.requires_grad or other.requires_grad:
            output.grad_fn = back_op.SubBackward(self, other)
        return output
    
    def __mul__(self, other: Self) -> Self:
        output = Tensor(self.data * other.data)

        if self.requires_grad or other.requires_grad:
            output.grad_fn = back_op.MulBackward(self, other)
        return output
    
    def __truediv__(self, other: Self) -> Self:
        output = Tensor(self.data / other.data)
            
        if self.requires_grad or other.requires_grad:
            output.grad_fn = back_op.DivBackward(self, other)
        return output
    
    
    def __neg__(self) -> Self:
        # output = array * -1
        output = Tensor(-self.data)

        if self.requires_grad:
            output.grad_fn = back_op.NegBackward(self)
        return output
    
    
    def __pow__(self, exponent: int) -> Self:
        output = Tensor(self._device.power(self.data, exponent))

        if self.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = back_op.PowBackward(self, exponent)
        return output
    
    # ===== CLASS UTILITY =====

    def __hash__(self) -> int:
        return hash(self.data) + hash(self._device) \
                + hash(self._requires_grad) + hash(self._is_leaf) \
                + hash(self._grad_fn) + hash(self._grad)

    def __eq__(self, other):
        # per confronti del tipo Tensor1 == Tensor2
        if isinstance(other, Tensor):
            return self._device.allclose(self.data, other.data)
        # posso farlo per np e cp, ma non so se lo usero' mai e per ora evito
        return NotImplementedError(
            f"Comparison with class {other.__class__} not yet supported :("
            )

    #def __neq__(self, other):
        # per confronti del tipo Tensor1 == Tensor2
        #if isinstance(other, Tensor):
            #return not self._device.allclose(self.data, other.data)
        # posso farlo per np e cp, ma non so se lo usero' mai e per ora evito
        #return NotImplementedError(f"Comparison with class {other.__class__} not yet supported :(")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        # funzione chiamata quando stampo il Tensor
        print_str = self._device.array2string(self.data, threshold=self._device.inf, max_line_width=self._device.inf, separator=',', precision=4).replace('\n', '')
        print_str = print_str.replace(',', ", ")

        if self.grad_fn is not None:
            print_str = print_str + f", grad_fn={self.grad_fn}"
        elif self.requires_grad:
            print_str = print_str + f", requires_grad={self.requires_grad}"

        return f"Tensor({print_str})"