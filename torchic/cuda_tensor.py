from torchic.settings import CONFIG, CUPY_AVAILABLE, get_current_backend
from typing import Callable, Optional, Union, Any, overload
from torchic.dtypes import choose_dtype, DTYPE_MAP_V2
import torchic.autograd.functions as back_op
from torchic.autograd.functions import register_backward_if_needed
from torchic.autograd.backprop import start_backprop
from torchic.amp import autocast_operation
import numpy as np
import warnings
from types import ModuleType
import sys


if CUPY_AVAILABLE:
    import cupy as cp



ArrayType = Union[np.ndarray, Optional['cp.ndarray']]

# (!) Bisogna costruire una sorta di documentazione/spiegazione delle implementazioni!
# Crea un file apposito con funzioni per visualizzare i parametri/"modelli mentali"
# della rete e l'apprendimento durante il training


# NOTE: fare Tensor(a+b) è meglio di Tensor(a)+Tensor(b) (oppure b)
# perché ci sono meno chiamate e non ci sono if. Perciò, conviene sempre la prima forma
# se non si necessita della backward

# NOTE: NON fare operazioni che modificano data in-place (senza un return di un nuovo
# tensor) perche' altrimenti rompi la backward! Ogni tensore ha gli attributi che ruotano
# attorno al proprio .data. Se cambi quello (anche con un banale cambio di segno) svalvoli!

# NOTE: Si puo' pensare di togliere la limitazione che i tensori possono effettuare
# operazioni solo tra di loro. Fai che, se il tipo ricevuto non e' un tensor,
# lo casti come tale e poi se la vede la funzione. Tanto l'utente non fa altro
# che creare un tensor prima di fare l'op, tantovale automatizzare


# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

class Tensor:
    __slots__ = ("data", "_requires_grad", "_device", "_is_leaf", "_grad_fn", "_grad")

    def __init__(self, data, requires_grad: bool=False, device: str=None, dtype: np.dtype=None, _fast_init=True):
        '''
        Returns a `Tensor` on given or inferred `device` with given or inferred `dtype`.
        User values overrides the default behaviors.

        Args:
            data (ArrayLike): Tenno, skuuu!
            requires_grad (bool, optional): flag to request the computation of
                the gradient. Default is False.
            device (str/None, optional): can be either `"cpu"`, `"cuda"` or
                `None`, which defaults to the library's default settings,
                (`torch.settings.CONFIG["default_device"]`).
                Default is `None`.
            dtype (np.dtype/None, optional): NumPy data type to cast `data` to.
                It can be `None`, which allows to get the dtype from `data`
                (casted to `torch.settings.CONFIG["precision"]`).
                Default is `None`.
        '''
        if _fast_init:
            self.data, self._device = self._fast_init(data, device, dtype)
            # il grad di un tensore intero non ha senso
            #if requires_grad and self._device.issubdtype(self.data.dtype, self._device.integer):
            if requires_grad and self.data.dtype.kind != 'f':
                raise RuntimeError(
                    "Si puo' richiedere l'aggiornamento dei gradienti "
                    "solo per tensors con dtype float o complex."
                )
        else:
            self.data, self._device = self._manage_data(data, requires_grad, device, dtype)
        
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._grad_fn = None
        self._grad = None


    # ===== PROPERTIES =====
    # nei getter spiego le property, come da docstring

    @property
    def requires_grad(self) -> bool:
        """It's a flag to request the *computation* of the gradient.
        It can be modified only if self is not leaf.
        If a `grad_fn` is assigned, `requires_grad` will be automatically
        set to `True` because it has to pass the gradient
        
        The setter doesn't allow to change `requires_grad` if `is_leaf=False`,
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
        return "cuda" if CUPY_AVAILABLE and self._device is cp else "cpu"
    
    @property
    def is_leaf(self) -> bool:
        """It's an internal flag to log if self is or isn't a leaf.
        If `False`, requests the *update* of the gradient.
        It's automatically set to `False` if a `grad_fn` is assigned because
        that means that self was obtained through an operation *and* one of the
        parent has `requires_grad` set to `True`.
        By convention, it's set to `True` if self does not requires the gradient.
        Default value is `True`.
        
        The setter doesn't accept any value, making this attribute not writable."""
        return self._is_leaf
    
    @property
    def grad_fn(self) -> Callable:
        """If one of the parent of self has `requires_grad` set to `True`,
        stores the `BackwardFunction` corresponding to the operation that
        generated self, otherwise is `None`.
        After a backward pass, `grad_fn` is set to `None` when it's not
        needed anymore for other operations in the same backward pass.
        
        The setter method doesn't accept `None` as value. Setting a `grad_fn`
        will also set `is_leaf=False` and `requires_grad=True`.
        """
        return self._grad_fn
    
    @property
    def grad(self) -> bool:
        """Contains the gradient of the loss w.r.t. self. It's
        computed during the backward pass with the `BackwardFunction` stored
        in `grad_fn` using the chain rule.

        The setter warns the user if he tries to read the `grad`
        of a leaf tensor, which should be `None` because it's not
        updated."""
        if not self.is_leaf:
            warnings.warn("\nYou are trying to read the gradient of a tensor which is "
                          "NOT leaf, but, by default, is not updated. Call .retain_grad() "
                          "(stocazzo, non l'ho ancora implementato) if you want "
                          "to save it.\n")
        return self._grad
    
    @property
    def dtype(self) -> bool:
        """Wrapper to read the dtype of data."""
        return self.data.dtype
    
    @property
    def T(self) -> 'Tensor':
        """Wrapper to transpose the data for 2D tensors.
        It's the equivalent of self.transpose(0, 1)"""
        if self.dim() != 2:
            raise RuntimeError(
                ".T is usable only on 2D tensors."
            )
        return self.transpose(0,1)
    

    # definisco i setter
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        # impedisco all'utente di cambiare la flag se il tensor fa parte della backprop
        if not self._is_leaf:
            raise RuntimeError(
                "you can only change requires_grad flags of leaf variables."
                )
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
        raise AttributeError (
            "attribute 'is_leaf' of Tensor is an internal flag and it's not writable."
            )

    @grad_fn.setter
    def grad_fn(self, value: bool) -> None:
        raise AttributeError (
            "attribute 'grad_fn' of Tensor is an internal field and it's not writable."
            )
        # in teoria cosi' permetto all'utente di settare un valore a caso
        # quando invece dovrei blindare grad_fn. Per ora lo lascio
        # comunque cosi' perche' mi e' comodo settare in automatico
        # tutto quanto, anche se effettivamente mi basterebbe spostare
        # il cambio dei valori a register_backward_if_needed...
        # sarebbe anche piu' pulito in quanto questa e' logica dell'autograd
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
    

    # ===== ?? FUNCTIONS =====
    
    # Overload per type hinting e ease of use
    # L'asterisco serve da "barriera": cio' che c'e' dopo
    # deve essere passato specificandone il nome
    @overload
    def to(self, dtype: type, *, inplace: bool=False) -> 'Tensor': ...
    @overload
    def to(self, device: str, *, inplace: bool=False) -> 'Tensor': ...
    @overload
    def to(self, device: str, dtype: type, *, inplace: bool=False) -> 'Tensor': ...

    def to(self, *args: Any, inplace=False) -> 'Tensor':
        '''
        NOTE: this function has always inplace=True if it is called on
        a nn.Module (which is also done by PyTorch and I see why)
        '''
        #device=self.device
        #dtype=self.dtype

        match args:
            # solo dtype
            case (type() | np.dtype(),):
                if self.dtype == args[0]:
                    return self
                device = self.device
                dtype = args[0]
            # solo il device
            case (str(),):
                if self.device == args[0]:
                    return self
                device = args[0]
                dtype = self.dtype
            # unico caso di doppio input supportato da torch
            case (str(), type() | np.dtype()):
                device, dtype = args
                if self.device == device and self.dtype == dtype:
                    return self
            case _:
                raise ValueError(f"No match for args {args}")
            
        if inplace:
            print(device)
            self.data, self._device = Tensor._fast_init(self.data, device, dtype)
            return self
        
        output = Tensor(self.data, device=device, dtype=dtype)
        
        register_backward_if_needed(output, back_op.ToCopyBackward(self, device, dtype))
        return output
    
    def __faster_to_maybe(self, *args: Any, inplace=False) -> 'Tensor':
        '''
        NOTE: this function has always inplace=True if it is called on
        a nn.Module (which is also done by PyTorch and I see why)
        '''
        n_args = len(args)

        if n_args == 1:
            # e' piu' veloce che non accedere sempre all'index
            arg = args[0]

            # solo dtype
            if isinstance(arg, (type, np.dtype)):
                if self.dtype == arg:
                    return self
                device, dtype = self.device, arg
            # solo il device
            elif isinstance(arg, str):
                if self.device == arg:
                    return self
                device, dtype = arg, self.dtype
            else:
                raise ValueError(f"No match for args {args}")
        
        elif n_args == 2:
            arg0, arg1 = args

            # unico caso di doppio input supportato da torch
            if isinstance(arg0, str) and isinstance(arg1, (type, np.dtype)):
                if self.device == arg0 and self.dtype == arg1:
                    return self
                device = arg0
                dtype = arg1
            else:
                raise ValueError(f"No match for args {args}")
        else:
            raise ValueError(f"No match for args {args}")
            
        if inplace:
            print(device)
            self.data, self._device = Tensor._fast_init(self.data, device, dtype)
            return self
        
        output = Tensor(self.data, device=device, dtype=dtype)
        
        register_backward_if_needed(output, back_op.ToCopyBackward(self, device, dtype))
        return output
      
    
    # ==== STATIC FUNCTION ====
    # This function doesn't require self, so it's static
    @staticmethod
    def _fast_init(data: ArrayType, device=None, dtype=None) -> tuple[ArrayType, ModuleType]:
        '''
        Manages `data` during `Tensor`'s init.
        First addresses the device, then the dtype. User's params always
        prevale over internal logic.

        When `None`, the `device` is taken from `CONFIG["default_device"]`
        (which can be setted by `torchic.settings.set_default_device()`).
        When `None`, `data.dtype` is aligned with `CONFIG["precision"]`
        (which can be setted by `torchic.settings.set_default_precision()`)
        **only if data.dtype.precision > default_precision**. For example,
        nothing is changed if `data.dtype=int16` and `default_prec=32`, while
        it would be corrected if `data.dtype=int64`. With this approach,
        Tensors can maintain their native precision under AMP.
        '''
        # 1) non deve essere dtype > default precision (abbasso solo
        #    la precisione, non casto di default a float) 
        # 2) devo portare il tensore sul device (tendezialmente lo e' gia')
        # Secondo me conviene prima gestire il dtype e poi eventualmente
        # spostarlo su GPU

        #if CUPY_AVAILABLE and data.__class__ == cp.ndarray and device==None and dtype==None:
            #if data.dtype.itemsize*8 > CONFIG["precision"]:
                # mantengo il tipo di dato e casto alla precisione richiesta
                #dtype = DTYPE_MAP_V2[data.dtype.kind][CONFIG["precision"]]
                #data = data.astype(dtype, copy=False)
            #return data, cp
        

        # == Device Logic ==
        if device is None:
            device = CONFIG["default_device"]
            # so che cupy e' ON o l'utente non avrebbe potuto settarlo

        match device:
            case "cuda":
                if not CUPY_AVAILABLE:
                    raise RuntimeError("CuPy is not available. Install it to use CUDA features.")
                module = cp
            case "cpu":
                module = np
            case "dynamic":
                # provo ad inferirlo automaticamente dal tipo di .data
                # Sarebbe carino avere questo setting di default per
                # i tensor non leaf, ma per ora non ho trovato modo
                module = cp if CUPY_AVAILABLE and type(data) == cp.ndarray else np
            case _:
                # non dovrebbe mai verificarsi (a meno che l'utente non 
                # scrive cagate quando istanzia un Tensor)
                raise ValueError(
                    f"Expected 'cuda' or 'cpu', got {device} instead."
                    )
            
        # se data e' diverso 
        if data.__class__ != module.ndarray:
            # ti porto sul device e ti casto
            # .asarray e' meglio di .array perche' se puo' restituisce self

            # controllo se e' un CuPy perche' devo usare .get()
            # condizione non eseguita durante il training
            if CUPY_AVAILABLE and data.__class__ == cp.ndarray:
                data = data.get()
            data = module.asarray(data, dtype=dtype)

        # === Dtype Logic ===
        if dtype is None:
            # se il dtype ha precisione superiore rispetto a quella richiesta
            if data.dtype.itemsize*8 > CONFIG["precision"]:
                # mantengo il tipo di dato e casto alla precisione richiesta
                dtype = DTYPE_MAP_V2[data.dtype.kind][CONFIG["precision"]]
            else:
                return data, module
            
        if data.dtype != dtype:
            data = data.astype(dtype, copy=False)
        
        return data, module


    def _manage_data(self, data: ArrayType, requires_grad: bool=None, device=None, dtype=None):
        '''
        Requires_grad viene passato solo durante l'__init__ per verificare se
        quel dtype puo' richiedere l'aggiornamento dei gradienti. Tutte le
        altre volte (cioe' quando si cambia device), questo controllo
        non e' piu' necessario, e infatti si salta perche'
        requires_grad e' None, shortciruitando l'if

        Args:
            device (str, optional): if not given, defaults to the library's
                default device.
            dtype (np.dtype, optional): if not given, is inferred from
                `data`, but it's recasted to the library's default
                precision.
        '''
        #if data.__class__ == Tensor:
            # non dovrebbe mai ricevere un tensore
            #return data.data, data.device

        if device is None:
            device = CONFIG["default_device"]
        # bottleneck
        data, module = self.__to_device(data, device)

        # di default, numpy produrra' sempre operazioni con tipo
        # di dato a 8 byte, quindi devo ricastarli.
        # Con il nuovo approccio, mantengo il tipo dell'operazione,
        # evitando di castare int a float o fare cose strane
        # Gestisce anche la mixed precision
        if dtype is None:
            #dtype = dtype_map[get_type(data)][CONFIG["precision"]]
            dtype = choose_dtype(data.dtype)
            #print(f"chosen dtype {dtype}, {data.dtype}")

        # il grad di un tensore intero non ha senso
        if requires_grad and module.issubdtype(dtype, module.integer):
            raise RuntimeError(
                "Si puo' richiedere l'aggiornamento dei gradienti "
                "solo per tensors con dtype float o complex."
            )
        
        #print(data.dtype)
        # e' quasi sempre vero
        data = self.__to_dtype(data, dtype)
        return data, module


    # ==== STATIC FUNCTIONS ====
    # These functions don't require self, so they are static

    @staticmethod
    def __to_device(data, device: str):
        if device == "dynamic":
            # provo ad inferirlo automaticamente se l'utente
            # non ha esplicitamente cambiato il device di default
            device = "cuda" if CUPY_AVAILABLE and type(data) == cp.ndarray else "cpu"

        match device:
            case "cuda":
                if not CUPY_AVAILABLE:
                    raise RuntimeError("CuPy is not available. Install it to use CUDA features.")
                
                if data.__class__ != cp.ndarray:
                    data = cp.array(data, copy=False)
                module = cp
                    
            case "cpu":
                # controllo se e' un CuPy perche' devo usare .get()
                # condizione non eseguita durante il training
                if CUPY_AVAILABLE and data.__class__ == cp.ndarray:
                    data = np.array(data.get())
                elif data.__class__ != np.ndarray:
                    data = np.array(data)
                module = np
                
            case _:
                # questo caso non dovrebbe mai verificarsi a meno
                # che l'utente non setta un device di default
                # che non e' accettabile
                raise ValueError(
                    f"Expected 'cuda' or 'cpu', got {device} instead."
                    )
        return data, module

                
    @staticmethod
    def __to_dtype(data: ArrayType, target_dtype: np.dtype):
        # NOTE: e' proprio necessario questa condizione? Tanto se chiami
        # questo metodo interno assumo che tu sappia cosa stia facendo.
        # Alla peggio .astype restituisce data
        # cp.float32 == np.float32 da True, quindi non ci sono falsi negativi
        if data.dtype != target_dtype:
            return data.astype(target_dtype, copy=False)
        else:
            return data
        
    @staticmethod
    def _promote_to_tensor(x) -> 'Tensor':
        if type(x) is not Tensor:
            return Tensor(x)
        return x
    

    # ==== ?? FUNCTIONS ====

    def numpy(self, force: bool=True) -> np.ndarray:
        if self.device == "cpu":
            return self.data
        elif force:
            return self.clone().to("cpu").data
        else:
            raise RuntimeError(
                "the selected tensor is on gpu and force is False, so "
                "the data can't be converted to np.ndarray"
            )

    def clamp(self, min_value, max_value) -> 'Tensor':
        '''
        Returns a copy of self with values clipped between
        `min_value` and `max_value`.
        '''
        output = Tensor(self._device.clip(self.data, min_value, max_value))

        register_backward_if_needed(output, back_op.ClampBackward(self, min_value, max_value))
        return output

    def masked_fill(self, mask: 'Tensor', value) -> 'Tensor':
        '''
        Returns a `Tensor` with data of self replaced with `value`
        where mask is `True`.

        Args:
            mask (Tensor): tensor of booleans
            value (scalar): value to use where `mask` entries are `True`

        It's useful when it's necessary to replace the values in the
        positions marked by the mask, but, if you only need to put
        those to zero, it's better to do `self * mask` because it's
        faster and computationally cheaper.
        '''
        # "where True, yield x, otherwise yield y"
        output = Tensor(self._device.where(mask.data, value, self.data))

        register_backward_if_needed(output, back_op.MaskedFillBackward(self, mask.data))
        return output
    
    def __masked_fill_(self, mask: 'Tensor', value) -> 'Tensor':
        # qua mi posso permettere una in-place perche'
        # non sto modificando il tensore originale, ma
        # come lo attacco al grafo?
        if self.is_leaf and self.requires_grad:
            raise RuntimeError(
                "A leaf tensor can't be modified in-place."
            )
        # devo proibire l'operazione se:
        # - self e' una leaf che richiede il gradiente
        # - self e' usato in backward
        # - self e' coinvolto in altre operazioni (forward o backard che siano)
        #if self.grad_fn is not None and self.grad_fn.ref_count > 1:
            #raise RuntimeError(
                #"You can't modify a tensor that is used sas."
            #)
        
        # "where True, yield x, otherwise yield y"
        self.data[mask.data] = value
        
        if CONFIG["grad_enabled"] and self.requires_grad:
            if self.grad_fn is not None:
                self.grad_fn._pre_hook = back_op.MaskedFillBackward(self, mask.data)
                self.grad_fn.ref_count -=1
            else:
                self.grad_fn = back_op.MaskedFillBackward(self, mask.data)
        return self


    def expand(self, *shape) -> 'Tensor':
        '''
        Returns self with values copied in the new dim.
        
        Example:
        [1,2,3] -> broadcast_to([2,3]) -> [[1,2,3],[1,2,3]]

        Returns readonly view on the original array with the given shape.
        It is typically not contiguous. Furthermore, more than one element
        of a broadcasted array may refer to a single memory location.
        '''
        output = Tensor(self._device.broadcast_to(self.data, shape))
        # non so che pesci prendere per la backward
        if self.requires_grad:
            raise RuntimeError("Hai chiamato una funzione in beta.")
        return output
    
    def squeeze(self, dim: int) -> 'Tensor':
        output = Tensor(self.data.squeeze(dim))

        register_backward_if_needed(output, back_op.SqueezeBackward(self, dim))
        return output

    def unsqueeze(self, dim: int) -> 'Tensor':
        '''
        Insert a new dimension in the given dim.

        Example:
        [1,2] -> unsqueeze(0) -> [[1,2]] (new shape is 1x2)
        [1,2] -> unsqueeze(1) -> [[1],[2]] (new shape is 2x1)
        '''
        # NOTE: np ha squeeze, pero' "unsqueeze" e' expand_dim
        output = Tensor(self._device.expand_dims(self.data, axis=dim))

        register_backward_if_needed(output, back_op.UnsqueezeBackward(self, dim))
        return output

    def view(self, *shape) -> 'Tensor':
        '''
        Returns self with data rearranged to match the given shape.
        The desired shape has to have the same size (n of total elem)
        of self.
        It's not always possible to do a reshape without copying the data.
        When possible, the output share the exact same data as self
        (which is the definition of view).
        '''
        # NOTE: Da riadattare per fargli supportare anche le tuple come input
        output = Tensor(self.data.reshape(shape))

        register_backward_if_needed(output, back_op.ViewBackward(self))
        return output
    
    def transpose(self, dim1: int, dim2: int) -> 'Tensor':
        '''
        Note: this operation breaks the tensor's contiguity
        '''
        output = Tensor(self.data.swapaxes(dim1, dim2))
        
        register_backward_if_needed(output, back_op.TransposeBackward(self, dim1, dim2))
        return output

    def permute(self, *dims: tuple[int]) -> 'Tensor':
        '''
        Note: this operation breaks the tensor's contiguity
        Note: for the time being it only supports inputs as
        integers separated by commas (eg. .permute(1,0,2)) and
        NOT by tupe (.permute((1, 0, 2)))
        '''
        # equivalente della funzione sopra, ma con piu' di due dims alla volta
        output = Tensor(self.data.transpose(dims))

        register_backward_if_needed(output, back_op.PermuteBackward(self, dims))
        return output
    
    
    def contiguous(self) -> 'Tensor':
        '''
        Riarrangia i dati in memoria. Questa operazione NON richiede backward.
        '''
        if self.data.flags['C_CONTIGUOUS']:
            return self

        output = Tensor(self._device.ascontiguousarray(self.data, dtype=self.dtype))

        # passo all'output il gradiente del tensore precedente
        # NOTE: anche in PyTorch riciclano CloneBackward, d'altronde
        # contiguous() altro non e' che una copia. CloneBackward
        # passa semplicemente il gradiente al parent
        register_backward_if_needed(output, back_op.CloneBackward(self))
        #if self.grad_fn is not None:
            #output._grad_fn = self.grad_fn
        return output
    
    # ===== ?? FUNCTIONS =====

    def detach(self):
        '''
        Returns a new tensor detached from the computational graph.
        NOTE: data is shared between them, so changes made on one
        is reflected on the other.
        '''
        return Tensor(self.data.view(self.data.dtype), requires_grad=False)

    def clone(self):
        '''
        Returns a deep copy of self.
        '''
        output = Tensor(self.data.copy(), self.requires_grad, self.device, self.dtype)

        register_backward_if_needed(output, back_op.CloneBackward(self))
        return output
    
    def shape(self) -> tuple[int, int]:
        # (row, column)
        return self.data.shape
    
    def dim(self) -> int:
        '''n° of dim (scalar 0, vector 1, matrix 2, ecc.)'''
        return self.data.ndim
    
    def size(self, dim: int=None) -> int:
        # numero di ingressi
        # NOTE: OCCHIO! Questo e' True se dim=0!
        # if not dim:
        if dim is None:
            return self.data.size
        else:
            # NOTE: tensor.size(0) sembra dare risultati
            # strani, perlomeno con cuda
            return self._device.size(self.data, axis=dim)
    
    def backward(self, grad: 'Tensor'=None) -> None:
        '''
        Function used as the entrypoint of the backpropagation.
        It does some checks on the received gradient before starting
        it.
        '''
        # ==== Check sul gradiente ====
        if self.is_leaf:
            raise RuntimeError(
                "You called .backward() on a leaf tensor which, by definition, "
                "doesn't have a grad_fn."
            )
        
        if self.grad_fn is None:
            raise RuntimeError(
                "You probably called .backward() two times in a row, but can't "
                "be computed again because the computational graph was freed."
            )
        
        # ==== Avvio backprop ====
        # se self e' uno scalare
        if self.data.size == 1:
            # => posso calcolare il grad implicitamente
            #self.grad_fn(Tensor(1).data)
            # NOTE: mettere 1 non fa abbassare la loss nel transformer!
            start_backprop(self, Tensor(1.0, device=self.device).data)
        elif grad:
            if grad.__class__ is not Tensor:
                raise ValueError(
                    f"Gradient has to be Tensor or None, received {type(grad)} instead."
                )
            if grad.device != self.device:
                raise RuntimeError(
                    f"gradient has to be on the same device as receiving tensor ({self.device},"
                    f"got {grad.device} instead.)"
                )
            if self.data.shape != grad.data.shape:
                raise RuntimeError(
                    f"need gradient of shape {self.data.shape}, got {grad.data.shape} instead."
                )

            #self.grad_fn(grad.data)
            start_backprop(self, grad.data)
        else:
            raise RuntimeError(
                f"grad can be implicitly created only for scalar outputs because is not ambiguous."
                "Use .sum() or .mean() before calling .backward(), or give an all-ones tensor of shape {self.data.shape}."
                )

    
    # ===== MATH OPERATIONS =====

    @autocast_operation
    def matmul(self, other: 'Tensor') -> 'Tensor':
        # self: vector, other: matrix
        #print(self.dtype, other.dtype)
        output = Tensor(self.data @ other.data)
        
        register_backward_if_needed(output, back_op.MatmulBackward(self, other))
        return output
    
    @autocast_operation
    def sqrt(self):
        output = Tensor(self._device.sqrt(self.data))

        register_backward_if_needed(output, back_op.SqrtBackward(self, output.data))
        return output
    
    def exp(self):
        output = Tensor(self._device.exp(self.data))

        register_backward_if_needed(output, back_op.ExpBackward(self, output.data))
        return output
    
    # ===== REDUCTION FUNCTIONS =====
    
    # NOTE: la backward NON supporta piu' axis!
    def sum(self, dim: int=None, keepdim: bool=False) -> 'Tensor':
        """
        return: sum of data as scalar Tensor object
        """
        output = Tensor(self._device.sum(self.data, axis=dim, keepdims=keepdim))

        register_backward_if_needed(output, back_op.SumBackward(self, dim, keepdim))
        return output
    
    def mean(self, dim: int=None, keepdim: bool=False) -> 'Tensor':
        """
        Returns the mean of all dims as scalar Tensor object if dim is None,
        else only of the selected dim.

        Args:
            dim (int): 
        """
        output = Tensor(self._device.mean(self.data, axis=dim, keepdims=keepdim))

        register_backward_if_needed(output, back_op.MeanBackward(self, dim, keepdim))
        return output
    
    def var(self, dim: int=None, keepdim=False, unbiased=True):
        mean = self.data - self.data.mean(axis=dim, keepdims=True)
        n = self._device.size(self.data, dim)

        # NOTE: qua non paro il culo all'utente. Se mette cazzate ed
        # entra nell'if, puo' creare una div by zero
        if unbiased:
            n -= 1

        var = self._device.power(mean, 2).sum(axis=dim, keepdims=True) / n
        output = Tensor(var.squeeze()) if not keepdim else Tensor(var)
        
        register_backward_if_needed(output, back_op.VarBackward(self, dim, mean, n, keepdim))
        return output

    
    def std(self, dim: int=None, keepdim=False, unbiased=True):
        '''
        Returns standard deviation calculated with Bessel's correction
        (which is the -1 at denominator) if unbiased=True.

        std = sqrt((x-x.mean)^2 / (N-1))
        std = sqrt(var(x))
        '''
        # devo lasciare i keepdims interni a True e non a keepdim
        # o la funzione non va quando l'utente specifica solo la dim
        # senza tenere le dim
        numerator = self.data - self.data.mean(axis=dim, keepdims=True)
        n = self._device.size(self.data, dim)

        if unbiased:
            n -= 1

        std = self._device.sqrt( (numerator ** 2).sum(axis=dim, keepdims=True) / n )
        output = Tensor(std.squeeze()) if not keepdim else Tensor(std)
        #output = Tensor(std)
        
        register_backward_if_needed(output, back_op.StdBackward(self, dim, numerator, n*std, keepdim))
        return output

    
    # ===== IMPLICIT MATH OPERATIONS ===== 
    
    def __matmul__(self, other):
        return self.matmul(other)

    def __and__(self, other: 'Tensor') -> 'Tensor':
        # Nota: questa operazione non ha backward anche in
        # PyTorch perche' e' definita solo per gli int
        return Tensor(self.data & other.data, dtype=self.dtype)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        output = Tensor(self.data + other.data)
        
        register_backward_if_needed(output, back_op.AddBackward(self, other))
        return output
    
    def __iadd__(self, other: Any) -> 'Tensor':
        # PyTorch non permette operazioni in-place su 
        # metodo sperimentale
        if self.is_leaf:
            raise RuntimeError()
        
        other = self._promote_to_tensor(other)
        self.data += other.data
        # questo e' l'unico modo che mi viene in mente per non
        # perdere traccia di self quando ha gia' una grad_fn, ma
        # le operazioni in-place dovrebbero restituire self...
        output = Tensor(self.data)
        #print(output.data is self.data) True
        # se arriva qua, self e' un non-leaf, quindi ha sicuramente
        # requires_grad=True, percio' di default metto la backward
        #if other.requires_grad:
        register_backward_if_needed(output, back_op.AddBackward(self, other))
        return output

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        output = Tensor(self.data - other.data)   

        register_backward_if_needed(output, back_op.SubBackward(self, other))
        return output
    
    def _toBeTested_rsub__(self, other) -> 'Tensor':
        other = self._promote_to_tensor(other)
        output = Tensor(other.data - self.data)   

        register_backward_if_needed(output, back_op.SubBackward(other, self))
        return output

    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        output = Tensor(self.data * other.data)

        register_backward_if_needed(output, back_op.MulBackward(self, other))
        return output
    
    def __imul__(self, other: Any) -> 'Tensor':
        if self.is_leaf and self.requires_grad:
            raise RuntimeError()
        
        other = self._promote_to_tensor(other)
        self_clone = self.clone()
        self.data *= other.data
        # questo e' l'unico modo che mi viene in mente per non
        # perdere traccia di self quando ha gia' una grad_fn, ma
        # le operazioni in-place dovrebbero restituire self...
        output = Tensor(self.data)
        #print(output.data is self.data) True
        # se arriva qua, self e' un non-leaf, quindi ha sicuramente
        # requires_grad=True, percio' di default metto la backward
        #if other.requires_grad:
        register_backward_if_needed(output, back_op.MulBackward(self_clone, other))
        return output
    
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        output = Tensor(self.data / other.data)
            
        register_backward_if_needed(output, back_op.DivBackward(self, other))
        return output
    
    
    def __neg__(self) -> 'Tensor':
        # output = array * -1
        output = Tensor(-self.data)

        register_backward_if_needed(output, back_op.NegBackward(self))
        return output
    
    
    def __pow__(self, exponent: int) -> 'Tensor':
        output = Tensor(self._device.power(self.data, exponent))

        register_backward_if_needed(output, back_op.PowBackward(self, exponent))
        return output
    
    # ===== CLASS UTILITY =====

    def __hash__(self) -> int:
        return hash(self.data.tostring()) + hash(self._device) \
                + hash(self._requires_grad) + hash(self._is_leaf) \
                + hash(self._grad_fn) + hash(self._grad)

    def __eq__(self, other):
        # per confronti del tipo Tensor1 == Tensor2
        if isinstance(other, Tensor):
            #return self._device.allclose(self.data, other.data)
            return Tensor(self.data == other.data)
        if isinstance(other, (int, float, bool)):
            return Tensor(self.data == other)
        # posso farlo per np e cp, ma non so se lo usero' mai e per ora evito
        raise NotImplementedError(
            f"Comparison with class {other.__class__} not yet supported :("
            )
    
    def __ne__(self, other):
        # uguale a __eq__, ma con il diverso
        
        if isinstance(other, Tensor):
            return Tensor(self.data != other.data)
        if isinstance(other, (int, float, bool)):
            return Tensor(self.data != other)
        # posso farlo per np e cp, ma non so se lo usero' mai e per ora evito
        raise NotImplementedError(
            f"Comparison with class {other.__class__} not yet supported :("
            )
    
    def all(self) -> bool:
        return self.data.all()

    def __len__(self) -> int:
        return len(self.data)
    
    #def __getitem__(self, index):
        #return self.data[index]

    def __iter__(self):
        # NOTE: PyTorch restituisce dei tensor splittati
        # lungo l'asse 0 e aventi UnbindBackward come
        # grad_fn. Il mio problema e' che questa roba
        # e' chiamata anche quando non itero direttamente
        # sui tensori, ma su qualcosa che li contiene, e
        # questo spacca tante robe
        return iter(self.data)

    def __getitem__(self, index: int|slice) -> 'Tensor':
        # NOTE: se non e' presente una funzione __iter__,
        # QUESTA viene usata per iterare (e viene usata
        # quando meno te lo aspetti!)!
        output = Tensor(self.data[index])

        register_backward_if_needed(output, back_op.IndexBackward(self, index))
        return output
    
    def __setitem__(self, index: int|slice, value):
        # non si capisce bene se dovrebbe avere backward...
        # PyTorch sembra non volerla perche' e' un'operazione
        # in-place, ma alcune volte lo fa e la grad_fn
        # diventa "CopySlices", che pero' non garantisce
        # una backward funzionante (infatti fa cose strane)
        if value.__class__ == Tensor:
            self.data[index] = value.data
        else:
            self.data[index] = value

    # NOTE: metodi in fase di test, controlla spesso 
    # il loro comportamento. Se ci sono problemi, 
    # torna qui e fai dei test senza queste due property
    @property
    def __array_interface__(self):
        return self.data.__array_interface__
    
    @property
    def __cuda_array_interface__(self):
        # Restituisci l'interfaccia __cuda_array_interface__ di self.data
        return self.data.__cuda_array_interface__
    
    def __str__(self):
        # funzione chiamata quando stampo il Tensor
        print_str = self._device.array2string(
            self.data,
            threshold=self._device.inf,
            max_line_width=80,
            separator=', ',
            precision=4)

        if self.grad_fn is not None:
            print_str = print_str + f", grad_fn={self.grad_fn}"
        elif self.requires_grad:
            print_str = print_str + f", requires_grad={self.requires_grad}"

        return f"Tensor({print_str})"
    
    # sperimentale, per ora mai usata
    def __repr__(self):
        return f"Tensor({self.data.shape}, {self.requires_grad}, {self.device}, {self.dtype})"