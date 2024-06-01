import numpy as np
import cupy as cp
from typing import Callable, Self
from types import ModuleType
import warnings

from torchic.settings import DEFAULT


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
        _grad_fn: contiene la derivata parziale della funzione in cui è stato coinvolto
            il tensor. La funzione prende in input il cr_grad, ovvero un Tensor con il
            prodotto delle derivate parziali dei layer precedenti.
            La variabile è `None` se il tensor è leaf. Viene automaticamente reimpostata
            a None dopo ogni passaggio.
        grad: contiene il gradient della loss w.r.t self. Calcolato tramite
            la grad_fn se requires_grad è `True`.
            Viene automaticamente reimpostata a None se il tensor non è leaf.
    '''
    __slots__ = ("data", "_requires_grad", "_device", "_is_leaf", "_grad_fn", "grad")

    def __init__(self, data, requires_grad: bool=False, device: str=None):
        '''
        Returns a `Tensor` on `device` with dtype `torch.settings.DEFAULT["dtype"]`.

        Args:
            data (ArrayLike): Tenno, skuuu!.
            requires_grad (bool, optional): flag request the update of the gradient.
                Default is False.
            device (str, optional): can be either `"cpu"` or `"cuda"`.
                Default is `torch.settings.DEFAULT["device"]`.
        '''
        self.data, self._device = self.__data_manager(data, device)
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._grad_fn = None
        self.grad = None

    # ===== INIT FUNCTIONS =====

    # usata solo nell'init (double underscore per indicarlo, come da convenzione)
    @staticmethod
    def __data_manager(data, device: str=None) -> ModuleType:
        '''
        If dtype is None, default to config dict (which is float32)
        '''

        # questo caso si verificherà per tutti i child tensor, cioe'
        # per tutti i tensori usati nella libreria
        if not device:
            match(data.__class__):
                case np.ndarray:
                    return data.astype(DEFAULT["dtype"], copy=False), np
                case cp.ndarray:
                    return data.astype( DEFAULT["dtype"], copy=False), cp
                case Self.__class__:
                    warnings.warn(
                        "ATTENZIONE! Stai castando a tensor dei dati che sono già tensor!"
                        "In questo modo perderai i valori degli altri attributi."
                        "Usa il metodo SOOOS se vuoi clonare/copiare il tensor (.copy().detach())"
                        )
                    return data.data, data._device
                case _:
                    # data type non riconosciuto e non è specificato un device
                    # (questo codice non e' runnato durante il training)
                    if DEFAULT["device"] == "cpu":
                        return np.array(data, DEFAULT["dtype"]), np
                    else:
                        return cp.array(data, DEFAULT["dtype"]), cp
        # questi casi si verificheranno solo per i tensori senza parent (fondatori)
        # di conseguenza, è codice che non viene eseguito durante i training
        else:
            match(device):
                case "cuda":
                    return cp.array(data, DEFAULT["dtype"]), cp
                case"cpu":
                    return np.array(data, DEFAULT["dtype"]), np
                case _:
                    ValueError(
                        f"Expected None, 'cuda' or 'cpu', got {device} instead."
                        )

    # ===== PROPERTIES =====

    # NOTE: property, getter e setter causano un "rallentamento" incredibilmente
    # trascurabile, sull'ordine dei 1*10^-5 e 1*10^-6 secondi in r/w in totale...
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
    
    # ===== BASE FUNCTION =====

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

        def _unsqueeze_backward(cr_grad: np.ndarray|cp.ndarray):
            self._run_backward(
                # converto cr_grad da
                # output.shape a self.data.shape
                grad=cr_grad.squeeze(dim)
            )

        if self.requires_grad:
            output.grad_fn = _unsqueeze_backward
        return output

    def view(self, *shape) -> Self:
        '''
        Returns self with data rearranged to match the given shape.
        The desired shape has to have the same size (n of total elem)
        of self.
        '''
        output = Tensor(self.data.reshape(shape))

        def _view_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            self._run_backward(
                # converto cr_grad da
                # output.shape a self.data.shape
                grad=cr_grad.reshape(self.data.shape)
            )
        
        if self.requires_grad:
            output.grad_fn = _view_backward
        return output
    
    def transpose(self, dim1: int, dim2: int) -> Self:
        output = Tensor(self.data.transpose(dim1, dim2))

        def _transpose_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # rifaccio la trasposta al contrario
            self._run_backward(
                self_grad=cr_grad.transpose(dim2, dim1)
            )
        
        if self.requires_grad:
            output.grad_fn = _transpose_backward
        return output
    
    def contiguous(self) -> Self:
        '''
        Riarrangia i dati in memoria. Questa operazione NON richiede backward.
        '''
        output = Tensor(self._device.ascontiguousarray(self.data))

        if self.grad_fn is not None:
            output.grad_fn = self.grad_fn
        return output

    def to(self, device: str) -> None:
        # il dtype e' trasferito corettamente in autonomia
        # dalle librerie
        match(device):
            case "cuda":
                if self.device != "cuda":
                    self.data = cp.array(self.data)
                    self._device = cp
            case "cpu":
                if self.device != "cpu":
                    self.data = np.array(self.data.get())
                    self._device = np
            case _:
                ValueError(
                    f"Expected 'cuda' or 'cpu', got {device} instead."
                    )
    
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
            ValueError(
                f"Gradient has to be Tensor or None, received {type(gradient)} instead."
            )

        if self.grad_fn is not None:
            # se l'utente passa un gradient
            if gradient is not None:
                if gradient.device != self.device:
                    RuntimeError(
                        f"gradient has to be on the same device as receiving tensor ({self.device},"
                        f"got {gradient.device} instead.)"
                    )
                if self.data.shape != gradient.data.shape:
                    RuntimeError(
                        f"need gradient of shape {self.data.shape}, got {gradient.data.shape} instead."
                    )
                
                self.grad_fn(gradient.data)

            elif self.data.size == 1:
                # => posso calcolarlo implicitamente
                self.grad_fn(Tensor(1., device=self.device).data)
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
    
    # lascialo cosi' il requires_grad, e' corretto, chiaro e tutto si appggia su di esso
    def _run_backwardMK1(self, grad: np.ndarray|cp.ndarray) -> None:
        # funzione chiamata solo se self.requires_grad = True
        grad = grad.astype(DEFAULT["dtype"], copy=False)

        # caso che metto prima perche' e' il piu' frequente
        if self.grad is not None:
            self.grad += grad
        else:
            # questo puo' essere inutile perche' dopo
            # lo imposto a None se non e' leaf, ma cosi'
            # mi sara' piu' facile implementare retain_grad
            self.grad = grad
        
        # se i tensor hanno parent, continuo la backprop
        if not self.is_leaf:
            # NOTE FONDAMENTALE: self.grad e' diverso da grad!
            self.grad_fn(grad)
            # DA QUI IN POI il codice viene runnato DOPO la fine di tutte
            # le call partite da grad_fn()
            # elimino il grad perché è un leaf tensor e non mi serve
            self.grad = None
            self.grad_fn = None


    def _run_backward(self, grad: np.ndarray|cp.ndarray) -> None:
        # requires_grad: attiva il sistema di tracciamento delle backward.
        # Se requires_grad=False, is_leaf puo' essere solo True.
        # Se requires_grad=True, is_leaf e' settato a False.
        # requires_grad decide solo se salvare una backward_function e se
        # calcolare il suo gradiente.
        # is_leaf decide solo se aggiornare il .grad del tensore o continuare
        # la backprop.
        grad = grad.astype(DEFAULT["dtype"], copy=False)

        if self.is_leaf:
            # aggiorno il grad e fine

            # caso che metto prima perche' e' il piu' frequente
            if self.grad is not None:
                self.grad += grad
            else:
                # questo puo' essere inutile perche' dopo
                # lo imposto a None se non e' leaf, ma cosi'
                # mi sara' piu' facile implementare retain_grad
                self.grad = grad
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

        def _matmul_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # NOTE: l'ordine in cui si moltiplica il gradient con la derivata
            # parziale e' importante, ma non ho capito perche' cambia da self a other

            # x @ A
            if self.data.ndim == 1 and other.data.ndim == 2:
                if self.requires_grad:
                    # d/dx = A.T;   dL/dx = crt @ A.T
                    self._run_backward(cr_grad @ other.data.T)
                if other.requires_grad:
                    # d/dA = x.T;   dL/dA = x.T ⊗ crt
                    other._run_backward(other._device.outer(self.data.T, cr_grad))

            # A @ B
            elif self.data.ndim == other.data.ndim == 2:
                if self.requires_grad:
                    # d/dA = (A @ B)' = (A)' @ B = B.T
                    # dL/dA = crt @ B.T
                    self._run_backward(cr_grad @ other.data.T)
                if other.requires_grad:
                    # d/dB  = (A @ B)' = A @ (B)' = A.T
                    # dL/dB  = A.T @ crt
                   other._run_backward(self.data.T @ cr_grad)
            # A @ x oppure matmul con dim >= 3
            else:
                raise NotImplementedError(
                    'Per ora hai definito la matmul backward nel caso vettore @ matrice '
                    'e matrice @ matrice, ma l\'operazione che stai facendo '
                    'non rientra in questa casistica. '
                    f'dim self={self.data.ndim}, dim other={other.data.ndim}'
                )

        if self.requires_grad or other.requires_grad:
            output.grad_fn = _matmul_backward
        return output
    
    # NOTE: la backward NON supporta piu' axis!
    def sum(self, dim=None) -> Self:
        assert dim==None, ValueError(
            "mi hai chiesto di fare una sum su una dimensione, ma "
            "la mia backpropo NON supporta queta funzione."
            )
        """
        return: sum of data as scalar Tensor object
        """
        output = Tensor(self._device.sum(self.data, axis=dim))

        def _sum_backward(cr_grad: np.ndarray|cp.ndarray):
            self._run_backward(
                grad=self._device.ones(self.data.shape) * cr_grad
            )

        if self.requires_grad:
            output.grad_fn = _sum_backward
        return output
    
    def mean(self, dim: int=None, keepdim: bool=False) -> Self:
        """
        Returns the mean of all dims as scalar Tensor object if dim is None,
        else only of the selected dim.

        Args:
            dim (int): 
        """
        output = Tensor(self._device.mean(self.data, axis=dim, keepdims=keepdim))

        def _mean_backward(cr_grad: np.ndarray|cp.ndarray):
            # NOTE: la funzione x.expand_as(y) di PyTorch equivale a
            # np.broadcast_to(np.expand_dims(x, dim), y.data.shape) perche'
            # prima devo aggiungere la dimensione e poi devo far si' che i
            # suoi valori siano uguali a quelle delle altre dims.
            # Se x ha gia' lo stesso numero di dims, si omette .epxand_dims().
            # Con la mia classe Tensor, l'equivalente di x.expand_as(y) sarebbe
            # x.unsqueeze(dim).expand(y.shape())
            if dim is not None:
                cr_grad = self._device.expand_dims(cr_grad, dim)
            cr_grad = self._device.broadcast_to(cr_grad, self.data.shape)

            self._run_backward(
                # d/dx( (x + y + ... + z) / N) = 1/N * [d/dx(x) + d/dx(y) + ... + d/dx(z)]
                # sto derivando w.r.t. x => le altre le tratto come costanti
                # = 1/N * [1 + 0 + ... + 0] = 1/N
                # dL/dx = grad * 1/N (dove N = size(self) = n ingressi. totali di self o dim)
                grad=cr_grad / self._device.size(self.data, dim)
            )

        # in questi casi c'e' un po' di ridondanza
        if self.requires_grad:
            output.grad_fn = _mean_backward
        return output
    
    def var(self, dim: int=None, keepdim=False, unbiased=True):
        x = self.data - self.data.mean(axis=dim, keepdims=True)
        n = self._device.size(self.data, dim)

        if unbiased:
            n -= 1

        var = self._device.power(x, 2).sum(axis=dim, keepdims=True) / n
        
        if not keepdim:
            output = Tensor(var.squeeze())
        else:
            output = Tensor(var)

        def _var_backward(cr_grad: np.ndarray|cp.ndarray):
            if dim is not None:
                cr_grad = self._device.expand_dims(cr_grad, dim)
            cr_grad = self._device.broadcast_to(cr_grad, self.data.shape)

            self._run_backward(
                # d/dx( (x-x.mean)^2 / (N) )
                # = 2(x-x.mean) / N
                grad= 2*x/n * cr_grad
            )
        
        if self.requires_grad:
            output.grad_fn = _var_backward
        return output

    
    def std(self, dim: int=None, keepdim=False, unbiased=True):
        '''
        Returns standard deviation calculated with Bessel's correction
        (which is the -1 at denominator) if unbiased=True.

        std = sqrt((x-x.mean)^2 / (N-1))
        std = sqrt(var(x))
        '''
        #x = self.data - self.data.mean(axis=dim, keepdims=keepdim)
        x = self.data - self.data.mean(axis=dim, keepdims=True)
        n = self._device.size(self.data, dim)

        if unbiased:
            n -= 1

        std = self._device.sqrt( self._device.power(x, 2).sum(axis=dim, keepdims=True) / n )
        
        if not keepdim:
            output = Tensor(std.squeeze())
        else:
            output = Tensor(std)

        def _std_backward(cr_grad: np.ndarray|cp.ndarray):
            if dim is not None:
                cr_grad = self._device.expand_dims(cr_grad, dim)
            cr_grad = self._device.broadcast_to(cr_grad, self.data.shape)
            denominator = n*std

            self._run_backward(
                # d/dx( sqrt((x-x.mean)^2 / (N)) )
                # h(x) = (x-x.mean)^2 / (N); f(y) = sqrt(h(x))
                # chain rule => d/dx = df/dh * dh/dx
                # => 1/( 2*sqrt(h(x)) ) * 2(x-x.mean)/N
                # = 1/( N*sqrt(h(x)) ) * (x-x.mean)  =  (x-x.mean) / ( N*sqrt(h(x)) );
                # sqrt(h(x)) e' la formula della std, =>
                # (x - x.mean) / (N * std)

                # evito una possibile division by zero esplicando che voglio la divisione
                # solo dove i valori sono != 0
                grad= self._device.divide(x, denominator, where= denominator!=0) * cr_grad
            )
        
        if self.requires_grad:
            output.grad_fn = _std_backward
        return output
        
    
    # ===== IMPLICIT CLASS FUNCTIONS ===== 
    
    #def __matmul__(self, other):
        #return self.matmul(other)

    def __add__(self, other: Self) -> Self:
        output = Tensor(self.data + other.data)

        def _add_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # le somme che necessitano di backprop avvengono solo tra tensori,
            # quindi non la definisco per __radd__

            # forse ha piu' senso fare cr_grad.reshape()
            #cr_grad = cr_grad.reshape(self.data.shape)

            if self.requires_grad:
                self._run_backward(
                    # derivata parziale w.r.t self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
                    grad=self._device.ones(self.data.shape) * cr_grad
                )
            if other.requires_grad:
                other._run_backward(
                    # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                    grad=self._device.ones(self.data.shape) * cr_grad
                )
        
        if self.requires_grad or other.requires_grad:
            output.grad_fn = _add_backward
        return output
    
    #def __radd__(self, other) -> Self:
        #return Tensor(other.data + self.data) if type(other) == Tensor else Tensor(other + self.data) if other is not None else self

    def __sub__(self, other: Self) -> Self:
        output = Tensor(self.data - other.data)

        def _sub_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x-a)=1 ==> d/dx(x-y)=1-0, cioè self con i valori a 1.
                self._run_backward(self._device.ones(self.data.shape) * cr_grad)

            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a-y)=-1 ==> d/dy(x-y)=0-1, cioè other con i valori a -1.
                other._run_backward(-self._device.ones(other.data.shape) * cr_grad)

        if self.requires_grad or other.requires_grad:
            output.grad_fn=_sub_backward
        return output
    
    def __mul__(self, other: Self) -> Self:
        output = Tensor(self.data * other.data)

        def _mul_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            if self.requires_grad:
                # d/dx( x*y ) => d/dx(x)*y = 1*y = y
                self._run_backward(other.data * cr_grad)

            if other.requires_grad:
                # d/dy( x*y ) => d/dx(y)*x = 1*x = x
                other._run_backward(self.data * cr_grad)

        if self.requires_grad or other.requires_grad:
            output.grad_fn = _mul_backward
        return output
    
    def __truediv__(self, other: Self) -> Self:
        output = Tensor(self.data / other.data)

        def _div_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            if self.requires_grad:
                # d/dx( x/y ) = d/dx(x) * 1/y = 1 * 1/y = 1/y
                # usando la funzione evito errori div by zero
                self._run_backward(other._device.reciprocal(other.data)  *cr_grad)

            if other.requires_grad:
                # d/dy( x/y ) = d/dy(1/y)*x = -1/y^2 * x = -x/y^2
                # per il momento non gestisco div by zero
                other._run_backward(-self.data / other._device.power(other.data, 2) * cr_grad)

        if self.requires_grad or other.requires_grad:
            output.grad_fn = _div_backward
        return output

    # NOTE: QUESTE FUNZIONI SPACCANO TUTTO PERCHE' NON SUPPORTANO
    # LA BACKWARD E IL PASSAGGIO DEI GRAD!!!!
    
    # ma se deprecassi tutte queste funzioni inverse? Tanto non sono chiamate
    # mai perche' io chiedo sempre che i tensori operino tra tensori e nella
    # logica interna non chiamo le funzioni del tensor...
    #def __rsub__(self, other) -> Self:
        #return Tensor(other.data - self.data) if type(other) == Tensor else Tensor(other - self.data)

    #def __rmul__(self, other) -> Self:
        #return Tensor(other.data * self.data) if type(other) == Tensor else Tensor(other * self.data)
    
    #def __rtruediv__(self, other) -> Self:
        #return Tensor(other.data / self.data) if type(other) == Tensor else Tensor(other / self.data)
    
    def __neg__(self) -> Self:
        # output = array * -1
        output = Tensor(-self.data)

        def _neg_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # d/dx(x * -1) = d/dx(x) * -1 = ones(len(x)) * -1
            # NOTE: forse conviene fare -cr_grad.reshape
            self._run_backward(-self._device.ones(self.data.shape) * cr_grad)

        if self.requires_grad:
            output.grad_fn = _neg_backward
        return output
    
    def __pow__(self, number) -> Self:
        output = Tensor(self._device.power(self.data, number))

        def _pow_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            self._run_backward(number * self._device.power(self.data, number-1) * cr_grad)

        if self.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _pow_backward
        return output
    
    # ===== CLASS UTILITY =====

    def __hash__(self) -> int:
        return hash(self.data) + hash(self._device) \
                + hash(self._requires_grad) + hash(self._is_leaf) \
                + hash(self._grad_fn) + hash(self.grad)

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
        
        if self.grad_fn is not None:
            print_str = print_str + f", grad_fn={self.grad_fn.__name__}"
        elif self.requires_grad:
            print_str = print_str + f", requires_grad={self.requires_grad}"

        return f"Tensor({print_str})"
    
    #def __repr__(self):
        #str_data = self._device.array2string(self.data, threshold=self._device.inf, max_line_width=self._device.inf, separator=',', precision=4).replace('\n', '')
        #return f"Tensor(data={str_data}, requires_grad={self.requires_grad})"