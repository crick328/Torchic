import numpy as np
import cupy as cp
from typing import Callable, Self
from types import ModuleType
import warnings


# Controlla se la rete non si traina con certi init per gli exploding gradients
# puoi velocizzare ulteriormente il codice chiamando direttamente le variabili _

# (!) Bisogna costruire una sorta di documentazione/spiegazione delle implementazioni!
# Crea un file apposito con funzioni per visualizzare i parametri/"modelli mentali" della rete e l'apprendimento durante il training


# NOTE: fare Tensor(a+b) è meglio di Tensor(a)+Tensor(b) (oppure b)
# perché ci sono meno chiamate e non ci sono if. perciò, conviene sempre la prima forma
# se non si necessita della backward

# NOTE: Pytorch usa molta più cpu e gpu!

# NOTE: devi gestire il datatype di cp e np! Cosa usano default? Quale tipo
# si usa? Quale preferiscono?

# NOTE: Se cambi il dtype, fallo sia in .to() che in .__device_manager()

class Tensor:
    __slots__ = ("data", "_requires_grad", "_device", "dtype", "_is_leaf", "_grad_fn", "grad")

    # dtype come parametro?
    # NOTE: CI VUOLE QUALCOSA PER GESTIRE CVORRETAMENTE DTYPE. Se lo lasci None e giri su
    # cpu diventa np.float64, ma se lo specifici
    def __init__(self, data, requires_grad: bool=False, device: str=None, dtype: np.dtype|cp.dtype=None):
        '''
        data: array-like.
        requires_grad: optional bool to request the update of the gradient.
            Default is False.
        device: optional str which can be either 'cpu' or 'cuda'.
            Default is None (automatic choice).
        '''
        '''
        self.data: ndarray che funge da scalare, vettore, matrice o elementi a più dimensioni.
        self.requires_grad: Può essere modificato solo se il tensore è leaf.
            Se si assegna una grad_fn, viene automaticamente settato a True.
        self.device: contiene il modulo np o cp. Se si legge il suo valore, grazie al
            getter si riceve "cpu" per np e "cuda" per cp.
        self.is_leaf: True per convenzione se requires_grad è False: Altrimenti,
            è True se il tensor non è frutto di operazioni.
            Automaticamente impostato a False se viene assegnata una grad_fn. 
        self.grad_fn: contiene la derivata parziale della funzione in cui è stato coinvolto il tensor. La funzione
            prende in input il cr_grad, ovvero un Tensor con il prodotto delle derivate parziali dei layer precedenti.
            La variabile è None se il tensor è leaf. Viene automaticamente reimpostata a None dopo ogni passaggio.
        self.grad: contiene il gradient della loss w.r.t self. Calcolato tramite
            la grad_fn se requires_grad è True.
            Viene automaticamente reimpostata a None se il tensor non è leaf.
        '''
        self.data = data
        self._requires_grad = requires_grad
        self._device = self.__device_manager(device, dtype)
        self.dtype = dtype
        self._is_leaf = True
        self._grad_fn = None
        self.grad = None


    # NOTE: property, getter e setter causano un "rallentamento" incredibilmente
    # trascurabile, sull'ordine dei 1*10^-5 e 1*10^-6 secondi in r/w in totale...
    # NOTE: se rendi property anche grad puoi fare stampare l'alert quando
    # cerchi di leggere il grad di un leaf tensor (che logicamente è None)
    # ed evitarti infartini nel debugging


    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @property
    def device(self) -> str:
        if self._device is cp:
            return "cuda"
        else:
            return "cpu"
    
    @property
    def is_leaf(self) -> bool:
        return self._is_leaf
    
    @property
    def grad_fn(self) -> Callable:
        return self._grad_fn
    
    # definisco i setter
    @requires_grad.setter
    def requires_grad(self, value: bool):
        # impedisco all'utente di cambiare la flag se il tensor fa parte della backprop
        if not self._is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables.")
        else:
            self._requires_grad = value

    @device.setter
    def device(self, value):
        # impedisco di cambiare la flag
        raise AttributeError (
            "attribute 'device' of Tensor object is not writable."
            "Use '.to()' if you want to change the current device of this tensor."
            )
    
    @is_leaf.setter
    def is_leaf(self, value):
        # impedisco di cambiare la flag
        raise AttributeError ("attribute 'is_leaf' of Tensor object is not writable.")

    @grad_fn.setter
    def grad_fn(self, value: bool):
        if value is not None:
            # aggiornato a False perché si assegna una grad_fn
            self._is_leaf = False
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            self._requires_grad = True
        self._grad_fn = value

    # usata solo nell'init (double underscore per indicarlo, come da convenzione)
    # deve gestire anche dtype o i child tensor l'avranno a None
    def __device_manager(self, device: str, dtype: np.dtype|cp.dtype=None) -> ModuleType:
        '''
        If necessary, casts self.data in order to put it in
        the requested device.
        If dtype is None, default for Numpy is float64
        '''

        # questo caso si verificherà per tutti i child tensor
        if not device:
            match(self.data.__class__):
                case np.ndarray:
                    return np
                case cp.ndarray:
                    return cp
                case self.__class__:
                    warnings.warn(
                        "ATTENZIONE! Stai castando a tensor dei dati che sono già tensor!"
                        "In questo modo perderai i valori degli altri attributi."
                        "Usa il metodo SOOOS se vuoi clonare/copiare il tensor (.copy().detach())"
                        )
                    self.data = self.data.data
                    return self.data._device
                case _:
                    # data type non riconosciuto e non è specificato un device
                    # => di default lavoro su cpu
                    # Tutti i tensori creati dall'utente ci passano (se device=None)
                    self.data = np.array(self.data, dtype=np.float32)
                    return np
        # questi casi si verificheranno solo per i tensori senza parent (fondatori)
        # di conseguenza, è codice che non viene eseguito durante i training
        else:
            match(device):
                case "cuda":
                    self.data = cp.array(self.data, dtype=dtype)
                    return cp
                case"cpu":
                    self.data = np.array(self.data, dtype=dtype)
                    return np
                case _:
                    ValueError(
                        f"Expected None, 'cuda' or 'cpu', got {self._device} instead."
                        )

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
    def _backward(self, self_grad: np.ndarray|cp.ndarray, other: Self=None, other_grad: np.ndarray|cp.ndarray=None) -> None:
        # se requires_grad e' False, sicuramente is_leaf e' True
        if self.requires_grad:
            # caso che metto prima perche' e' il piu' frequente
            if self.grad is not None:
                self.grad += self_grad
            else:
                # questo puo' essere inutilen perche' dopo
                # lo imposto a None se non e' leaf, ma cosi'
                # mi sara' piu' facile implementare retain_grad
                self.grad = self_grad
            
            # se i tensor hanno parent, continuo la backprop
            if not self.is_leaf:
                # NOTE FONDAMENTALE: self.grad e' diverso
                # da self_grad! Quale dei due e' corretto?
                # Io credo il secondo. Se ho ragione
                # significa che ho avuto un bug silente per
                # tutte le versioni! Non riesco a dimostrarlo pero'...
                self.grad_fn(self_grad)
                # elimino il grad perché è un leaf tensor e non mi serve
                self.grad = None
                self.grad_fn = None

        # steso codice per other. Non lo scrivo in una funzione da chiamare per
        # self e other perche' non conviene ne' in performance ne' in leggibilita'
        # e nemmeno in comodita'
        if other is not None and other.requires_grad:
            if other.grad is not None:
                other.grad += other_grad
            else:
                other.grad = other_grad
            
            if not other.is_leaf:
                other.grad_fn(other_grad)
                other.grad = None
                other.grad_fn = None

    
    # ===== BASE FUNCTION =====

    def view(self, *shape) -> Self:
        output = Tensor(self.data.reshape(shape))

        def _view_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            self._backward(
                # converto cr_grad da
                # output.shape a self.data.shape
                self_grad=cr_grad.reshape(self.data.shape)
            )
        
        if self.requires_grad:
            output.grad_fn = _view_backward
        return output
    
    def transpose(self, dim1: int, dim2: int) -> Self:
        output = Tensor(self.data.transpose(dim1, dim2))

        def _transpose_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # rifaccio la trasposta al contrario
            self._backward(
                self_grad=cr_grad.transpose(dim2, dim1)
            )
        
        if self.requires_grad:
            output.grad_fn = _transpose_backward
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
    
    def size(self) -> int:
        # numero di ingressi
        return self.data.size
    
    # ===== EXPLICIT OPERATIONS =====

    def matmul(self, other: Self) -> Self:
        # self: vector, other: matrix
        output = Tensor(self.data @ other.data)

        def _matmul_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # NOTE: l'ordine in cui si moltiplica il gradient con la derivata
            # parziale e' importante, ma non ho capito perche' cambia da self a other

            # x @ A
            if self.data.ndim == 1 and other.data.ndim == 2:
                self._backward(
                    # d/dx = A.T;   dL/dx = crt @ A.T
                    self_grad=cr_grad @ other.data.T,
                    other=other,
                    # d/dA = x.T;   dL/dA = x.T ⊗ crt
                    other_grad=other._device.outer(self.data.T, cr_grad)
                )
            # A @ B
            elif self.data.ndim == other.data.ndim == 2:
                self._backward(
                    # d/dA = (A @ B)' = (A)' @ B = B.T
                    # dL/dA = crt @ B.T
                    self_grad=cr_grad @ other.data.T,
                    other=other,
                    # d/dB  = (A @ B)' = A @ (B)' = A.T
                    # dL/dB  = A.T @ crt
                    other_grad=self.data.T @ cr_grad
                )
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
    
    # NOTE: MANCA LA FUNZIONE BACKWARD
    def sum(self) -> Self:
        """
        return: sum of data as scalar Tensor object
        """
        output = Tensor(self._device.sum(self.data))
        output.grad_fn = self.grad_fn
        warnings.warn(
            "Hai chiamato .sum(), ma NON ha una funzione backward!"
            "Se stai testando la derivata di una nuova funzione, usa .mean() o implementala."
            )
        return output
    
    # NOTE: supporta solo 1d array
    def mean(self) -> Self:
        """
        return: mean of data as scalar Tensor object
        """
        output = Tensor(self._device.mean(self.data))

        def _mean_backward(cr_grad: np.ndarray|cp.ndarray):
            self._backward(
                self_grad=1/self.data.size * self._device.ones(self.data.shape) * cr_grad
            )

        # in questi casi c'e' un po' di ridondanza
        if self.requires_grad:
            output.grad_fn = _mean_backward
        return output
    
    # ===== IMPLICIT CLASS FUNCTIONS ===== 
    
    def __matmul__(self, other):
        return self.matmul(other)

    def __add__(self, other: Self) -> Self:
        output = Tensor(self.data + other.data)

        def _add_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            # le somme che necessitano di backprop avvengono solo tra tensori,
            # quindi non la definisco per __radd__
            self._backward(
                # derivata parziale w.r.t self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
                self_grad=self._device.ones(self.data.shape) * cr_grad,
                other=other,
                # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                other_grad=self._device.ones(self.data.shape) * cr_grad
            )
        
        if self.requires_grad or other.requires_grad:
            output.grad_fn = _add_backward
        return output
    
    def __radd__(self, other) -> Self:
        return Tensor(other.data + self.data) if type(other) == Tensor else Tensor(other + self.data) if other is not None else self

    def __sub__(self, other: Self) -> Self:
        output = Tensor(self.data - other.data)

        def _sub_backward(crt: np.ndarray|cp.ndarray) -> None:
            self._backward(
                # derivata parziale w.r.t self, cioè x. d/dx(x-a)=1 ==> d/dx(x-y)=1-0, cioè self con i valori a 1.
                self_grad=self._device.ones(self.data.shape) * crt,
                other=other,
                # derivata parziale w.r.t other, cioè y. d/dy(a-y)=-1 ==> d/dy(x-y)=0-1, cioè other con i valori a -1.
                other_grad=-self._device.ones(other.data.shape) * crt
            )

        if self.requires_grad or other.requires_grad:
            output.grad_fn=_sub_backward
        return output
    
    def __rsub__(self, other) -> Self:
        return Tensor(other.data - self.data) if type(other) == Tensor else Tensor(other - self.data)
    
    def __mul__(self, other) -> Self:
        return Tensor(self.data * other.data) if type(other) == Tensor else Tensor(self.data * other)

    def __rmul__(self, other) -> Self:
        return Tensor(other.data * self.data) if type(other) == Tensor else Tensor(other * self.data)
    
    def __truediv__(self, other) -> Self:
        return Tensor(self.data / other.data) if type(other) == Tensor else Tensor(self.data / other)
    
    def __rtruediv__(self, other) -> Self:
        return Tensor(other.data / self.data) if type(other) == Tensor else Tensor(other / self.data)
    
    def __neg__(self) -> Self:
        return Tensor(self._device.negative(self.data))
    
    def __pow__(self, number) -> Self:
        output = Tensor(self._device.power(self.data, number))

        def _pow_backward(cr_grad: np.ndarray|cp.ndarray) -> None:
            self._backward(
                self_grad=number * self._device.power(self.data, number-1) * cr_grad
            )

        if self.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _pow_backward
        return output
    
    # ===== CLASS UTILITY =====

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        # funzione chiamata quando stampo il Tensor
        print_str = self._device.array2string(self.data, threshold=self._device.inf, max_line_width=self._device.inf, separator=',', precision=4).replace('\n', '')
        
        if self.requires_grad:
            print_str.join(f", requires_grad={self.requires_grad}")
        if self.grad_fn is not None:
            print_str.join(f", grad_fn={self.grad_fn}")

        return f"Tensor({print_str})"
    
    #def __repr__(self):
        #str_data = self._device.array2string(self.data, threshold=self._device.inf, max_line_width=self._device.inf, separator=',', precision=4).replace('\n', '')
        #return f"Tensor(data={str_data}, requires_grad={self.requires_grad})"