import numpy as np
import cupy as cp
from typing import Callable, Self
import warnings


# crea un mega commento con la struttura generale di tutte le operazioni con relativa backward, così hai un modello
# chiaro da leggere e debuggare
# Controlla se la rete non si traina con certi init per gli exploding gradients
# puoi velocizzare ulteriormente il codice chiamando direttamente le variabili _

# si riesce a fare una property per cancellare subito grad se is not leaf?
# Trovata interessante, ma il codice perderebbe di chiarezza...

# (!) Bisogna costruire una sorta di documentazione/spiegazione delle implementazioni!
# Crea un file apposito con funzioni per visualizzare i parametri/"modelli mentali" della rete e l'apprendimento durante il training
# teoria Transformer: https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452, ma
# prima vedi se qualche libro gli spiega (Deep learning o il pdf che hai)


# NOTE: fare CudaTensor(a+b) è generalmente meglio di CudaTensor(a)+CudaTensor(b)/b
# perché ci sono meno chiamate e non ci sono if. perciò, conviene sempre la prima forma
# se si è certi del tipo di entrambi i dati

# NOTE: Pytorch usa molta più cpu e gpu!
# Forse è meglio sostituire gli assert con degli if => RaiseError

# NOTE: devi gestire il datatype di cp e np! Cosa usano default? Quale tipo
# si usa? Quale preferiscono?

class CudaTensor:
    __slots__ = ("_device", "data", "_requires_grad", "_is_leaf", "_grad_fn", "grad")

    def __init__(self, data, requires_grad: bool=False, device: str=None):
        '''
        data: array-like
        requires_grad: flag per richiedere l'aggiornamento del gradient.
        device: per gli utenti, str che può essere 'cpu' o 'cuda'. Nel codice,
            è usato anche per passare direttamente cp e np.

        self.data: ndarray che funge da scalare, vettore o matrice.
        self.is_leaf: True per convenzione se requires_grad è False: Altrimenti, è True se il tensor non è frutto di operazioni.
            Lo aggiorno a False se viene assegnata una grad_fn da una operazione. 
        self.grad: contiene il gradient della loss w.r.t self. Calcolato tramite la grad_fn se requires_grad è True.
            Viene automaticamente reimpostata a None se il tensor non è leaf.
        self.grad_fn: contiene la derivata parziale della funzione in cui è stato coinvolto il tensor. La funzione
            prende in input il chain_rule_tensor, ovvero un Tensor con il prodotto delle derivate parziali dei layer precedenti.
            La variabile è None se il tensor è leaf. Viene automaticamente reimpostata a None dopo ogni passaggio.
        '''
        #self._device = device if device.__class__ is ModuleType else (cp if device == "cuda" else np)
        #self.data = self._device.array(data) if data.__class__ is not self._device.ndarray else data
        #self.data, self._device = self.data_type(data, device) 
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._grad_fn = None
        self.grad = None
        self.data, self._device = self.data_type(data, device) 

    def data_type(self, data, device) -> tuple[np.ndarray|cp.ndarray, str]:
        # questo caso si verificherà per tutti i child tensor
        if not device:
            match(data.__class__):
                case np.ndarray:
                    return data, np
                case cp.ndarray:
                    return data, cp
                case self.__class__:
                    warnings.warn(
                        "ATTENZIONE! Stai castando a tensor dei dati che sono già tensor!"
                        "Creo un nuovo tensor con gli stessi dati di quello ricevuto."
                        )
                    self._requires_grad = data._requires_grad
                    self._is_leaf = data._is_leaf
                    self._grad_fn = data._grad_fn
                    self.grad = data.grad
                    return data.data, data._device
                case _:
                    # data type non riconosciuto e non è specificato un device
                    # di default lavoro su cpu
                    # Qua ci possono finire il gradient i dataset
                    return np.array(data), np
        # questi casi si verificheranno solo per i tensori senza parent (fondatori)
        # di conseguenza, è codice che non vede il training
        else:
            match(device):
                case "cuda":
                    return cp.array(data, dtype=cp.float32), cp
                case"cpu":
                    return np.array(data), np
                case _:
                    ValueError(
                        f"Expected None, 'cuda' or 'cpu', got {device} instead."
                        )


    # NOTE: property, getter e setter causano un "rallentamento" incredibilmente
    # trascurabile, sull'ordine dei 1*10^-5 e 1*10^-6 secondi in r/w in totale...
    # NOTE: se rendi property anche grad puoi fare stampare l'alert quando
    # cerchi di leggere il grad di un leaf tensor (che logicamente è None)
    # ed evitarti infartini nel debugging

    # definisco le property, necessario per poter
    # definire i setter e quindi avere controllo sulla variabile
    @property
    def device(self) -> str:
        if self._device is cp:
            return "cuda"
        else:
            return "cpu"

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @property
    def is_leaf(self) -> bool:
        return self._is_leaf
    
    @property
    def grad_fn(self) -> Callable:
        return self._grad_fn
    
    # definisco i rispettivi setter
    @device.setter
    def device(self, value):
        # impedisco di cambiare la flag
        raise AttributeError (
            "attribute 'device' of Tensor object is not writable."
            "Use '.to()' if you want to change the current device of this tensor."
            )

    @requires_grad.setter
    def requires_grad(self, value: bool):
        # impedisco all'utente di cambiare la flag se il tensor fa parte della backprop
        if not self._is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables.")
        else:
            self._requires_grad = value
    
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

    
    def to(self, device: str) -> None:
        match(device):
            case "cuda":
                self.data = cp.array(self.data)
                self._device = cp
            case "cpu":
                self.data = np.array(self.data)
                self._device = np
            case _:
                ValueError(
                    f"Expected 'cuda' or 'cpu', got {device} instead."
                    )


    def backward(self, gradient=None) -> None:
        '''
        gradient: array-like richiesto se self.data non è uno scalare (cioè size != 1).

        Funzione richiamata sull'ultimo tensor (solitamente loss) per far partire la backprop.
        Inizia il processo di richiamo a catena delle funzioni (grad_fn) contenenti le derivate
        parziali dei layer.
        La backprop non può essere fatta più volte di fila perché il grafo viene eliminato ad
        ogni passaggio per liberare memoria (le grad_fn vengono impostate a None) a meno che
        non si lavori con retain_graph o create_graph, che non ho implementato :) .
        '''
        if gradient is not None and gradient.__class__ is not CudaTensor:
            ValueError(
                f"Gradient has to be CudaTensor or None, received {type(gradient)} instead."
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
                
                self.grad_fn(gradient)

            elif self.data.size == 1:
                # => posso calcolarlo implicitamente
                self.grad_fn(CudaTensor(1., device=self.device))
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

    def matmul(self, other: Self) -> Self:
        # self: vector, other: matrix
        output = CudaTensor(self.data @ other.data)

        def _matmul_backward(chain_rule_tensor: Self) -> None:
            if self.requires_grad:
                # d/dx = A.T
                self.grad += CudaTensor(chain_rule_tensor.data @ other.data.T)

                # se i tensor hanno parent, continuo la backprop
                if not self.is_leaf:
                    self.grad_fn(self.grad)
                    # elimino il grad perché è un leaf tensor e non mi serve
                    self.grad = None
                    self.grad_fn = None

            if other.requires_grad:
                # d/dA = x.T
                # outer product: out[i, j] = a[i] * b[j]
                other.grad += CudaTensor(self._device.outer(self.data.T, chain_rule_tensor.data))

                if not other.is_leaf:
                    other.grad_fn(other.grad)
                    # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
                    # potrei metterlo nel get di grad_fn
                    other.grad = None 
                    other.grad_fn = None      

        if self.requires_grad or other.requires_grad:
            #output.requires_grad = True
            output.grad_fn = _matmul_backward
        return output

    # NOTE: MANCA LA FUNZIONE BACKWARD
    def sum(self) -> Self:
        """
        return: sum of data as scalar Tensor object
        """
        output = CudaTensor(self._device.sum(self.data), requires_grad=self.requires_grad)
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
        output = CudaTensor(self._device.mean(self.data), requires_grad=self.requires_grad)

        def _mean_backward(gradient: Self):
            #self.grad += 1/self.data.size * gradient
            #self.grad += 1/self.data.size * gradient * self._device.ones(self.data.shape)
            self.grad += CudaTensor(1/self.data.size* self._device.ones(self.data.shape) * gradient.data)

            # se i tensor hanno parent, continuo la backprop
            if not self.is_leaf:
                self.grad_fn(self.grad)
                # elimino il grad perché è un leaf tensor e non mi serve
                self.grad = None
                # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
                self.grad_fn = None

        if self.requires_grad:
            output.grad_fn = _mean_backward
        return output
    
    def shape(self) -> tuple[int, int]:
        # (row, column)
        return self.data.shape
    
    def dim(self) -> int:
        # n° di dimensioni (scalare 0, vettore 1, matrice 2, ecc.)
        return self.data.ndim
    
    def size(self) -> int:
        # numero di ingressi
        return self.data.size

    def exp(self) -> Self:
        return CudaTensor(self._device.exp(self.data))
    
    def __neg__(self) -> Self:
        return CudaTensor(self._device.negative(self.data))

    def __add__(self, other: Self) -> Self:
        output = CudaTensor(self.data + other.data)

        def _add_backward(chain_rule_tensor: Self) -> None:
            # le somme che necessitano di backprop avvengono solo tra tensori,
            # quindi non la definisco per __radd__
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
                #self.grad += Tensor([1]*len(self.data)) * chain_rule_tensor
                self.grad += CudaTensor(self._device.ones(self.data.shape)) * chain_rule_tensor
                #self.grad += chain_rule_tensor

                if not self.is_leaf:
                    self.grad_fn(self.grad)
                    self.grad = None
                    self.grad_fn = None
                    
            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                other.grad += CudaTensor(self._device.ones(self.data.shape)) * chain_rule_tensor
                #other.grad += chain_rule_tensor

                if not other.is_leaf:
                    other.grad_fn(other.grad)
                    other.grad = None
                    other.grad_fn = None
        
        if self.requires_grad or other.requires_grad:
            output.grad_fn = _add_backward
        return output
    
    def __radd__(self, other) -> Self:
        '''
        Funzione richiamata per aggiornare i grad (grad+=).
        Se grad è None, restituisce self
        '''
        return CudaTensor(other.data + self.data) if type(other) == CudaTensor else CudaTensor(other + self.data) if other is not None else self

    def __sub__(self, other: Self) -> Self:
        output = CudaTensor(self.data - other.data)

        def _sub_backward(chain_rule_tensor: Self) -> None:
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x-a)=1 ==> d/dx(x-y)=1-0, cioè self con i valori a 1.
                self.grad += CudaTensor(self._device.ones(self.data.shape)) * chain_rule_tensor
                #self.grad += chain_rule_tensor

                if not self.is_leaf:
                    self.grad_fn(self.grad)
                    self.grad = None
                    self.grad_fn = None

            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a-y)=-1 ==> d/dy(x-y)=0-1, cioè other con i valori a -1.
                other.grad += CudaTensor(-self._device.ones(other.data.shape)) * chain_rule_tensor
                #other.grad += -chain_rule_tensor

                if not other.is_leaf:
                    other.grad_fn(other.grad)
                    other.grad = None
                    other.grad_fn = None

        if self.requires_grad or other.requires_grad:
            output.grad_fn=_sub_backward
        return output
        #return Tensor(self.data - other.data) if type(other) == Tensor else Tensor(self.data - other)

    def __rsub__(self, other) -> Self:
        return CudaTensor(other.data - self.data) if type(other) == CudaTensor else CudaTensor(other - self.data)
    
    def __mul__(self, other) -> Self:
        return CudaTensor(self.data * other.data) if type(other) == CudaTensor else CudaTensor(self.data * other)

    def __rmul__(self, other) -> Self:
        return CudaTensor(other.data * self.data) if type(other) == CudaTensor else CudaTensor(other * self.data)
    
    def __truediv__(self, other) -> Self:
        return CudaTensor(self.data / other.data) if type(other) == CudaTensor else CudaTensor(self.data / other)
    
    def __rtruediv__(self, other) -> Self:
        return CudaTensor(other.data / self.data) if type(other) == CudaTensor else CudaTensor(other / self.data)
    
    def __pow__(self, number) -> Self:
        output = CudaTensor(self._device.power(self.data, number))

        def _pow_backward(chain_rule_tensor: Self) -> None:
            self.grad += CudaTensor(number * self._device.power(self.data, number-1)) * chain_rule_tensor
            
            if not self.is_leaf:
                self.grad_fn(self.grad)
                self.grad = None
                # elimino la reference in modo che il garbage collector possa cancellare le variabili inutili
                self.grad_fn = None

        if self.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _pow_backward
        return output

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return f"Tensor({self.data})"
    
    def __repr__(self):
        str_data = self._device.array2string(self.data, threshold=self._device.inf, max_line_width=self._device.inf, separator=',').replace('\n', '')
        return f"Tensor(data={str_data}, requires_grad={self.requires_grad})"