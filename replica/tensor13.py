import numpy as np
#from numba import njit

# L'unica salvezza per sfruttare la gpu (e quindi parallelizzare) e CuPy, che è il
# pacchetto ufficiale di Python per CUDA.
# https://medium.com/@geminae.stellae/introduction-to-gpu-programming-with-python-cuda-577bfdaa47f3

# NON PENSARE SUBITO ALL'OTTIMIZZAZIONE. Preoccupati di scrivere qualcosa che funziona
# con le ''' puoi creare un commento visualizzabile quando usi la classe
# NOTA IMPORTANTISSIMA: gli array sono vettori algebrici "sdraiati", cioè con shape 1xN! Questo si riflette sulle matrici!



# Nota: potresti star castando a Tensor dei risultati che sono già Tensor! Non c'è coerenza!
# crea un mega commento con la struttura generale di tutte le operazioni con relativa backward, così hai un modello
# chiaro da leggere e debuggare
# Controlla se la rete non si traina con certi init per gli exploding gradients
# (Troppo presto) Torch riesce ad usare il 55-60% di CPU, tu solo la metà durante i picchi! => parallelizza con numba @jit(parallel=True)
# occhio che devi commentare molto di più! fottitene se esce lungo, spiega cosa fai e perché!
# puoi velocizzare ulteriormente il codice chiamando direttamente le variabili _

# si riesce a fare una property per cancellare subito grad se is not leaf?

# nel momento in cui si definisce un attributo come property, l'attributo nell'init deve avere un nome
# diverso da quello del suo metodo property, altrimenti crea ricorsione (e spacca slots).
# a tutti gli effetti, diventa privato.
# (!) Bisogna costruire una sorta di documentazione/spiegazione delle implementazioni!
# Crea un file apposito con funzioni per visualizzare i parametri/"modelli mentali" della rete e l'apprendimento durante il training
# teoria Transformer: https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452, ma
# prima vedi se qualche libro gli spiega (Deep learning o il pdf che hai)



class Tensor:
    __slots__ = ("data", "_requires_grad", "_is_leaf", "_grad_fn", "grad")

    def __init__(self, data, requires_grad=False):
        '''
        data: array-like
        requires_grad: flag per richiedere l'aggiornamento del gradient.

        self.data: ndarray che funge da scalare, vettore o matrice.
        self.is_leaf: True per convenzione se requires_grad è False: Altrimenti, è True se il tensor non è frutto di operazioni.
            Lo aggiorno a False se viene assegnata una grad_fn da una operazione. 
        self.grad: contiene il gradient della loss w.r.t self. Calcolato tramite la grad_fn se requires_grad è True.
            Viene automaticamente reimpostata a None se il tensor non è leaf.
        self.grad_fn: contiene la derivata parziale della funzione in cui è stato coinvolto il tensor. La funzione
            prende in input il chain_rule_tensor, ovvero un Tensor con il prodotto delle derivate parziali dei layer precedenti.
            La variabile è None se il tensor è leaf. Viene automaticamente reimpostata a None dopo ogni passaggio.
        '''
        self.data = np.array(data) if type(data) != np.ndarray else data
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._grad_fn = None
        self.grad = None

    # NOTE: property, getter e setter causano un "rallentamento" incredibilmente
    # trascurabile, sull'ordine dei 1*10^-5 e 1*10^-6 secondi in r/w in totale...

    # definisco le property, necessario per poter
    # definire i setter e quindi avere controllo sulla variabile
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @property
    def is_leaf(self) -> bool:
        return self._is_leaf
    
    @property
    def grad_fn(self):
        return self._grad_fn
    
    # definisco i rispettivi setter
    @requires_grad.setter
    def requires_grad(self, value: bool):
        # impedisco all'utente di cambiare la flag se il tensor fa parte della backprop
        if self._is_leaf == False:
            raise RuntimeError("you can only change requires_grad flags of leaf variables.")
        else:
            self._requires_grad = value
    
    @is_leaf.setter
    def is_leaf(self, value):
        # impedisco di cambiare la flag
        raise AttributeError ("attribute 'is_leaf' of Tensor object is not writable.")

    @grad_fn.setter
    def grad_fn(self, value):
        if value is not None:
            # aggiornato a False perché si assegna una grad_fn
            self._is_leaf = False
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            self._requires_grad = True
        self._grad_fn = value


    def backward(self, gradient=None):
        '''
        gradient: array-like richiesto se self.data non è uno scalare (cioè size != 1).

        Funzione richiamata sull'ultimo tensor (solitamente loss) per far partire la backprop.
        Inizia il processo di richiamo a catena delle funzioni (grad_fn) contenenti le derivate
        parziali dei layer.
        La backprop non può essere fatta più volte di fila perché il grafo viene eliminato ad
        ogni passaggio per liberare memoria (le grad_fn vengono impostate a None).
        '''
        if self.grad_fn is not None:
            if gradient is not None:
                gradient = Tensor(gradient) if type(gradient) != Tensor else gradient
                #assert self.data.shape == gradient.data.shape, f"need gradient of shape {self.data.shape}, got {gradient.data.shape} instead."
                self.grad_fn(gradient)
            elif self.size() == 1:
                self.grad_fn(Tensor([1.]))
            else:
                raise RuntimeError(f"grad can be implicitly created only for scalar outputs because is not ambiguous. Use .sum() or .mean() before calling .backward(), or give an all-ones tensor of shape {self.data.shape}.")
            # elimino il pointer in modo che il garbage collector possa cancellare le variabili inutili
            self.grad_fn = None
        elif self.is_leaf:
            raise RuntimeError("You called .backward() on a leaf tensor which, by definition, doesn't have a grad_fn.")
        else:
            raise RuntimeError("You called .backward() two times in a row, but can't be computed again because the computational graph was freed.")

    def matmul(self, other):
        # self: vector, other: matrix
        output = Tensor(np.matmul(self.data, other.data))

        def _matmul_backward(chain_rule_tensor):
            if self.requires_grad:
                # d/dx = A.T
                self.grad += Tensor(chain_rule_tensor @ other.data.T)

                # se i tensor hanno parent, continuo la backprop
                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    # elimino il grad perché è un leaf tensor e non mi serve
                    self.grad = None
                    self.grad_fn = None

            if other.requires_grad:
                # d/dA = x.T
                other.grad += Tensor(np.outer(self.data.T, chain_rule_tensor))

                if other.is_leaf == False:
                    #print("PULISCO")
                    other.grad_fn(other.grad)
                    # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
                    # potrei metterlo nel get di grad_fn
                    other.grad = None 
                    other.grad_fn = None      

        if self.requires_grad or other.requires_grad:
            #output.requires_grad = True
            output.grad_fn = _matmul_backward
        return output

    def sum(self):
        """
        return: sum of data as scalar Tensor object
        """
        output = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        output.grad_fn = self.grad_fn
        return output
    
    def mean(self):
        """
        return: mean of data as scalar Tensor object
        """
        output = Tensor(np.mean(self.data), requires_grad=self.requires_grad)

        def _mean_backward(gradient):
            self.grad += 1/self.data.size * gradient

            # se i tensor hanno parent, continuo la backprop
            if self.is_leaf == False:
                self.grad_fn(self.grad)
                # elimino il grad perché è un leaf tensor e non mi serve
                self.grad = None
                # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
                self.grad_fn = None

        if self.requires_grad:
            output.grad_fn = _mean_backward
        return output
    
    def shape(self):
        # (row,column)
        return self.data.shape
    
    def dim(self):
        return len(self.shape())
    
    def size(self):
        # numero di ingressi
        return self.data.size

    def exp(self):
        return Tensor(np.exp(self.data))
    
    def __neg__(self):
        return Tensor(np.negative(self.data))

    def __add__(self, other):
        # GESTIRE CASO IN CUI OTHER =/= TENSOR (vale per tutti i metodi)
        output = Tensor(self.data + other.data)

        def _add_backward(chain_rule_tensor):
            # le somme che necessitano di backprop avvengono solo tra tensori,
            # quindi non la definisco per __radd__
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
                #self.grad += Tensor([1]*len(self.data)) * chain_rule_tensor
                #self.grad += Tensor(np.ones(self.data.shape)) * chain_rule_tensor
                self.grad += chain_rule_tensor

                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    self.grad = None
                    self.grad_fn = None
                    
            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                #other.grad += Tensor(self.data.shape) * chain_rule_tensor
                other.grad += chain_rule_tensor

                if other.is_leaf == False:
                    other.grad_fn(other.grad)
                    other.grad = None
                    other.grad_fn = None
        
        if self.requires_grad or other.requires_grad:
            output.grad_fn = _add_backward
        return output
    
    def __radd__(self, other):
        '''
        Funzione richiamata per aggiornare i grad (grad+=).
        Se grad è None, restituisce self
        '''
        return Tensor(other.data + self.data) if type(other) == Tensor else Tensor(other + self.data) if other is not None else self

    def __sub__(self, other):
        output = Tensor(self.data - other.data)

        def _sub_backward(chain_rule_tensor):
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x-a)=1 ==> d/dx(x-y)=1-0, cioè self con i valori a 1.
                #self.grad += Tensor(np.ones(self.data.shape)) * chain_rule_tensor
                self.grad += chain_rule_tensor

                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    self.grad = None
                    self.grad_fn = None

            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a-y)=-1 ==> d/dy(x-y)=0-1, cioè other con i valori a -1.
                #other.grad += Tensor(-np.ones(other.data.shape)) * chain_rule_tensor
                other.grad += -chain_rule_tensor

                if other.is_leaf == False:
                    other.grad_fn(other.grad)
                    other.grad = None
                    other.grad_fn = None

        if self.requires_grad or other.requires_grad:
            output.grad_fn=_sub_backward
        return output
        #return Tensor(self.data - other.data) if type(other) == Tensor else Tensor(self.data - other)

    def __rsub__(self, other):
        return Tensor(other.data - self.data) if type(other) == Tensor else Tensor(other - self.data)
    
    def __mul__(self, other):
        return Tensor(self.data * other.data) if type(other) == Tensor else Tensor(self.data * other)

    def __rmul__(self, other):
        return Tensor(other.data * self.data) if type(other) == Tensor else Tensor(other * self.data)
    
    def __truediv__(self, other):
        return Tensor(self.data / other.data) if type(other) == Tensor else Tensor(self.data / other)
    
    def __rtruediv__(self, other):
        return Tensor(other.data / self.data) if type(other) == Tensor else Tensor(other / self.data)
    
    def __pow__(self, number):
        output = Tensor(np.power(self.data, number))

        def _pow_backward(chain_rule_tensor):
            self.grad += Tensor(number * np.power(self.data, number-1)) * chain_rule_tensor
            
            if self.is_leaf == False:
                self.grad_fn(self.grad)
                self.grad = None
                # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
                self.grad_fn = None

        if self.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = _pow_backward
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return f"Tensor({self.data})"
    
    def __repr__(self):
        str_data = np.array2string(self.data, threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', '')
        return f"Tensor(data={str_data}, requires_grad={self.requires_grad})"