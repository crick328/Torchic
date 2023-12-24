import numpy as np

# NON PENSARE SUBITO ALL'OTTIMIZZAZIONE. Preoccupati di scrivere qualcosa che funziona
# con le ''' puoi creare un commento visualizzabile quando usi la classe
# NOTA IMPORTANTISSIMA: gli array sono vettori algebrici "sdraiati", cioè con shape 1xN! Questo si riflette sulle matrici!


# trova un modo per impedire di cambiare grad_fn a weight e bias (non è importante, tanto non sono chiamate). Pensavo di
# renderlo sempre immutabile e di assegnare il suo valore nelle funzioni con self.__dict__["grad_fn"]=)
# occhio perché hai un sacco di operazioni che restituiscono tensori senza grad_fn. Potrebbero interrompere tutta la backward!
# è meglio così comunque, così non ti ritrovi con un super grafo di grad_fn che nessuno chiamerà mai
# Nota: potresti star castando a Tensor dei risultati che sono già Tensor! Non c'è coerenza!
# chiami grad_fn = None anche sui leaf tensor! Metti quelle condizioni dentro all'if is_leaf == False della backward
# crea un mega commento con la struttura generale di tutte le operazioni con relativa backward, così hai un modello
# chiaro da leggere e debuggare


class Tensor:
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
        # fare il cast qua suca troppo tempo... Mi auguro che il child di due float16 sia un float16, quindi non serve qui
        self.data = np.array(data, dtype=np.float16) if type(data) != np.ndarray else np.float16(data)
        self.is_leaf = True # aggiornato a False se si assegna una grad_fn
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def backward(self, gradient=None):
        '''
        gradient: array-like richiesto se self.data non è uno scalare (cioè size!=1).

        Funzione richiamata sull'ultimo tensor (solitamente loss) per far partire la backprop.
        Inizia il processo di richiamo a catena delle funzioni (grad_fn) contenenti le derivate
        parziali dei layer.
        La backprop non può essere fatta più volte di fila perché il grafo viene eliminato ad
        ogni passaggio per liberare memoria (le grad_fn vengono impostate a None).
        '''
        if self.grad_fn is not None:
            if gradient is not None:
                gradient = Tensor(gradient) if type(gradient) != Tensor else gradient
                assert self.data.shape == gradient.data.shape, f"need gradient of shape {self.data.shape}, got {gradient.data.shape} instead."
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
                self.grad += Tensor(chain_rule_tensor @ other.data.T)

                # se i tensor hanno parent, continuo la backprop
                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    # elimino il grad perché è un leaf tensor e non mi serve
                    self.grad = None

            if other.requires_grad:
                other.grad += Tensor(np.outer(self.data.T, chain_rule_tensor))

                if other.is_leaf == False:
                    #print("PULISCO")
                    other.grad_fn(other.grad)
                    other.grad = None 

            # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
            # QUESTI NON VANNO QUI
            self.grad_fn = None
            other.grad_fn = None       

        if self.requires_grad or other.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.requires_grad = True
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
                self.grad += Tensor(np.ones(self.data.shape)) * chain_rule_tensor

                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    self.grad = None
                    
            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                other.grad += Tensor(self.data.shape) * chain_rule_tensor

                if other.is_leaf == False:
                    other.grad_fn(other.grad)
                    other.grad = None
            
            # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
            self.grad_fn = None
            other.grad_fn = None  
        
        if self.requires_grad or other.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.requires_grad = True
            output.grad_fn = _add_backward
        return output
    
    def __radd__(self, other):
        '''
        Funzione richiamata per aggiornare i grad (grad+=).
        Se grad è None, restituisce self
        '''
        return Tensor(other.data + self.data) if type(other) == Tensor else Tensor(other + self.data) if other is not None else self

    def __sub__(self, other):
        assert type(other) == Tensor, f"sub operation is not defined between Tensor and {type(other)}."
        output = Tensor(self.data - other.data)

        def _sub_backward(chain_rule_tensor):
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x-a)=1 ==> d/dx(x-y)=1-0, cioè self con i valori a 1.
                self.grad += Tensor(np.zeros(self.data.shape)) * chain_rule_tensor

                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    self.grad = None

            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a-y)=-1 ==> d/dy(x-y)=0-1, cioè other con i valori a -1.
                other.grad += Tensor([-1]*len(other.data)) * chain_rule_tensor

                if other.is_leaf == False:
                    other.grad_fn(other.grad)
                    other.grad = None
            
            # elimino i pointer in modo che il garbage collector possa cancellare le variabili inutili
            self.grad_fn = None
            other.grad_fn = None

        if self.requires_grad or other.requires_grad:
            output.requires_grad=True
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
            output.requires_grad = True
            output.grad_fn = _pow_backward
        return output

    def __len__(self):
        return len(self.data)
    
    def __setattr__(self, attr, value):
        if hasattr(self, attr):
            match attr:
                case "requires_grad":
                    # impedisco all'utente di cambiare la flag se il tensor è interno al grafo della backprop
                    if self.is_leaf == False:
                        raise RuntimeError("you can only change requires_grad flags of leaf variables.")
                case "is_leaf":
                    # se ho una grad_fn e mi stai settando is_leaf a True, te lo concedo
                    if self.grad_fn is not None and value is True:
                        # alcune volte ci sono dei leaf tensor che ricevono un aggiornamento dei valori, ma
                        # questo li trasforma in non-leaf tensor. Perciò, devo risettarli manualmente a True.
                        # Un esempio è l'aggiornamento dei params in .step()
                        # Tuttavia, deprecherò questo metodo, in quanto, in quei casi, ti consiglio operare
                        # con i .data
                        pass
                    else:
                        # impedisco di cambiare la flag
                        raise AttributeError ("attribute 'is_leaf' of Tensor object is not writable.")
                case "grad_fn":
                    # se me la setti ad un valore =/= da None, e quindi mi assegni una grad_fn, aggiorno la flag
                    if value is not None:
                        self.__dict__["is_leaf"] = False
        self.__dict__[attr] = value

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return f"Tensor({self.data})"
    
    def __repr__(self):
        str_data = np.array2string(self.data, threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', '')
        return f"Tensor(data={str_data}, requires_grad={self.requires_grad})"