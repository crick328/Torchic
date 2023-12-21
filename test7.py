from typing import Any
import numpy as np
from math import exp

# deve valere per scalari, liste e matrici.
# NON PENSARE SUBITO ALL'OTTIMIZZAZIONE. Preoccupati di scrivere qualcosa che funziona
# con le ''' puoi creare un commento visualizzabile quando usi la classe
# flag "use_numpy"?
# NOTA IMPORTANTISSIMA: gli array sono vettori algebrici "sdraiati", cioè con shape 1xN! Questo si riflette sulle matrici!


# COMMENTARE/SPIEGARE IL CODICE
# FARE BENE I METODI FATTI ALLA VELOCE SOLO PER VEDERE SE FUNZIONAVA (TUTTA LA BACKPROP, LA SUA NOMENCLATURA E LE OPERAZIONI __OP__)
# ORDINARE IL CODICE
# non va bene che certe operazioni hanno la _backward e altre no... Tanto non aggiunge peso al grafo finché non richiami esplicitamente con ._backward()
# somma gradient e metti funzione per azzerarli (da chiamare dopo aver aggiornato i param)
# controlla se sono più veloci le tue dumb_matmul o conviene far fare la matmul a numpy
# non va bene che _backward si possa cambiare dall'init
# _backward non è un nome abbastanza esplicito. Chiamala _backward_fn o _grad_fn
# crea logica per rimuovere i gradient se non servono (con del)
# rifai gli if delle backward, tipo gli if requires_grad e if is_leaf



def dumb_matmul1(matrix):
    '''
    Matrix multiplication between all-ones vector (matmul) matrix
    Anziché calcolarlo, sommo direttamente le colonne

    :param p1: MxN tensor
    :return: 1xN tensor, sum of the columns of the matrix
    '''
    output = []
    for i in range(0, len(matrix)):
        result = 0
        for j in range(0, len(matrix[0])):
            result += matrix[i][j]
        output.append(result)
    return Tensor(output)


def dumb_matmul2(vector, columns):
    '''
    Matrix multiplication between all-ones matrix and vector.
    Anzichè calcolarlo, compongo direttamente la matrice con n trasposte del vettore, dove n=columns

    :param p1: 1xN tensor
    :param p2: describe about parameter p2
    :return: columns xN
    '''
    output = []
    for i in range(0, len(vector)):
        output.append([vector[i]]*columns)
    return Tensor(output)


def sigmoid(tensor_layer):
    '''
    Input: 1 dimensional Tensor
    Output: 1 dimensional Tensor having same shape as input
    '''
    output = []
    for value in tensor_layer.data:
        output.append( 1/(1 + exp(-value)) )

    def _sigmoid_backward(chain_rule_tensor):
        if tensor_layer.requires_grad:
            tensor_layer.grad += (sigmoid(tensor_layer)*(1-sigmoid(tensor_layer)) * chain_rule_tensor)
        tensor_layer._backward(tensor_layer.grad)
    return Tensor(output, _backward=_sigmoid_backward)


def MSE(predicted, target):
    '''
    Mean Squared Error
    Input:
    - predicted: 1 dimensional Tensor object containing the output of the network
    - target: 1 dimensional Tensor object containing the desired output values
    '''
    
    (predicted-target).pow(2)
    
    def _MSE_backward():
        2*(predicted-target)



class Tensor:
    def __init__(self, data, requires_grad=False, _backward=None):
        '''
        self.data: array che funge da vettore o matrice.
        self._backward: contiene la funzione per calcolare le derivate parziali dei parent tensor. È None se il tensor è leaf.
        self.is_leaf: flag che indica se il tensor è il risultato di un calcolo. Valore non modificabile.
        self.requires_grad: flag per richiedere l'aggiornamento del gradient. Non incide sul proseguimento della backward.
            Se il tensore è leaf lascio scegliere all'utente, sennò viene impostato a True. Valore modificabile se is_leaf=True
        self.grad: contiene il gradient (derivata parziale) w.r.t self secondo l'operazione in cui è coinvolto il tensor.
        '''
        self.data = np.array(data) if type(data) != np.ndarray else data
        self._backward = _backward
        self.is_leaf = True if self._backward == None else False
        # All Tensors that have requires_grad which is False will be leaf Tensors by convention.
        self.requires_grad = requires_grad if self.is_leaf else True
        self.grad = None


    def backward(self, tensor):
        assert type(tensor) == Tensor, f"Type error: Tensor object required, {type(tensor)} object given instead."
        self._backward(tensor) #calcola il gradient dei tensori che hanno creato questo tensore


    def matmul(self, other):
        output = Tensor(np.matmul(self.data, other.data))

        def _matmul_backward(chain_rule_tensor):
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x×a)=a ==> d/dx(x×y)=1×y, cioè other × self con tutte le entrate ad 1.
                # è inserita nel calcolo della chain rule per trovare il gradient w.r.t output, che è quello che serve per aggiornare i params.
                self.grad += dumb_matmul1(other.data) * chain_rule_tensor

                # se i tensor hanno parent, continuo la backprop
                if self.is_leaf == False:
                    self._backward(self.grad)
            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a×y)=a ==> d/dy(x×y)=x×1, cioè self × other con tutte le entrate ad 1.
                # è inserita nel calcolo della chain rule per trovare il gradient w.r.t output, che è quello che serve per aggiornare i params.
                other.grad += dumb_matmul2(self.data, len(other.data[0])) * chain_rule_tensor

                if other.is_leaf == False:
                    other._backward(other.grad)                
            # NOTA: × sta per matrix multpilication. Una volta trovata la derivata parziale, bisogna svolgere la matmul!
            # Ho codificato i "dumb_matmul" per ottenere i risultati di matmul per vettori/matrici aventi solo 1 senza fare i calcoli.

        # MA ALLORA NON HO BISOGNO DI backward NELL'INIT
        if self.requires_grad or other.requires_grad:
            output._backward = _matmul_backward

        return output
    
    
    def shape(self):
        # (row,column)
        return self.data.shape
    
    def dim(self):
        return len(self.shape())
    
    def size(self):
        # numero di ingressi
        return self.data.size

    def __add__(self, other):
        def _sum_backward(chain_rule_tensor):
            # le somme che necessitano di backprop avvengono solo tra tensori,
            # quindi non la definisco per __radd__
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
                self.grad += Tensor([1]*len(self.data)) * chain_rule_tensor
            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                other.grad += Tensor([1]*len(other.data)) * chain_rule_tensor

            # se i tensor hanno parent, continuo la backprop
            # per ora è indifferente chiamare self.backward() o self._backward()
            if self.is_leaf() == False:
                self._backward(self.grad)
            if other.is_leaf() == False:
                other._backward(other.grad)

        return Tensor(
            data = self.data + other.data,
            _backward = _sum_backward
            )
    
    def __radd__(self, other):
        return Tensor(other.data + self.data) if type(other) == Tensor else Tensor(other + self.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data) if type(other) == Tensor else Tensor(self.data - other)

    def __rsub__(self, other):
        return Tensor(other.data - self.data) if type(other) == Tensor else Tensor(other - self.data)
    
    def __mul__(self, other):
        return Tensor(self.data * other.data) if type(other) == Tensor else Tensor(self.data * other)

    def __rmul__(self, other):
        return Tensor(other.data * self.data) if type(other) == Tensor else Tensor(other * self.data)
    
    def __pow__(self, number):
        def _pow_backward(chain_rule_tensor):
            if self.requires_grad:
                self.grad += Tensor(number * np.power(self.data, number-1)) * chain_rule_tensor
            if self.is_leaf() == False:
                self._backward()

        return Tensor(
            data = np.power(self.data, number),
            _backward = _pow_backward
            )

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
                    # impedisco di cambiare la flag
                    raise AttributeError ("attribute 'is_leaf' of Tensor object is not writable.")
        self.__dict__[attr] = value

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"Tensor({self.data})"
    
a = Tensor([0.,1.,3.], requires_grad=False)
c = a+a
print(c.requires_grad)
