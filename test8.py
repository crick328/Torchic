from typing import Any
import numpy as np
from math import exp

# deve valere per scalari, liste e matrici.
# NON PENSARE SUBITO ALL'OTTIMIZZAZIONE. Preoccupati di scrivere qualcosa che funziona
# con le ''' puoi creare un commento visualizzabile quando usi la classe
# flag "use_numpy"?
# NOTA IMPORTANTISSIMA: gli array sono vettori algebrici "sdraiati", cioè con shape 1xN! Questo si riflette sulle matrici!


# somma gradient e metti funzione per azzerarli (da chiamare dopo aver aggiornato i param)
# controlla se sono più veloci le tue dumb_matmul o conviene far fare la matmul a numpy
# da leggere bene e implementare (post creazione NN base)
# i gradient sono puliti solo alla fine di tutta la backward, non va bene!
# trova un modo per impedire di cambiare grad_fn a weight e bias (non è importante, tanto non sono chiamate.
# Pensavo di renderlo sempre immutabile e di assegnare il suo valore nelle funzioni con self.__dict__["grad_fn"]=)
# 



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
        # inizializzo il gradient se necessario
        if tensor_layer.grad is None:
            tensor_layer.zero_grad()

        # QUESTO += CREA UN TENSOR CON REQUIRES_GRAD??
        tensor_layer.grad += sigmoid(tensor_layer)*(1-sigmoid(tensor_layer)) * chain_rule_tensor
        print("SIGMOID")
        tensor_layer.grad_fn(tensor_layer.grad)
        # è sempre leaf, quindi pulisco il gradient
        print("PULIZIA SIGMA")
        tensor_layer.grad = None

    output = Tensor(output)
    if tensor_layer.requires_grad:
        # da qua in avanti, requires_grad non sarà più modificabile perché
        # non è più considerato leaf tensor
        output.requires_grad = True
        output.grad_fn = _sigmoid_backward
    return output


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
    def __init__(self, data, requires_grad=False):
        '''
        self.data: ndarray che funge da scalare, vettore o matrice.
        self.requires_grad: flag per richiedere l'aggiornamento del gradient.
        self.is_leaf: True per convenzione se requires_grad è False: Altrimenti, è True se il tensor non è frutto di operazioni. 
        self.grad: contiene il gradient della loss w.r.t self. Calcolato tramite la grad_fn se requires_grad è True.
        self.grad_fn: contiene la derivata parziale della funzione in cui è stato coinvolto il tensor. La funzione
            prende in input il chain_rule_tensor, ovvero un Tensor con il prodotto delle derivate parziali dei layer precedenti.
            La funzione è None se il tensor è leaf.
        '''
        self.data = np.array(data) if type(data) != np.ndarray else data
        self.is_leaf = True # aggiornato a False se si dà una grad_fn #True if not requires_grad else True
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    def backward(self, tensor):
        assert type(tensor) == Tensor, f"Type error: Tensor object required, {type(tensor)} object given instead."
        self.grad_fn(tensor) #calcola il gradient dei tensori che hanno creato questo tensore

    def matmul(self, other):
        output = Tensor(np.matmul(self.data, other.data))

        def _matmul_backward(chain_rule_tensor):
            print("MATMUL BACKWARD")
            # NOTA: × sta per matrix multpilication. Una volta trovata la derivata parziale, bisogna svolgere la matmul!
            # Ho codificato i "dumb_matmul" per ottenere i risultati di matmul per vettori/matrici aventi solo 1 senza fare i calcoli.

            if self.requires_grad:
                # inizializzo il gradient se necessario
                if self.grad is None:
                    self.zero_grad()
                # derivata parziale w.r.t self, cioè x. d/dx(x×a)=a ==> d/dx(x×y)=1×y, cioè other × self con tutte le entrate ad 1.
                # è inserita nel calcolo della chain rule per trovare il gradient w.r.t output, che è quello che serve per aggiornare i params.
                self.grad += dumb_matmul1(other.data) * chain_rule_tensor

                # se i tensor hanno parent, continuo la backprop
                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    # elimino il grad perché è un leaf tensor e non mi serve
                    self.grad = None

            if other.requires_grad:
                # inizializzo il gradient se necessario
                if other.grad is None:
                    other.zero_grad()
                # derivata parziale w.r.t other, cioè y. d/dy(a×y)=a ==> d/dy(x×y)=x×1, cioè self × other con tutte le entrate ad 1.
                # è inserita nel calcolo della chain rule per trovare il gradient w.r.t output, che è quello che serve per aggiornare i params.
                other.grad += dumb_matmul2(self.data, len(other.data[0])) * chain_rule_tensor

                if other.is_leaf == False:
                    other.grad_fn(other.grad)
                    other.grad = None              

        if self.requires_grad or other.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.requires_grad = True
            output.grad_fn = _matmul_backward
        return output
    

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)
    
    def shape(self):
        # (row,column)
        return self.data.shape
    
    def dim(self):
        return len(self.shape())
    
    def size(self):
        # numero di ingressi
        return self.data.size

    def __add__(self, other):
        print("FORWARD!")
        output = Tensor(self.data + other.data)

        def _sum_backward(chain_rule_tensor):
            # le somme che necessitano di backprop avvengono solo tra tensori,
            # quindi non la definisco per __radd__
            if self.requires_grad:
                # inizializzo il gradient se necessario
                if self.grad is None:
                    self.zero_grad()
                # derivata parziale w.r.t self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
                self.grad += Tensor([1]*len(self.data)) * chain_rule_tensor

                if self.is_leaf == False:
                    self.grad_fn(self.grad)
                    self.grad = None
                    
            if other.requires_grad:
                # inizializzo il gradient se necessario
                if other.grad is None:
                    other.zero_grad()
                # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                other.grad += Tensor([1]*len(other.data)) * chain_rule_tensor

                if other.is_leaf == False:
                    other.grad_fn(other.grad)
                    other.grad = None
        
        if self.requires_grad or other.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.requires_grad = True
            output.grad_fn = _sum_backward
        return output
    
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
        output = Tensor(np.power(self.data, number))

        def _pow_backward(chain_rule_tensor):
            # inizializzo il gradient se necessario
            if self.grad is None:
                self.zero_grad()

            self.grad += Tensor(number * np.power(self.data, number-1)) * chain_rule_tensor
            
            if self.is_leaf() == False:
                self.grad_fn(self.grad)
                self.grad = None

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
                    # impedisco di cambiare la flag
                    raise AttributeError ("attribute 'is_leaf' of Tensor object is not writable.")
                case "grad_fn":
                    # se grad_fn is None e serve il grad, permetto di aggiornare il valore di grad_fn solo una volta
                    if self.grad_fn is None and self.requires_grad == True:
                        self.__dict__["is_leaf"] = False
                    else:
                        # se la backward function è già definita, impedisco di cambiarla
                        raise AttributeError ("attribute 'grad_fn' of Tensor object is not writable.")
                
        #elif attr == "grad_fn" and self.requires_grad == True:
            #self.__dict__["is_leaf"] = False

        self.__dict__[attr] = value

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, grad_fn={self.grad_fn})"


x = Tensor([0.,1.,3.], requires_grad=False) # True per testing
W = Tensor([
    [1.,-1.],
    [5.,5.],
    [-5.,-1.]
    ], requires_grad=True)
b = Tensor([-8.,4.], requires_grad=True)


k = x.matmul(W)     #[-10, 2]
f = b+k
h = sigmoid(f)

print(h)

h.backward(Tensor([1.,1.]))
#h.backward(Tensor([1.,1.]))
#print(f"Gradient vettore: {x.grad}")
print(f"Gradient matrice: {W.grad}")
print(f"Gradient bias: {b.grad}")
#print(f"Gradient k: {k.grad}") # da capire bene
#print(f"Gradient sigmoid: {f.grad}") # il gradient e i valori sono giusti, ma non applichi la chain rule!

#print(x) #Tensor([0. 1. 3.], requires_grad=False, grad_fn=None)
#print(W) #requires_grad=True, grad_fn=None
#b.grad_fn=3 # ME LO FA FARE!
#print(b) #requires_grad=True, grad_fn=None
#print(k) #requires_grad=True, grad_fn=<function Tensor.matmul.<locals>._matmul_backward at 0x000002359C4EE0E0>
