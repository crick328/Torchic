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
# ci vorrebbe un super metodo che gestisce le oper

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
            tensor_layer.grad=(sigmoid(tensor_layer)*(1-sigmoid(tensor_layer))*chain_rule_tensor)
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
    def __init__(self, data, requires_grad=False, _backward=None,):
        '''
        self.data: array che funge da vettore o matrice.
        self._backward: contiene la funzione per calcolare le derivate parziali dei parent tensor. È None se il tensor è leaf.
        self.requires_grad: flag per richiedere l'aggiornamento del gradient. Non incide sul proseguimento della backward
        self.grad: contiene il gradient (derivata parziale) w.r.t self secondo l'operazione in cui è coinvolto il tensor.
        '''
        self.data = data
        self._backward = _backward
        self.requires_grad = requires_grad if self.is_leaf() else True   # se è leaf lascio scegliere all'utente, sennò mi serve
        self.grad = None


    def backward(self, tensor):
        assert type(tensor) == Tensor, f"Type error: Tensor object required, {type(tensor)} object given instead."
        self._backward(tensor) #calcola il gradient dei tensori che hanno creato questo tensore


    def matmul(self, other):
        # self: vettore, other: matrice/vettore
        assert self.shape()[1] == other.shape()[0], f"Shapes mismatch: %dx%d %dx%d" % (*self.shape(), *other.shape())
        output = [] # vettore o matrice, dipende

        for column in range(0, len(other[0])):
            result = 0
            for row in range(0, len(other)):
                result += self.data[row]*other.data[row][column]
            output.append(result)

        def _matmul_backward(chain_rule_tensor):
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x×a)=a ==> d/dx(x×y)=1×y, cioè other × self con tutte le entrate ad 1.
                # è inserita nel calcolo della chain rule per trovare il gradient w.r.t output, che è quello che serve per aggiornare i params.
                self.grad = dumb_matmul1(other.data)*chain_rule_tensor
            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a×y)=a ==> d/dy(x×y)=x×1, cioè self × other con tutte le entrate ad 1.
                # è inserita nel calcolo della chain rule per trovare il gradient w.r.t output, che è quello che serve per aggiornare i params.
                other.grad = dumb_matmul2(self.data, len(other.data[0]))*chain_rule_tensor
            # NOTA: × sta per matrix multpilication. Una volta trovata la derivata parziale, bisogna svolgere la matmul!
            # Ho codificato i "dumb_matmul" per ottenere i risultati di matmul per vettori/matrici aventi solo 1 senza fare i calcoli.

            # se i tensor hanno parent, continuo la backprop
            # per ora è indifferente chiamare self.backward() o self._backward()
            if self.is_leaf() == False:
                self._backward(self.grad)
            if other.is_leaf() == False:
                other._backward(other.grad)
        return Tensor(output, _backward=_matmul_backward)
    
    
    def is_leaf(self):
        # definizione di leaf tensor: tensore non ottenuto da un calolo
        #print(self._backward)
        return True if self._backward == None else False
    


    

    def dim(self):
        if type(self.data) == float or type(self.data) == int:
            return 0
        elif type(self.data[0]) == list:
            return 2
        else:
            return 1
    

    def size(self):
        return


    def shape(self):
        # (row,column)
        try:
            return(len(self.data), len(self.data[0]))
        except TypeError:
            # è un vettore o uno scalare
            return (1, len(self.data))


    def __add__(self, other):
        # per ora è l'unico caso che mi interessa è questo, solo per i vettori
        assert self.shape() == other.shape()

        output = []
        for i in range(0, len(self.data)):
            output.append(self.data[i]+other.data[i])


        def _sum_backward(chain_rule_tensor):
            # pensato solo in caso di somma di vettori
            if self.requires_grad:
                # derivata parziale w.r.t self, cioè x. d/dx(x+a)=1 ==> d/dx(x+y)=1+0, cioè self con i valori a 1.
                self.grad=Tensor([1]*len(self.data))*chain_rule_tensor #*tensor = chain rule
            if other.requires_grad:
                # derivata parziale w.r.t other, cioè y. d/dy(a+y)=1 ==> d/dy(x+y)=0+1, cioè other con i valori a 1.
                other.grad=Tensor([1]*len(other.data))*chain_rule_tensor

            # se i tensor hanno parent, continuo la backprop
            # per ora è indifferente chiamare self.backward() o self._backward()
            if self.is_leaf() == False:
                print(self.is_leaf())
                self._backward(self.grad)
            if other.is_leaf() == False:
                other._backward(other.grad)
        return Tensor(output, _backward=_sum_backward)


    def __sub__(self, other):
        # per ora mi interessa solo il caso predicted-target
        output = []
        for i in range(0, len(self.data)):
            output.append(self.data[i]-other.data[1])
        return Tensor(output)


    def __rsub__(self, other):
        # per ora mi interessa solo il caso 1-sigmoid()
        output = []
        for i in range(0, len(self.data)):
            output.append(other-self.data[i])
        return Tensor(output)
    

    def __mul__(self, other):
        output=[]
        self_shape = self.shape()
        other_shape = other.shape()

        if self_shape == other_shape:
            match self.dim():
                case 0:
                    return Tensor([self.data *other.data])
                case 1:
                    for i in range(0, len(self.data)):
                        output.append(self.data[i] * other.data[i])
                    return Tensor(output)
                case 2:
                    for row in range(0, len(self.data)):
                        tmp=[]
                        for column in range(0, len(self.data[0])):
                            tmp.append(self.data[row][column] * other.data[row][column])
                        output.append()
                    return Tensor(output)
        
        # stesso numero di RIGHE
        elif self_shape[0] == other_shape[0]:
            # caso Mx1 op MxN
            if self_shape[1] == 1:
                for row in range(0, len(other.data)):
                    tmp=[]
                    for column in range(0, len(other.data[0])):
                        tmp.append(self.data[row] * other.data[row][column])
                    output.append(tmp)
                return Tensor(output)
            
            # caso MxN op Mx1
            elif other_shape[1] == 1:
                for row in range(0, len(self.data)):
                    tmp=[]
                    for column in range(0, len(self.data[0])):
                        tmp.append(self.data[row][column] * other.data[row])
                    output.append(tmp)
                return Tensor(output)
            # caso MxK * MxN || MxN * MxK
            else:
                raise Exception(f"Cannot do multiplication between tnesors with shapes {self_shape} and {other_shape}.")
        
        # stesso numero di COLONNE
        elif self_shape[1] == other_shape[1]:
            # caso 1xN op MxN
            if self_shape[0] == 1:
                for row in range(0, len(other.data)):
                    tmp=[]
                    for column in range(0, len(other.data[0])):
                        tmp.append(self.data[column] * other.data[row][column])
                    output.append(tmp)
                return Tensor(output)
            
            # caso  MxN op 1xN
            elif other_shape[0] == 1:
                for row in range(0, len(self.data)):
                    tmp=[]
                    for column in range(0, len(self.data[0])):
                        tmp.append(self.data[row][column] * other.data[column])
                    output.append(tmp)
                return Tensor(output)
            else:
                raise Exception(f"Cannot do multiplication between tnesors with shapes {self_shape} and {other_shape}.")
        
        elif self.dim() == 0:
            match other.dim():
                case 1:
                    for i in range(0, len(self.data)):
                        output.append(self.data * other.data[i])
                    return Tensor(output)
                
                case 2:
                    for row in range(0, len(self.data)):
                        tmp=[]
                        for column in range(0, len(self.data[0])):
                            tmp.append(self.data * other.data[row][column])
                        output.append()
                    return Tensor(output)
                
        elif other.dim() == 0:
            match self.dim():
                case 1:
                    for i in range(0, len(self.data)):
                        output.append(self.data[i] * other.data)
                    return Tensor(output)
                
                case 2:
                    for row in range(0, len(self.data)):
                        tmp=[]
                        for column in range(0, len(self.data[0])):
                            tmp.append(self.data[row][column] * other.data)
                        output.append()
                    return Tensor(output)
        else:
            raise Exception(f"Cannot do multiplication between tnesors with shapes {self_shape} and {other_shape}.")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]


    def __repr__(self):
        return f"Tensor({self.data})"


'''
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


h.backward(Tensor([1.,1.]))
#print(f"Gradient vettore: {x.grad}")
print(f"Gradient matrice: {W.grad}")
print(f"Gradient bias: {b.grad}")
print(f"Gradient k: {k.grad}") # da capire bene
print(f"Gradient sigmoid: {f.grad}") # il gradient e i valori sono giusti, ma non applichi la chain rule!
# ergo, non moltiplichi i gradient!
'''
