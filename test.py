# NOTA: c'è differenza tra torch.Tensor e torch.tensor:
# https://stackoverflow.com/questions/53201921/torch-tensor-with-requires-grad-parameter
# NOTA: RuntimeError: Only Tensors of floating point and complex dtype can require gradients
# https://discuss.pytorch.org/t/dont-understand-why-only-tensors-of-floating-point-dtype-can-require-gradients/158282

import torch

x = torch.tensor([0.,1.,3.], requires_grad=False)
W = torch.tensor([
        [1.,-1.],
        [5.,5.],
        [-5.,-1.]
    ], requires_grad=True)
b = torch.tensor([-8.,4.], requires_grad=True)

#1x3 e 3x2
k = x.matmul(W)
f=k+b
sigmoid=torch.nn.Sigmoid()
h=sigmoid(f)

#k.requires_grad=True => RuntimeError: you can only change requires_grad flags of leaf variables.

#k.backward(torch.tensor([1.,1.]))
k.retain_grad()
f.retain_grad()
h.backward(torch.tensor([1.,1.]))
#k.backward(torch.tensor([1.,1.]))
#print(f"Gradient vettore: {x.grad}")
print(f"Gradient matrice: {W.grad}")
print(f"Gradient bias: {b.grad}")
print(f"Gradient risultato matmul: {k.grad}")
print(f"Gradient sigmoid: {f.grad}")















def __mul__(self, other):
    Tensor = "lol"
    output = []
    if self.shape() == other.shape():
        for row in range(0, len(other)):
            tmp = []
            if type(other[0]) == list:
                for column in range(0, len(other[0])):
                    tmp.append(self.data[row][column]*other.data[row][column])
                output.append(tmp)
            else:
                output.append(self.data[row]*other.data[row])
        return Tensor(output)

    elif self.shape()[0] == other.shape()[0]:
        raise NotImplementedError("Questo prodotto ancora devo farlo...")
    elif self.shape()[1] == other.shape()[1]:
        # sto gestendo solo il caso matrice * vettore 1xN per moltiplicazione di chain rule nella matmul, ma andrebbero gestiti tutti!
        for row in self.data:
            tmp=[]
            for i in range(0, len(other.data)):
                tmp.append(row[i]*other.data[i])
            output.append(tmp)
        return Tensor(output)

    else:
        raise Exception(f"Unable to multiply shape %dx%d with shape %dx%d" % (*self.shape(), *other.shape()))
        


'''
def operation(t1, other, operator):
    output=[]
    t1_shape = t1.shape()
    other_shape = other.shape()

    if t1_shape == other_shape:
        match t1.dim():
            case 0:
                return Tensor([operator(t1, other)])
            case 1:
                for i in range(0, len(t1.data)):
                    output.append(operator(t1.data[i], other.data[i]))
                return Tensor(output)
            case 2:
                for row in range(0, len(t1.data)):
                    tmp=[]
                    for column in range(0, len(t1.data[0])):
                        tmp.append(operator(t1.data[row][column], other.data[row][column]))
                    output.append()
                return Tensor(output)
    
    # Mx1 op MxN o MxN op Mx1 ||
    elif t1_shape[0] == other_shape[0] or t1_shape[1] == other_shape[1]:
        # caso Mx1 op MxN || 1xN op MxN
        if t1_shape[1] == 1 or t1_shape[0] == 1:
            for i in range(0, len(other.data)):
                tmp=[]
                for j in range(0, len(other.data[0])):
                    tmp.append(operator(t1.data[i], other.data[i][j]))
                output.append(tmp)
            return Tensor(output)
        
        # caso MxN op Mx1 || MxN op 1xN
        elif other_shape[1] == 1 or other_shape[0] == 1:
            for i in range(0, len(t1.data)):
                tmp=[]
                for j in range(0, len(t1.data[0])):
                    tmp.append(operator(t1.data[i][j], other.data[i]))
                output.append(tmp)
            return Tensor(output)
        else:
            raise Exception(f"Cannot do {operator} between tnesors with shapes {t1_shape} and {other_shape}.")
        
    elif t1.dim() == 0:
        match other.dim():
            case 1:
                for i in range(0, len(t1.data)):
                    output.append(operator(t1.data, other.data[i]))
                return Tensor(output)
            
            case 2:
                for row in range(0, len(t1.data)):
                    tmp=[]
                    for column in range(0, len(t1.data[0])):
                        tmp.append(operator(t1.data, other.data[row][column]))
                    output.append()
                return Tensor(output)
            
    elif other.dim() == 0:
        match t1.dim():
            case 1:
                for i in range(0, len(t1.data)):
                    output.append(operator(t1.data[i], other.data))
                return Tensor(output)
            
            case 2:
                for row in range(0, len(t1.data)):
                    tmp=[]
                    for column in range(0, len(t1.data[0])):
                        tmp.append(operator(t1.data[row][column], other.data))
                    output.append()
                return Tensor(output)
    else:
        raise Exception(f"Cannot do {operator} between tnesors with shapes {t1_shape} and {other_shape}.")






if self.shape() == other.shape():
            match self.dim():
                case 0:
                    return Tensor([self-other])
                case 1:
                    for i in range(0, len(self.data)):
                        output.append(self.data[i]-other.data[i])
                    return Tensor(output)
                case 2:
                    for row in range(0, len(self.data)):
                        tmp=[]
                        for column in range(0, len(self.data[0])):
                            tmp.append(self.data[row][column]-other.data[row][column])
                        output.append()
                    return Tensor(output)
        else:
            raise NotImplementedError("per ")







def test_function(p1, p2, p3):
    """
    test_function does blah blah blah.

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
    """ 
    pass


def _to_ones(self):
        # metodo utilizzato nel calcolo delle derivate parziali
        tmp_data = []
        rows,columns = self.shape()

        for _ in range(0, rows):
            tmp_data.append([1]*columns)
        return Tensor(tmp_data)


for i in range(0, len(other)):
    result = 0
    for j in range(0, len(self.data)):
        result += self.data[i] * other[i][j]
    output.append(result)

def vector_dot_matrix(vector, matrix):
    print(len(matrix))
    assert len(vector) == len(matrix[0])
    output_vector = []
    deriv = []

    for i in range(0, len(matrix)):
        result = 0
        deriv.append([])
        for j in range(0, len(vector)):
            result += vector[j] * matrix[i][j]
            deriv[i]
        output_vector.append(result)

    def _partial_derivate():
        # calcolo le derivate e le metto 
        return _partial_derivate

    return Tensor(output_vector, op=_partial_derivate())
'''