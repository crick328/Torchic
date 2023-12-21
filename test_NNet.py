from test10_tensor import Tensor
import numpy as np
#from numpy import exp


# Bisogna aggiustare la loss function. Non può essere SE, ma così:
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
def SELoss(predicted, target):
    # per ora è solo una potenza, basta solo fare il return dell'operazione con **
    '''
    Squared Error
    Input:
    - predicted: 1 dimensional Tensor object containing the output of the network
    - target: 1 dimensional Tensor object containing the desired output values
    '''
    target = Tensor(target) if type(target) != Tensor else target
    loss = Tensor(np.power(predicted-target, 2))
    
    def _SELoss_backward(gradient):
        # non aggiorno il grad di loss perché la derivata di loss w.r.t. loss è 1
        predicted.grad_fn(2*(predicted-target) * gradient)
        # rilascio il grafo computazionale in modo tale che il garbage collector possa cancellare le variabili inutili
        predicted.grad_fn = None

    # scritto per uniformità e flessibilità, si può valutare di rimuoverlo
    if predicted.requires_grad:
        loss.requires_grad=True
        loss.grad_fn=_SELoss_backward
    return loss


class Sigmoid():
    # ho bisogno che sia una classe per usare __repr__
    '''
    Input: 1 dimensional Tensor
    Output: 1 dimensional Tensor having same shape as input
    '''
    def __call__(self, x):
        output = Tensor(1/(1 + np.exp(-x)))

        def _sigmoid_backward(chain_rule_tensor):
            # d/dσ = σ(x) * (1 - σ(x)); dL/dσ = d/dσ * prodotto derivate parziali delle operazioni precedenti
            x.grad += self(x)*(1-self(x)) * chain_rule_tensor
            # non può essere leaf, quindi chiamo direttamente la grad_fn
            x.grad_fn(x.grad)
            # non è mai leaf, quindi pulisco il gradient
            x.grad = None
            # elimino il pointer in modo che il garbage collector possa cancellare le variabili inutili
            x.grad_fn = None

        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.requires_grad = True
            output.grad_fn = _sigmoid_backward
        return output
    
    def __repr__(self):
        return "Sigmoid()"


def ReLU(tensor_layer):
    output = []
    for value in tensor_layer.data:
        output.append( value if value >= 0 else 0 )

    def _ReLU_backward(chain_rule_tensor):
        partial_derivative = []
        for value in tensor_layer.data:
            partial_derivative.append( 1 if value > 0 else 0 )

        tensor_layer.grad += partial_derivative * chain_rule_tensor
        tensor_layer.grad_fn(tensor_layer.grad)

        tensor_layer.grad = None
        tensor_layer.grad_fn = None

    output = Tensor(output)
    if tensor_layer.requires_grad:
        # da qua in avanti, requires_grad non sarà più modificabile perché
        # non è più considerato leaf tensor
        output.requires_grad = True
        output.grad_fn = _ReLU_backward
    return output


def softmax(tensor_layer):
    output = []
    denominatore = np.exp(tensor_layer).sum()

    for value in tensor_layer:
        output.append(np.exp(value) / denominatore)

    def _softmax_backward(chain_rule_tensor):
        # https://www.youtube.com/watch?v=AbLvJVwySEo
        '''
        n = np.size(output)
        tmp = np.tile(output, n)
        tensor_layer.grad += np.dot(tmp * (np.identity(n) - np.transpose(n)), chain_rule_tensor)
        '''
        output_size = output.size()
        # Construct an array by repeating A the number of times given by reps.
        tmp = np.tile(output, output_size)
        tensor_layer.grad += tmp * (np.identity(output_size) - np.transpose(output_size)) * chain_rule_tensor
        print(tensor_layer.grad)

        tensor_layer.grad_fn(tensor_layer.grad)

        tensor_layer.grad = None
        tensor_layer.grad_fn = None

    output = Tensor(output)
    if tensor_layer.requires_grad:
        # da qua in avanti, requires_grad non sarà più modificabile perché
        # non è più considerato leaf tensor
        output.requires_grad = True
        output.grad_fn = _softmax_backward
    return output
        



# Linear Layer
class LLayer():
    def __init__(self, units: int, next_layer_size: int, _w_matrix=None, _b_vector=None):
        self.w_matrix = self._init_weight_matrix(units, next_layer_size) if _w_matrix is None else _w_matrix
        self.b_vector = self._init_bias_vector(next_layer_size) if _b_vector is None else _b_vector

    def _init_weight_matrix(self, units, next_layer_size):
        return Tensor(np.random.uniform(low=-1.0, high=1.0, size=(units, next_layer_size)), requires_grad=True)
    
    def _init_bias_vector(self, next_layer_size):
        return Tensor(np.zeros(next_layer_size), requires_grad=True)
    
    def __call__(self, x):
        return x.matmul(self.w_matrix) + self.b_vector
    
    def __str__(self):
        # metodo chiamato con print()
        return f"LLayer(soooos)"
    
    def __repr__(self):
        return f"LLayer(units={self.w_matrix.shape()[0]}, next_layer_size={self.b_vector.shape()[0]}, _w_matrix={repr(self.w_matrix)}, _b_vector={repr(self.b_vector)})"


class NNetwork():
    def __init__(self, structure):
        self.structure = structure

    def forward(self, x):
        x = Tensor(x) if type(x) != Tensor else x
        for i in range(0, len(self.structure)):
            x = self.structure[i](x)
        return x
    
    def parameters(self):
        """
        returns all trainable tensors of the model
        """
        params = []
        for element in self.structure:
            if type(element) == LLayer:
                params.extend((element.w_matrix, element.b_vector))
        return params

    def __call__(self, x):
        return self.forward(x)
    

import base64
import gzip

def save_model_type1(model, filename="type1"):
    data = "{}".format(repr(model.structure)).encode()
    data = base64.b64encode(data)
    open(f"{filename}.rpl", "wb").write(gzip.compress(data)) # save data

def load_model_type1(filename):
    data = open(filename, "rb").read()
    data = gzip.decompress(data)
    data = base64.b64decode(data)
    return data.decode()