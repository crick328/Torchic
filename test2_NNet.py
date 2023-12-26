from test11_tensor import Tensor
import numpy as np
import init


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
    #target = Tensor(target) if type(target) != Tensor else target
    loss = Tensor(np.power(predicted.data-target, 2))
    
    def _SELoss_backward(gradient):
        # non aggiorno il grad di loss perché la derivata di loss w.r.t. loss è 1
        predicted.grad_fn(Tensor(2*(predicted.data-target) * gradient.data))
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


class ReLU():
    def __call__(self, x):
        #output = []
        #for value in tensor_layer.data:
            #output.append(max(0, value)) #value if value >= 0 else 0 )
        output = Tensor(np.maximum(0, x.data))

        def _ReLU_backward(chain_rule_tensor):
            #partial_derivative = []
            #for value in x.data:
                #partial_derivative.append( 1 if value >= 0 else 0 )
            #nums = np.where(nums > 0, 1, nums)

            #print(partial_derivative)
            #x.grad += partial_derivative * chain_rule_tensor
            # crea overflow dopo una cinquantina di epochs
            #x.grad += Tensor(np.where(x.data > 0, 1, x.data) * chain_rule_tensor.data)
            # OCCHIO: QUESTO METODO MODIFICA X!
            np.put(x.data, x.data.nonzero()[0], 1)
            x.grad += Tensor(x.data * chain_rule_tensor.data)
            x.grad_fn(x.grad)

            x.grad = None
            x.grad_fn = None

        #output = Tensor(output)
        if x.requires_grad:
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
    def __init__(self, n_input: int, n_output: int, _weights=None, _bias=None):
        '''
        n_input: number of values received
        n_output: number of values returned
        _weights: sets the value of the weight matrix. Used when loading a model and it's not intended
            for user usage.
        _bias: sets the value of the bias vector. Used when loading a model and it's not intended
            for user usage.
        '''
        #self.weights = self._init_weight_matrix(n_input, n_output) if _weights is None else _weights
        #self.bias = self._init_bias_vector(n_input, n_output) if _bias is None else _bias
        self.weights = init.xavier_norm(n_input, n_output) if _weights is None else _weights
        self.bias = init.xavier_norm(n_input, n_output, is_bias=True) if _bias is None else _bias
    
    def __call__(self, x):
        return x.matmul(self.weights) + self.bias
    
    def __str__(self):
        # metodo chiamato con print()
        return f"LLayer(weights = {self.weights}, bias = {self.bias})"
    
    def __repr__(self):
        return f"LLayer(n_input={self.weights.shape()[0]}, n_output={self.bias.shape()[0]}, _weights={repr(self.weights)}, _bias={repr(self.bias)})"

    def state_dict(self):
        return {
            self.name: {
                "a": 3
            }
        }



class NNetwork():
    def __init__(self, *args):
        for i, element in enumerate(args):
            self.__dict__[f"{element.__class__.__name__.lower()}{i}"] = element

    def forward(self, x):
        assert type(x) == Tensor, f"Expected Tensor type object, received {type(x)}"

        # itero gli args
        for key in self.__dict__:
            x = self.__dict__[key](x)
        return x
    
    def parameters(self):
        """
        returns all trainable tensors of the model
        """
        import copy

        output = []
        for key in self.__dict__:
            if type(self.__dict__[key]) == LLayer:
                output.append(copy.copy(self.__dict__[key]))
        return output

    def number_of_params(self):
        number_of_params = 0
        for key in self.__dict__:
            if type(self.__dict__[key]) == LLayer:
                number_of_params += self.__dict__[key].weights.data.size + self.__dict__[key].bias.data.size
        return number_of_params
    
    def state_dict(self):
        nnetwork_class = self.__class__.__name__
        output = {nnetwork_class: []}
        for key in self.__dict__:
            output[nnetwork_class].append(self.__dict__[key])
        return
    
    def __repr__(self):
        output = "NNetwork("
        for key in self.__dict__:
            output = output + f"{repr(self.__dict__[key])}, "
        return output[0:-2] + ")"

    def __call__(self, x):
        return self.forward(x)
    

import base64
import gzip
#import ast

def save_model_type1(model, filename="type1"):
    data = "{}".format(repr(model)).encode()
    data = base64.b64encode(data)
    open(f"{filename}.rpl", "wb").write(gzip.compress(data)) # save data

def load_model_type1(filename):
    data = open(f"{filename}.rpl", "rb").read()
    data = gzip.decompress(data)
    data = base64.b64decode(data)
    #return ast.literal_eval(data.decode())[0]#data.decode()
    return eval(data.decode())