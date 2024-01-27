#from test12_tensor_backup import Tensor
from test13_tensor import Tensor
# import poco chiaro
import init   

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
        self.weights = init.xavier_norm(n_input, n_output) if _weights is None else _weights
        self.bias = init.xavier_norm(n_input, n_output, is_bias=True) if _bias is None else _bias
    
    def __call__(self, x):
        return x.matmul(self.weights) + self.bias
    
    def __str__(self):
        # metodo chiamato con print()
        return f"LLayer(weights = {self.weights.shape()}, bias = {self.bias.shape()})"
    
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
        params = []

        for key in self.__dict__:
            if type(self.__dict__[key]) == LLayer:
                if self.__dict__[key].weights.requires_grad:
                    params.append(self.__dict__[key].weights)

                if self.__dict__[key].bias.requires_grad:
                    params.append(self.__dict__[key].bias)

        return params

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