import numpy as np

class Value:
    # classe presa da Andrej Karpathy
    def __init__(self, value, require_grad=True):
        self.data = value
        self.require_grad = require_grad
        self._backward = None

        self.grad = 0 # occhio perché a 0 può dare problemi...?

    def __add__(self, value):
        if type(value) == Value:
            return Value(value=self.data+value.data)
        else:
            return Value(value=self.data+value)
    
    def __mul__(self, value):
        return Value(value=self.data*value)
    
    def __repr__(self):
        return  f"{self.data}"


class VVector:
    def __init__(self, values: list):
        self.vector = [Value(value) for value in values]
    
    def __add__(self, v):
        result = []
        if type(v) == list:
            for i, data in enumerate(self.vector):
                result.append(data+v[i])
        else:
            for data in self.vector:
                result.append(data+v)
        return VVector(result)
    
    # inutile in teoria
    def __mul__(self, v):
        result = []
        if type(v) == list or type(v) == np.ndarray:
            for i, data in enumerate(self.vector):
                result.append(data*v[i])
        else:
            for data in self.vector:
                result.append(data*v)
        return VVector(result)
    
    def __repr__(self):
        return str(self.__dict__)
    

class VMatrix:
    def __init__(self, size: tuple):
        self.matrix = self._init_matrix()

    def _init_matrix(self):
        return 
    
    def __add__(self, m):
        pass

