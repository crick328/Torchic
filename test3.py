from typing import Any
import activation_functions as act_f#import sigmoid, softmax
import loss_functions as loss_f
import numpy as np


def matrix_multipliction(vector, matrix):
    assert len(vector) == len(matrix[0])
    output_vector = []

    for i in len(matrix):
        result = 0
        for j in len(vector):
            result += vector[j] * matrix[i][j]
        output_vector.append(result)

def vectors_sum(v1, v2):
    assert len(v1) == len(v2)

    for i in len(v1):
        v1[i] = v1[i] + v2[i]
    return v1
            
def update_w(w, grad, learning_rate):
    # si sottrae il gradient perché vogliamo scende
    return w - learning_rate*grad

def error_deriv(predicted, target):
    # derivata della cost function
    return 2*(predicted - target)

#def sigmoid_deriv(sop):
#    return sigmoid(sop)*(1.0-sigmoid(sop))


# Linear Layer
class LLayer():
    def __init__(self, units: int, next_layer_size: int):
        self.nodes = np.array()     # valore neuroni pre act_f
        self.w_matrix =  self._init_weight_matrix(units, next_layer_size)
        self.b_vector = self._init_bias_vector(next_layer_size)

    def _init_weight_matrix(self, units, next_layer_size):
        return np.array(np.random.uniform(low=0.0, high=1.0, size=(next_layer_size, units)))
    
    def _init_bias_vector(self, next_layer_size):
        return np.zeros(next_layer_size)
    
    def __call__(self, x):
        return vectors_sum(matrix_multipliction(x, self.w_matrix), self.b_vector)


class NNetwork():
    def __init__(self):
        self.layers = (
            LLayer(2, 4),
            act_f.sigmoid,
            LLayer(4, 6),
            act_f.sigmoid,
            LLayer(6, 3),
            act_f.sigmoid,
            LLayer(3, 2),
        )
        self.loss = loss_f.mse

    # o __call__?
    def _forward_prop(self, x):
        for i in range(0, len(self.layers)-2, 2):
            self.layers[i].nodes = self.layers[i](x)  # f
            x = self.layers[i+1](self.layers[i].nodes)  # h
        self.layers[-1].nodes = self.layers[i](x)   # l'ultimo layer non ha act_f dopo, quindi lo tratto fuori dal ciclo
        # no return perché sarebbe scomodo nel training. Leggiti il valore con una f o dall'ultimo layer
    
    def __call__(self, x):
        self._forward_prop(x)
        return self.layers[-1].nodes
    
    def _calc_loss(self, expected_values):
        return self.loss(self.layers[-1], expected_values)

