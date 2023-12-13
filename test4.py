# https://www.kaggle.com/code/residentmario/pytorch-autograd-explained
# lo stesso ragazzo ha parlato della APi di pytorch più nel dettaglio

from typing import Any
import activation_functions as act_f
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
    v3=[]
    for i in range(0, len(v1)):
        v3.append(v1[i] + v2[i])
    return v3

            
def update_w(w, grad, learning_rate):
    # si sottrae il gradient perché vogliamo scende
    return w - learning_rate*grad


# Linear Layer
class LLayer():
    # vale la pena indagare su autograd e requires_grad
    def __init__(self, units: int, next_layer_size: int, use_bias=True):
        self.nodes = np.array()     # valore neuroni pre act_f
        self.w_matrix =  self._init_weight_matrix(units, next_layer_size)
        # IMPORTANTE. La backprop e la call devono aggiustarsi di conseguenza!
        if use_bias:
            self.b_vector = self._init_bias_vector(next_layer_size)
    
    def _expand_chain_graph(self):
        # mi basta scrivere gli indici della w_matrix
        # e predisporre la backprop
        pass

    def _init_weight_matrix(self, units, next_layer_size):
        return np.array(np.random.uniform(low=0.0, high=1.0, size=(next_layer_size, units)))
    
    def _init_bias_vector(self, next_layer_size):
        return np.zeros(next_layer_size)
    
    def __call__(self, x):
        return vectors_sum(matrix_multipliction(x, self.w_matrix), self.b_vector)


class NNetwork():
    def __init__(self):
        self.layers = (
            LLayer(2, 3),
            act_f.sigmoid,
            LLayer(3, 2, False)
        )
        self.weight_tree = []
        self.bias_tree = []
        self._build_chain_graph()

    def _build_chain_graph(self):
        # è necessario qualcosa di simile per capire quando usare le derivate parziali già calcolate
        chain_graph = {}

    def _forward_prop(self, x):
        for i in range(0, len(self.layers)-2, 2):
            self.layers[i].nodes = self.layers[i](x)  # f
            x = self.layers[i+1](self.layers[i].nodes)  # h
        self.layers[-1].nodes = self.layers[i](x)   # l'ultimo layer non ha act_f dopo, quindi lo tratto fuori dal ciclo
        # no return perché sarebbe scomodo nel training. Leggi il valore con __call__
    
    def __call__(self, x):
        self._forward_prop(x)
        return self.layers[-1].nodes
    
    def _deriv_loss(self, predicted, target):
        # si, ma dove la uso? Si usa solo per i grafici, non nella backprop
        return 2(predicted-target)
    
    def _backprop(self, target_values):
        # per ora implementala vanilla, poi si affinerà quando il quadro sarà più completo,
        # cioè quando avrai l'esperienza
        assert len(self.layers[-1]) == len(target_values), "Mismatch between len of last layer and target values vector."

        for i, out_node in enumerate(self.layers[-1]):
            dloss_i = loss_f.SE(out_node, target_values[i], derivate=True)
            chain_i = [dloss_i]

            # itero al contrario i layers (enumerate faceva casini con il value)
            for j in range(len(self.layers)-1, 0, -2):
                for k, node in enumerate(self.layers[j-2]):
                    dh = np.prod(chain_i)*self.layers[j].w_matrix[j][k]
                    chain_i.append()














