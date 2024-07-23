import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.0")
from torchic import Tensor, nn
from torch import tensor, optim
import torch
import torchic
import numpy as np


torchic.set_defalt_dtype(torchic.float64)
np.random.seed(0)


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        updated_params = []
        
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p_update = p - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(p_update)
        
        return updated_params
adam = AdamOptimizer(learning_rate=0.001)



# creo i valori di base con numpy e poi istanzio i tensori
in_vector = np.random.random([1,10])
w_matrix = np.random.random([10,10])
b_vector = np.random.random([1,10])

updated_params = adam.update([w_matrix, b_vector],
                             [0.1*np.ones([10,10]), 0.1*np.ones([1,10])])
updated_params = adam.update(updated_params,
                             [0.1*np.ones([10,10]), 0.1*np.ones([1,10])])



# PyTorch
t_input = tensor(in_vector)
t_weights = tensor(w_matrix, requires_grad=True)
t_bias = tensor(b_vector, requires_grad=True)
t_relu = torch.nn.ReLU()
t_optim = optim.Adam([t_weights, t_bias])

t_output = t_relu(t_input.matmul(t_weights) + t_bias)
t_output.mean().backward()

print("PyTorch")
print(t_bias)
t_optim.step()
t_optim.step()
print(t_bias)



# Torchic
my_input = Tensor(in_vector)
my_weights = Tensor(w_matrix, requires_grad=True)
my_bias = Tensor(b_vector, requires_grad=True)
my_relu = nn.ReLU()
my_optim = nn.optim.AdamV3([my_weights, my_bias])

my_output = my_relu(my_input.matmul(my_weights) + my_bias)
my_output.mean().backward()

print("Torchic")
print(my_bias)
my_optim.step()
my_optim.step()
print(my_bias)



#assert torch.allclose(tensor(my_bias.data), t_bias)
#assert torch.allclose(tensor(my_weights.data), t_weights)
#assert torch.allclose(tensor(my_bias.grad), t_bias.grad)
#assert torch.allclose(tensor(my_weights.grad), t_weights.grad)



# Aggiornamento dei parametri usando i gradienti


print("Parametri aggiornati:")
print(updated_params[1])