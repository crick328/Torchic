# fonte utile per Xavier e Kaiming:
# https://www.youtube.com/watch?v=ScWTYHQra5E

from replica import CudaTensor
import numpy as np # gli init possono sempre creare su cpu, tanto bisogna fare model.to(device)
import warnings


def naive_init(n_input, n_output):
    return CudaTensor(np.random.uniform(low=0.0, high=1.0, size=(n_input, n_output)), requires_grad=True)

def naive_init_v2(n_input, n_output):
    return CudaTensor(np.random.uniform(low=-0.01, high=0.01, size=(n_input, n_output)), requires_grad=True)

# Chiamata anche Glorot initialization, in quanto l'autore si chiama Xavier Glorot
# Le Xavier init sembrano più indicate per layer aventi tanh o sigmoid
# come activation function perché la loro media è 0
def xavier_uniform(n_input, n_output, is_bias=False):
    k = np.sqrt(6/(n_input + n_output))
    if is_bias:
        return CudaTensor(np.random.uniform(low=-k, high=k, size=n_output), requires_grad=True)
    else:
        return CudaTensor(np.random.uniform(low=-k, high=k, size=(n_input, n_output)), requires_grad=True)

def xavier_norm(n_input, n_output, is_bias=False):
    # un po' di storia: https://medium.com/@shauryagoel/kaiming-he-initialization-a8d9ed0b5899
    # standard deviation
    stdv = np.sqrt(2/(n_input + n_output))
    if is_bias:
        return CudaTensor(np.random.normal(loc=0.0, scale=stdv, size=n_output), requires_grad=True)
    else:
        return CudaTensor(np.random.normal(loc=0.0, scale=stdv, size=(n_input, n_output)), requires_grad=True)

# Chiamata anche He initialization, in quanto l'autore si chiama Kaiming He
# Le Kaiming init sembrano più indicate per layer aventi la ReLU
# come activation function perché la sua media non è zero (la funzione
# "non è centrata" in 0)
def kaiming_uniform(n_input: int, n_output: int, mode: str = "fan_in", non_linearity: str = "relu"):
    # https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/
    # standard deviation
    warnings.warn("Il parametro 'non_linearity' è inutilizzato al momento!")
    assert mode == "fan_in" or mode == "fan_out", f"Option {mode} is not an allowed value."
    
    if mode == "fan_in":
        k = np.sqrt(6/n_input)
    else:
        k = np.sqrt(6/n_output)
    return CudaTensor(np.random.uniform(low=-k, high=k, size=(n_input, n_output)), requires_grad=True)

def kaiming_norm(n_input: int, n_output: int, mode: str = "fan_in", non_linearity: str = "relu"):
    # https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/
    # standard deviation
    warnings.warn("Il parametro 'non_linearity' è inutilizzato al momento!")
    assert mode == "fan_in" or mode == "fan_out", f"Option {mode} is not an allowed value."
    
    if mode == "fan_in":
        stdv = np.sqrt(2/n_input)
    else:
        stdv = np.sqrt(2/n_output)
    return CudaTensor(np.random.normal(loc=0.0, scale=stdv, size=(n_input, n_output)), requires_grad=True)
