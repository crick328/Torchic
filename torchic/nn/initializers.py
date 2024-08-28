# NOTE: facendo device = DEFAULT["dtype"] uso le chiavi del dict per specificare gli args
# ai quali mi riferisco, quindi le chiavi del dict DEFAULT devono coincidere con
# i nomi usati negli args dell'init di Tensor

from torchic.cuda_tensor import Tensor
#from torchic.settings import DEFAULT
import numpy as np # gli init possono sempre creare su cpu, tanto bisogna fare model.to(device)
import warnings


def naive_init(n_input, n_output):
    return Tensor(np.random.uniform(low=0.0, high=1.0, size=(n_input, n_output)), requires_grad=True)

def naive_init_v2(n_input, n_output):
    return Tensor(np.random.uniform(low=-0.01, high=0.01, size=(n_input, n_output)), requires_grad=True)


def calculate_gain(activation_function):
    if activation_function == 'relu':
        return np.sqrt(2.0)
    elif activation_function == 'tanh':
        return 5.0 / 3  # Xavier raccomanda un gain diverso per tanh
    # Aggiungi altre funzioni di attivazione se necessario
    return 1.0  # Default gain per altre funzioni di attivazione


# Chiamata anche Glorot initialization, in quanto l'autore si chiama Xavier Glorot
# Le Xavier init sembrano più indicate per layer aventi tanh o sigmoid
# come activation function perché la loro media è 0
def xavier_uniform(n_input, n_output, is_bias=False, gain=1.0):
    limit = gain * np.sqrt(6 / (n_input + n_output))
    
    if is_bias:
        return Tensor(np.random.uniform(-limit, limit, size=n_output), requires_grad=True)
    else:
        return Tensor(np.random.uniform(-limit, limit, size=(n_input, n_output)), requires_grad=True)

def xavier_norm(n_input, n_output, is_bias=False, gain=1.0):
    # standard deviation
    std = gain * np.sqrt(2 / (n_input + n_output))

    if is_bias:
        return Tensor(np.random.normal(loc=0.0, scale=std, size=n_output), requires_grad=True)
    else:
        return Tensor(np.random.normal(loc=0.0, scale=std, size=(n_input, n_output)), requires_grad=True)

# Chiamata anche He initialization, in quanto l'autore si chiama Kaiming He
# Le Kaiming init sembrano più indicate per layer aventi la ReLU
# come activation function perché la sua media non è zero (la funzione
# "non è centrata" in 0)
def kaiming_uniform(n_input: int, n_output: int, mode: str = "fan_in", non_linearity: str = "relu", is_bias=False):
    # https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/
    # standard deviation
    warnings.warn("Il parametro 'non_linearity' è inutilizzato al momento!")
    assert mode == "fan_in" or mode == "fan_out", f"Option {mode} is not an allowed value."
    
    if mode == "fan_in":
        k = np.sqrt(6/n_input)
    else:
        k = np.sqrt(6/n_output)

    if is_bias:
        return Tensor(np.random.uniform(low=-k, high=k, size=n_output), requires_grad=True)
    else:
        return Tensor(np.random.uniform(low=-k, high=k, size=(n_input, n_output)), requires_grad=True)

def kaiming_norm(n_input: int, n_output: int, mode: str = "fan_in", non_linearity: str = "relu"):
    # https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/
    # standard deviation
    warnings.warn("Il parametro 'non_linearity' è inutilizzato al momento!")
    assert mode == "fan_in" or mode == "fan_out", f"Option {mode} is not an allowed value."

    if mode == "fan_in":
        std = np.sqrt(2/n_input)
    else:
        std = np.sqrt(2/n_output)
    return Tensor(np.random.normal(loc=0.0, scale=std, size=(n_input, n_output)), requires_grad=True)
