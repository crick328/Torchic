from test11_tensor import Tensor
import numpy as np


def naive_init(n_input, n_output):
    return Tensor(np.random.uniform(low=0.0, high=1.0, size=(n_input, n_output)), requires_grad=True)

def naive_init_v2(n_input, n_output):
    return Tensor(np.random.uniform(low=-0.01, high=0.01, size=(n_input, n_output)), requires_grad=True)

# Xavier init: https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
def xavier_uniform(n_input, n_output, is_bias=False):
    k = np.sqrt(1/n_input)
    if is_bias:
        return Tensor(np.random.uniform(low=-k, high=k, size=n_output), requires_grad=True)
    else:
        return Tensor(np.random.uniform(low=-k, high=k, size=(n_input, n_output)), requires_grad=True)

def xavier_norm(n_input, n_output, is_bias=False):
    # standard deviation
    stdv = np.sqrt(1/n_input)
    if is_bias:
        return Tensor(np.random.normal(loc=0.0, scale=stdv, size=n_output), requires_grad=True)
    else:
        return Tensor(np.random.normal(loc=0.0, scale=stdv, size=(n_input, n_output)), requires_grad=True)