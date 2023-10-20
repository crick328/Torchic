import math


_3b1b_network = {
    'L-2': {
        'type': "input layer",
        'value': [],
        'weight': 0.7
    },
    'L-1': {
        'type': "hidden layer",
        'value': [],
        'weight': -0.2
    },
    'L': {
        'type': "output layer",
        'value': [],
        'weight': 0.1
    },
}


def sigmoid(x):
    return 1/(1 + math.exp(-x))