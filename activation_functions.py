from numpy import exp


def sigmoid(x):
    for i, value in enumerate(x):
        x[i] = 1/(1 + exp(-value))
    return 1/(1 + exp(-x))

def softmax(x):
    pass