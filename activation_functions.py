from math import exp

def act_funct_manager(self, name):  # also known as transfer function
    if name.lower() not in {"sigmoid", "relu", "lrelu", "tanh"}:
        raise Exception("Activation function not supported.")
    
    match name:
        case "sigmoid":
            return sigmoid

        case "relu":
            # from layer import convolutionLayer
            pass

        case "lrelu":
            pass
        
        case "tanh":
            pass
    pass


def sigmoid(x):
    return 1/(1 + exp(-x))

def cazzo():
    return sigmoid

print(cazzo())