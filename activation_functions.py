from numpy import exp

def act_funct_manager(name):  # also known as transfer function
    if name:
        assert name.lower() in {"sigmoid", "relu", "lrelu", "tanh", "softmax"}, "Activation function not supported."
    
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

            case "softmax":
                return softmax


def sigmoid(x):
    for i, value in enumerate(x):
        x[i] = 1/(1 + exp(-value))
    return 1/(1 + exp(-x))

def softmax(x):
    pass