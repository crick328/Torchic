from parameters_initialization import old_weight_inizialization
from activation_functions import act_funct_manager

# neuron vanilla class
class Neuron:
    def __init__(self, activation_function='sigmoid'):
        self.bias = 0
        self.weight = old_weight_inizialization()  # weight di coda, quindi  (O)-w- ; floating point, aka float
        self.act_funct = act_funct_manager(activation_function)

    def output(self, weighted_sum):
        return self.act_funct(weighted_sum + self.bias)


# test neuron class
class SpNeuron:
    def __init__(self):
        pass

    # classe basata sulla tua comprensione e ricerca dei neuroni biologici