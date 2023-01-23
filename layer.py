from neuron import Neuron

class Layer:
    def __init__(self, n_neurons):
        self.layer = []
        self.create(n_neurons)

    def create(self, n_neurons):
        for i in range(n_neurons):
            self.layer.append(Neuron())

    def summation(self):
        '''
        return weighted sum of all layers' neurons
        '''
        sum = 0

        for neuron in self.layer:
            sum += neuron
        return sum


class InputLayer:
    def __init__(self):
        pass


class OutputLayer:
    def __init__(self):
        pass
