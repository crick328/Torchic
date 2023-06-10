from numpy import array, random, zeros

class Layer:
    def __init__(self, n_neurons):
        self.weights = random.uniform(low=0.0, high=1.0, size=n_neurons)
        self.bias = 0
        self.activation_function = 0

    def summation(self):
        '''
        return weighted sum of all layers' neurons
        '''
        sum = 0

        for neuron in self.layer:
            sum += neuron
        return sum
