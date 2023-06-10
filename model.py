from layer import Layer, InputLayer, OutputLayer

class Model:
    def __init__(self, input, output, debug_mode = False):
        self.debug_mode = debug_mode  # print progress
        self.input_layer = InputLayer(input)
        self.layers = []
        self.output_layer = OutputLayer(output)

    def add(self, n_neurons):
        self.layers.add( Layer(n_neurons) )

    def forward_propagation(self):
        pass

    def back_propagation(self):
        pass

    def architecture(self, name):
        if name not in {"ANN", "CNN", "RNN", "RL"}:
            raise Exception("Neural network type not supported.")
        
        match name:
            case "ANN":
                pass

            case "CNN":
                # from layer import convolutionLayer
                pass

            case "RNN":
                pass
            
            case "RL":
                pass

    def train(self, data, optimizer):
        pass