from layer import Layer

class Model:
    def __init__( self, input, architecture="ANN", debug_mode = False ):
        self.debug_mode = debug_mode  # print progress
        self.layers = [ Layer( input, 0, activ_funct="sigmoid" ) ]   # da castare in oggetto numpy prima del training. Gli oggetti numpy non sono fatti per cambiare shape, quindi ci vuole questo escamotage
        self._architecture( architecture )

    def add( self, units, activ_funct ):
        self.layers.append( Layer( units, self.layers[-1].nodes.size, activ_funct ) )

    def _forward_propagation( self, nodes_vector ):
        # W x i + b = y
        for i, layer in enumerate( self.layers, start=1 ):
            layer._compute_forward_prop_step( self.layers[i-1].nodes )
        pass

    def _loss_function( self ):
        pass

    def _back_propagation( self ):
        pass

    def _architecture( self, name ):
        assert name in {"ANN", "CNN", "RNN", "RL"}, "Neural network type not yet supported."
        '''
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
        '''

    def train(self, data, optimizer):
        pass

    def inference(self, input_vector):
        pass

    def info(self):
        '''
        print number of layers, param, ecc.
        '''
        for i in range(1, len(self.layers)):
            print(self.layers[i].weight_matrix.shape)