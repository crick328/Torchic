from layer import Layer

class Model:
    def __init__( self ):
        self.input_layer = Layer( units=1, prec_layer_size=0, activ_funct=None, use_bias=False )
        self.hidden_layer = Layer( units=1, prec_layer_size=1, activ_funct="sigmoid" )
        self.hidden_layer = Layer( units=1, prec_layer_size=1, activ_funct="sigmoid" )
        self.output_layer = Layer( units=1, prec_layer_size=1, activ_funct="softmax" )

    def _forward_propagation( self, input_vector ):
        # W x i + b = y
        self.input_layer.nodes[0] = input_vector[0]
        #y1 = 
        pass

    def _loss_function( self ):
        pass

    def _back_propagation( self ):
        pass

a = [1,2,3,4]
print(a[-1])