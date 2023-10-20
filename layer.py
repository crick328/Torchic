import numpy as np
from activation_functions import act_funct_manager

class Layer:
    def __init__( self, units: int, prec_layer_size: int, activ_funct: str, use_bias=True ):
        self.activation_function = act_funct_manager(activ_funct)
        self.nodes = np.zeros( units )

        if use_bias:
            self.bias = self._init_bias_vector( units )

        if prec_layer_size > 0:
            self.weight_matrix = self._init_weight_matrix( units, prec_layer_size )

    def _init_bias_vector( self, units ):
        return np.zeros( units )
    
    def _init_weight_matrix( self, units, prec_layer_size ):
        '''
        args
        unit: number of "nodes" of the layer
        prec_layer_size: automatically given by the Model class. Used to create
                         a weight matrix with correct size

        logic
        Creates a matrix with shape (units x prec_layer_size) (row x columns)
        '''
        assert units > 0, "Error occured in the init of the weight matrix. Units can't be <= 0."

        # poi ci saranno altre init, quindi lascia questa funzione _init_weight_matrix()
        w_matrix = np.array( np.random.uniform( low=0.0, high=1.0, size=( units, prec_layer_size ) ) )

        return w_matrix
    
    def _compute_forward_prop_step( self, prec_layer_nodes):
        self.nodes = self.activation_function( self.weight_matrix.dot( prec_layer_nodes ) + self.bias)