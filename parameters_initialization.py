from random import uniform

def old_weight_inizialization(low = -0.5, high = 0.5):
    '''
    vecchio metodo di inizializzazione dei weights. Vengono estratti dei valori da range piccoli, come
    [-1, 1], [-0.5, 0.5], [0, 1]
    '''

    return uniform(0,1)


def new_weight_inizialization(activation_function, n_nodes):
    '''
    inizializzazione basata sui nodi e sulla activaction function
    '''

    return uniform(0,1)