from activation_functions import sigmoid, softmax
import numpy as np

input_layer = [0]

hidden_layer1 = [0]
hidden_layer1_w = [np.random.uniform( low=0.0, high=1.0)]
hidden_layer1_b = [0]

hidden_layer2 = [0]
hidden_layer2_w = [np.random.uniform( low=0.0, high=1.0)]
hidden_layer2_b = [0]

output_layer = [0]
output_layer_w = [np.random.uniform( low=0.0, high=1.0)]
output_layer_b = [0]


def matrix_multipliction(vector, matrix):
    assert len(vector) == len(matrix[0])

    output_vector = []

    for i in len(matrix):
        result = 0

        for j in len(vector):
            result += vector[j] * matrix[i][j]

        output_vector.append(result)

def vectors_sum(v1, v2):
    assert len(v1) == len(v2)

    for i in len(v1):
        v1[i] = v1[i] + v2[i]
    return v1
            
def update_w(w, grad, learning_rate):
    # si sottrae il gradient perché vogliamo scende
    return w - learning_rate*grad

def error_deriv(predicted, target):
    # derivata della cost function
    return 2*(predicted - target)

def sigmoid_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))


# forward prop
input_layer[0] = input()

z1 = vectors_sum(matrix_multipliction(input_layer, hidden_layer1_w), hidden_layer1_b[0])
hidden_layer1 = sigmoid(z1) # f(i*w+b)

z2 = vectors_sum(matrix_multipliction(hidden_layer1, hidden_layer2_w), hidden_layer2_b[0])
hidden_layer2 = sigmoid(z2)

z3 = vectors_sum(matrix_multipliction(hidden_layer2, output_layer_w), output_layer_b[0])
output_layer = sigmoid(z3)


# cost/loss function
desired_output_value = 1

cost0 = pow(output_layer[0] - desired_output_value, 2)
#cost1 = pow(output_layer[1] - desired_output_value[1], 2) ecc.

# BACKPROP (fai calcoli su carta per estenderla a L-1)
# --- calcolo derivata (parziale) di c0 sul weight tra ultimo e penultimo layer (dC/dwo) ---

# calcolo derivata di c0 rispetto all'attivazione di output_layer
dC_da = error_deriv(output_layer[0], desired_output_value) #g1
# calcolo derivata dell'attivazione di output_layer rispetto al suo valore
da_dz3 = sigmoid_deriv(z3)  #g2
# calcolo derivata del valore di output_layer rispetto ai weights tra hl2 e output_layer
#g3
dz3_dwo = hidden_layer2

# componente 0 del gradient ( gradient[0][0] = dC_dwo)
#gradw1
dC_dwo = dz3_dwo * da_dz3 * dC_da # chain rule


# --- calcolo derivata (parziale) di c0 sul bias del penultimo layer (dC/db2) ---
dz3_db2 = 1
# componente 0 del gradient ( gradient[0][1] = dC_db2)
dC_db2 = dz3_db2 * da_dz3 * dC_da # chain rule
