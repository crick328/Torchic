# cost function: somma delle loss dei singoli nodi di output
'''
def MSE(output_layer, expected_values, derivate=False):
    # Mean Squared Error (categoria: Regression Loss Function)
    cost = 0

    if not derivate:
        for i, value in output_layer:
            cost += pow(value-expected_values[i], 2)
        return cost/len(output_layer)
    else:
        for i, value in output_layer:
            cost += 2(value-expected_values[i])
        return cost/len(output_layer)
'''
    
def SSE(output_layer, expected_values, derivate=False):
    # Sum of Squared Error (categoria: Regression Cost Function)
    cost = 0

    if not derivate:
        for i, value in output_layer:
            cost += pow(value-expected_values[i], 2)
        return cost
    else:
        for i, value in output_layer:
            cost += 2(value-expected_values[i])
        return cost
    
def MSE(output_layer, expected_values, derivate=False):
    # Mean Squared Error (categoria: Regression Cost Function)
    return SSE(output_layer, expected_values, derivate)/len(output_layer)
