def mse(output_layer, expected_values):
    # Mean Squared Error (categoria: Regression Loss Function)
    loss = 0
    for i, value in output_layer:
        loss += pow(value-expected_values[i], 2)
    return loss/len(output_layer)