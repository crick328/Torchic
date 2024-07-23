import numpy as np

# Define the first row of the circulant matrix
weights = np.random.random([3,3])
image = np.random.random([4,4])




def create_circulant_matrix(kernel, input_shape):
    """
    Creates a circulant matrix for a given kernel and input shape for convolution.

    Args:
        kernel (np.ndarray): The convolution kernel matrix of size (n, n).
        input_shape (tuple): The shape of the input matrix (m, m).

    Returns:
        np.ndarray: The circulant matrix of shape ((m - n + 1)^2, m^2) representing the convolution operation.
    """

    n = kernel.shape[0]  # size of the kernel (assuming square)
    m = input_shape[0]  # size of the input (assuming square)

    output_size = m - n + 1
    circulant_kernel = np.zeros(shape=(output_size ** 2, m ** 2))

    # Populate circulant_kernel
    for i in range(output_size):
        for j in range(output_size):
            row = i * output_size + j

            for k in range(n):
                for l in range(n):
                    col = (i + k) * m + (j + l)
                    circulant_kernel[row, col] = kernel[k, l]

    return circulant_kernel

# Example usage:
# Define a 3x3 kernel
kernel = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Define the input shape (4x4 input image)
input_shape = (4, 4)

# Create the circulant matrix
W_prime = create_circulant_matrix(kernel, input_shape)

print("W':\n", W_prime)