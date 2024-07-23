from torchic.tensor13 import Tensor
import torchic.nn.initializers as init
import numpy as np



def _create_circulant_matrix(kernel, input_shape):
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


# search query: Convolutional layer matrix operation
# https://ai.stackexchange.com/questions/11172/how-can-the-convolution-operation-be-implemented-as-a-matrix-multiplication

# in realta' puo' ereditare da Linear...
class Conv2d:
    # per la versione MK1 conviene togliere args, senno' e' un mattone di roba che
    # per ora non mi serve e che comunque richede molto tempo
    def __init__(self, in_channels = 1, out_channels = 1, group: int = 1, kernel_size: tuple = (3,3), stride: int = 1, padding_mode: str = "valid", bias: bool = True, _weights=None, _bias=None):
        # a quanto pare usa una Kaiming uniform init:
        # https://discuss.pytorch.org/t/what-are-the-default-initialised-matrix-values-of-the-convolution-kernels-in-pytorch/92233/3
        self.kernels = init.kaiming_uniform(kernel_size[0], kernel_size[1]) if _weights is None else _weights
        self.biases
        self.output_shape

        # codice per rendere la convoluzione una matmul

    def forward(self, x: Tensor) -> Tensor:
        # flatten deve diventare un'operazione di Tensor e deve
        # essere differenziabile
        x.data = x.data.flatten()
        '''
        x è un'immagine, quindi una matrice
        '''
        # https://www.baeldung.com/cs/convolution-matrix-multiplication
        # https://ai.stackexchange.com/questions/11172/how-can-the-convolution-operation-be-implemented-as-a-matrix-multiplication
        # https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
        pass


