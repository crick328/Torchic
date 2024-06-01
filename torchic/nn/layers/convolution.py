from torchic.tensor13 import Tensor
import torchic.nn.initializers as init
import numpy as np

# (non implementa le convolution come matmul, quindi no) https://www.youtube.com/watch?v=Lakz2MoHy6o
class Conv2d:
    def __init__(self, in_channels = 1, out_channels = 1, group: int = 1, kernel_size: tuple = (3,3), stride: int = 1, padding_mode: str = "valid", bias: bool = True, _weights=None, _bias=None):
        # a quanto pare usa una Kaiming uniform init:
        # https://discuss.pytorch.org/t/what-are-the-default-initialised-matrix-values-of-the-convolution-kernels-in-pytorch/92233/3
        self.kernels = init.kaiming_uniform(kernel_size[0], kernel_size[1]) if _weights is None else _weights
        self.biases
        self.output_shape

        # codice per rendere la convoluzione una matmul

    def forward(self, x):
        '''
        x è un'immagine, quindi una matrice
        '''
        # https://www.baeldung.com/cs/convolution-matrix-multiplication
        # https://ai.stackexchange.com/questions/11172/how-can-the-convolution-operation-be-implemented-as-a-matrix-multiplication
        # https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
        pass