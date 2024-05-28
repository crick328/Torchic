from torchic.settings import set_defalt_device, set_defalt_dtype, get_default_device, get_default_dtype
from torchic.cuda_tensor import Tensor
import torchic.tools.helper as helper
from torchic import nn
from torchic.datasets.my_mnist import mnist
from torchic import tools

from numpy import float64, float32, float16, int8, int64