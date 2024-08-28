from torchic.settings import set_default_device, set_default_precision, get_default_device, get_default_dtype
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import rand, randint
from torchic import nn
#from torchic.datasets.my_mnist import mnist
from torchic.tools.visualization.graphs import draw_graph
from torchic.dtypes import int8, int16, int32, int64, float16, float32, float64, complex64