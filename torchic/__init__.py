# default settings
from torchic.settings import set_default_device, set_default_precision, get_default_dtype
# cuda settings
from torchic.settings import set_max_async_streams, set_time_threshold_ms, set_use_null, set_seed
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import tensor, rand, randint, ones, zeros, zeros_like
from torchic import nn
from torchic.autograd.grad_context import no_grad
from torchic.tools.visualization.graphs import draw_graph
#from torchic.dtypes import int8, int16, int32, int64, float16, float32, float64, complex64
from torchic.dtypes import int8, int16, int32, int64
from torchic.dtypes import float16, float32, float64
from torchic.dtypes import complex64
from torchic.dtypes import np_bool
from torchic.dtypes import uint8, uint16, uint32, uint64
from torchic.dtypes import bfloat16