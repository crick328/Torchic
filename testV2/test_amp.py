from preset import torchic
from torchic import Tensor
from torchic.tools.code_analysis.amp_stability import is_safe_in_float16
from torchic.amp import autocast
from torchic.nn.activations import LogSoftmax
from torchic.tools.helpers import rand
from torchic.settings import CONFIG
import cupy as cp

torchic.set_default_precision(32)
torchic.set_default_device("cpu")


#print(is_safe_in_float16(a.std))
#print(analyze_function(ls.__call__))



a = rand((100, 100))
b = rand((100, 100))

print(f"nome: {a.dtype == cp.float32}")

print(f"dtype finale: {a.dtype}")

print(cp.array([1e99], dtype=cp.float16))

with autocast:
    result = a @ b
    print(result.dtype)
# questo dovrebbe essere float32 o 16?
print(a.dtype)