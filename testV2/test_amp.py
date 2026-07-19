from preset import torchic
from torchic import Tensor
from torchic.amp.amp_stability import is_safe_in_float16 #, _classify_lib_functions
from torchic.amp.autocast_ctx import autocast
from torchic.nn.activations import LogSoftmax, ReLU
from torchic.tools.helpers import rand
from torchic.settings import CONFIG
import cupy as cp


#_classify_lib_functions()




print(is_safe_in_float16.__qualname__)


print("\n=== Start! ===\n")
torchic.set_default_precision(32)
torchic.set_default_device("cuda")


#print(is_safe_in_float16(a.std))
#print(analyze_function(ls.__call__))


a = rand((100, 100), requires_grad=True)
b = rand((100, 100))

with autocast():
    #for _ in range(10):
    #a.matmul(b)
    #a.matmul(b)
    #a.matmul(b)
    #a.matmul(b)
    c=a.matmul(b)
    assert c.dtype == torchic.float16

print(c.dtype)
exit()

relu = ReLU()

print(is_safe_in_float16(relu.__call__))

# NOTE: il wrapper si attiva anche senza autocast!
with autocast():
    #c = a.sqrt()
    c = relu(a)
    #result = a.matmul(b)
print(c.dtype)
exit()
# da' True anche se il tensor e' su cpu
#print(f"nome: {a.dtype == cp.float32}")

#print(f"dtype finale: {a.dtype}")

#print(cp.array([1e99], dtype=cp.float16))
from torchic.settings import  is_mixed_precision_enabled, get_mixed_precision_dtype
#print(cp.dtype(cp.int64) > cp.dtype(cp.float16))

logsoft = LogSoftmax()

print("-----")
with autocast():
    print(a.device)
    print(a.dtype)
    #result = a @ b
    result = a.matmul(b)
    result = logsoft(result)
    #result = a.sqrt()
    print(result.dtype)
    print(result.device)
# questo dovrebbe essere float32 o 16?
#print(a.dtype)