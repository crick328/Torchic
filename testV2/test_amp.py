from preset import torchic
from torchic.amp.autocast_ctx import autocast
from torchic.amp.wrappers import amp_cache


USE_BFLOAT16=False

torchic.set_seed(42)
torchic.set_default_precision(32)
torchic.set_default_device("cuda")
torchic.use_brain_float(USE_BFLOAT16)

a = torchic.rand((10, 20), requires_grad=True)
b = torchic.rand((20, 10))
c = torchic.rand((10, 10))

with autocast():
    with torchic.no_grad():
        x = a @ b
        # beccato il problema: i master sono pur sempre in float32. Se la matmul
        # mi da' un float16 ed io lo sommo con un float32 del bias, non c'e' santo
        # che tenga... Devono quindi essere wrappate e gestite anche queste op neutrali.
        y = x+c
        print(amp_cache)
        #print(y.dtype, y.grad_fn.ctx)
        
        print(y.mean)
        #y.mean().backward()