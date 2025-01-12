from preset import torchic
from torchic.autograd.grad_ctx import no_grad, enable_grad
from torchic.settings import is_grad_enabled
from torchic import Tensor

a = Tensor([1.,2.], requires_grad=True)

print(False and (False or True))

print(is_grad_enabled())
with no_grad():  # Imposta grad_enabled = False
    # Qui grad_enabled è False
    print(is_grad_enabled())
    b = a - a
    print(b.grad_fn)
    with enable_grad():  # Imposta grad_enabled = True
        print(is_grad_enabled())