from preset import torchic
from torchic import Tensor
from torchic.settings import get_current_backend
from torchic.nn import Softmax, LogSoftmax
from torchic.autograd.functions import SoftmaxBackward


torchic.set_default_precision(32)


def compare_and_check(sp_tensors, hp_tensors):
    lib = get_current_backend()
    count = 0
    for sp_tensor, hp_tensor in zip(sp_tensors, hp_tensors):
        assert all(lib.isfinite(sp_tensor))
        assert all(lib.isfinite(hp_tensor)), hp_tensor

        assert lib.allclose(sp_tensor.data, hp_tensor.data), (sp_tensor.data, hp_tensor.data)
        
        if sp_tensor.grad is not None:
            assert lib.allclose(sp_tensor.grad, hp_tensor.grad), (sp_tensor.grad, hp_tensor.grad)
        count += 1


# 1e5 e' overflow per float16
# 1e-8 e' underflow
sp_tensor = Tensor([1e-7, 1e-1, 1e4, -1e-4, -1e2, 5.5, -7.2], requires_grad=True)
hp_tensor = sp_tensor.detach().clone().to(torchic.float16)
hp_tensor.requires_grad = True


# ==== Softmax Benchmark ====
softmax = Softmax()
class TestSoftmax:
    __slots__ = ("dim")

    # Può essere calcolata solo su una dimensione
    def __init__(self, dim=0) -> None:
        self.dim = dim

    def __call__(self, x: Tensor) -> Tensor:
        # questa singola linea di codice previene overflow senza modificare
        # l'efficacia del gradient.
        shiftx = x.data - x.data.max(axis=self.dim, keepdims=True)
        #exps = x._device.exp(shiftx)
        #softmax = exps / x._device.sum(exps, axis=self.dim, keepdims=True)
        #shiftx = shiftx.astype(x._device.float32)
        softmax = x._device.exp(shiftx)
        softmax /= x._device.sum(softmax, axis=self.dim, keepdims=True)
        output = Tensor(softmax)#.to(torchic.float16)
        
        if x.requires_grad:
            # da qua in avanti, requires_grad non sarà più modificabile perché
            # non è più considerato leaf tensor
            output.grad_fn = SoftmaxBackward(x, self.dim, softmax)
        return output
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"
test_softmax = TestSoftmax()

sp_result = softmax(sp_tensor)
print(hp_tensor)
hp_result = test_softmax(hp_tensor)

sp_result.sum().backward()
hp_result.sum().backward()

compare_and_check(
    [sp_result],
    [hp_result]
)


