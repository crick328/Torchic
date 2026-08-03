import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.2")
from torchic import nn, set_default_precision, set_seed
from torchic.autograd.grad_context import no_grad
from torchic.cuda_tensor import Tensor
from collections.abc import Callable
from numpy.linalg import norm
import numpy as np
import warnings




def check_gradient(operation: Callable, inputs: tuple[Tensor, ...], eps=1e-6, tol=1e-7) -> None:
    '''
    Evaluate the gradients of the trainable leaf parameters. It's suggested to
    cast all tensors to float64 for better stability.
    **NOTE**: **Random layers/functions must be disabled**, otherwise the
    numerical grads will explode!

    Args:
        operation (Callable | nn.Module): can be any component that inherits from `nn.Module`
            (such as a layer or a block of layers) or a mathematical operation (matmul).
        inputs (tuple): collection of tensors that the layer should receive.
            If it's only one, use the notation (tensor, ) for correct behavior.
        eps (float): amount of the perturbation applied to every entry of the tensors.
            Used to compute numerical grads. If too big, the numerical grads won't
            approximate the gradient, while if too small it will lead to errors
            for the limited precision and numerical cancellation.
            An ideal range is [1e-5, 1e-7] for float64. [1e-3, 1e-4] for float32.
            NOTE: if `operation` contains non-differentiable functions (ReLU
            for example), this value can be pushed to the range of the float32.
        tol (float): maximum tolerable difference between each analytical and
            numerical-computed grad. Note that using a very small value doesn't
            necessarly translate to strictier checks because the system is
            more susceptible to machine precision and compond errors.
            An ideal range is [1e-6, 1e-8] for float64, [1e-4, 1e-5] for float32

    Example::

        >>> tensor = rand([BATCH, SEQ_LEN, D_MODEL], requires_grad=True)
            # it's very important to turn-off the dropout!
        >>> mha = MultiHeadAttention(D_MODEL, 2, dropout=0.0, allow_fusion=False, clip=False)
        >>> check_gradient(mha, (tensor, tensor, tensor, None))
            
    If the test fails with very small numerical grads, there are two
    possibilities:
    1) epsilon is too small;
    2) the layer contains too many tensors, which nullifies the perturbation.

    Analyitical gradients are Torchic's one, while numerical grads are
    calculated by this function.
    The idea is simple: if Torchic's grad are correct, then they should
    be identical to the ones obtained with the central incremential ratio
    limit. In reality, they wont' be identical for various reasons (numerical
    stability, finite machine precision, ecc.), so we do an approximation:
    if the difference is lower than a threshold `tol`, then we consider
    them equals.
    
    It's still an estimation: there can be false positive and
    and false negative.
    In general, if `relative_diff < 1e-7`, chanches are the grads
    are very fine. Else, if `relative_diff > 1e-2`, code should be
    revisited because it's very likely that there's an error.
    '''
    # gradienti ottenuti dalla mia backward
    analytical_grads = []
    # gradienti calcolati con il rapporto incrementale
    numerical_grads = []
    to_be_tested= []

    # ==== STEP 0: pre-prep ====
    # setto i gradienti a zero per evitare qualsiasi inquinamento
    if isinstance(operation, nn.Module):
        # disattivo in corner cose come Dropout e company
        operation.eval()
        for name, param in operation.parameters(with_names=True):
            param._grad = param._device.zeros(param.shape())

            if param.requires_grad:
                to_be_tested.append((f"param: {name}", param))
    # prendo gli input che richiedono gradiente
    for i, inp in enumerate(inputs):
        if isinstance(inp, Tensor) and inp.requires_grad:
            inp._grad = inp._device.zeros(inp.shape())
            to_be_tested.append((f"input_{i}", inp))

    if len(to_be_tested) == 0:
        warnings.warn(
            "Didn't find any Tensor with 'requires_grad=True'. Skipping" \
            f"tests on {operation}")
        return


    # ==== STEP 1: computation of analytical grads (Torchic's backprop) ====
    output = operation(*inputs)
    output.sum().backward()
    # salvo subito i grads. Potrei farlo nel ciclo subito successivo,
    # ma meglio evitare qualsiasi tipo di perturbazione ipotetica
    for name, tensor in to_be_tested:
        analytical_grads.append((name, tensor._grad.copy()))


    # ==== STEP 2: computation of numerical grad ====
    # disattivo il mio tracciamento dei gradienti, sarebbe solo overhead e potenziale noise
    with no_grad():
        for name, tensor in to_be_tested:
            #analytical_grads.append((name, tensor.grad.copy()))
            #approx_grad = []
            approx_grad = tensor._device.zeros(tensor.shape(), dtype=np.float64)

            # bisogna perturbare un singolo ingresso alla volta
            with tensor._device.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite']) as it:
                for x in it:
                    # applico le perturbazioni
                    x[...] += eps
                    out1 = operation(*inputs).sum()
                    x[...] -= 2*eps
                    out2 = operation(*inputs).sum()

                    # rapporto incrementale centrato:
                    # ∂f/∂θₙ = ( f(x)_θₙ⁺ - f(x)_θₙ⁻ ) / 2*eps
                    delta = (out1 - out2).data
                    #approx_grad.append( delta / (2 * eps) )
                    approx_grad[it.multi_index] = delta / (2 * eps)
                    x[...] += eps  # ripristino il valore originale

                    # se il calcolo e' sano, il risultato di questa
                    # sottrazione dovrebbe essere attorno all'ordine
                    # di grandezza di epsilon
                    if delta == 0.0 and np.abs(tensor.grad[it.multi_index]) > 1e-9:
                        warnings.warn(
                            "Attention: theoretical gradient is zero, but Torchic's gradient is not! "
                            f"Choose a bigger eps. Name: {name}, index: {it.multi_index}",
                            category=UserWarning
                        )
    
            # prendo i grad dei singoli scalari e li uso per creare
            # un tensor avente la stessa shape del parametro
            #numerical_grads.append(tensor._device.array(approx_grad).reshape(tensor.shape()))
            numerical_grads.append(approx_grad.reshape(tensor.shape()))


    # ==== STEP 3: gradient check with L2 norm comparison ====
    for (name, grad), grad_approx  in zip(analytical_grads, numerical_grads):
        # appiattiasco a 1D per calcolare la norma L2 su qualsiasi shape
        grad_norm = norm(grad.reshape(-1))
        approx_norm = norm(grad_approx.reshape(-1))

        # se entrambi i gradienti sono vicini a zero
        if grad_norm < 1e-7 and approx_norm < 1e-7:
            print(f"✅ Gradient Check PASSED per '{name}' (Gradiente nullo: ~0.0)")
            continue

        relative_diff = norm(grad - grad_approx) / (grad_norm + approx_norm )#+ 1e-12)
        # la differenza tra i due deve essere molto piccola. L'idea e' che se la tua
        # backprop e' corretta, dovrebbe essere uguale al calcolo numerico. Nella pratica,
        # accettiamo un po' di errore per via della precisione finita dei float e altro noise,
        # quindi li riteniamo uguali se la loro differenza e' quasi zero (< eps)
        if relative_diff > tol:
            print(grad_norm, approx_norm)
            print(f"❌ Gradient check failed in {name}! {relative_diff:.2e}. Shape: {grad.shape}")
            print("Analytical Gradient:\n", grad)
            print("Numerical Gradient:\n", grad_approx)
        else:
            print(f"✅ Gradient check successfull for {name}! {relative_diff:.2e}")




'''
from torchic.nn.models.transformer import MultiHeadAttention
from torchic import rand
set_default_precision(64)
set_seed(42)



D_MODEL = 20
VOCAB_SIZE = 30
SEQ_LEN = 30
BATCH = 1
DROPOUT = 0.0
D_FF = 20



# Linear OK!
#tensor = rand([5,], requires_grad=False)
#linear = torchic.nn.Linear(5,5)
#check_gradient(linear, (tensor, ))

# MHA OK!
tensor = rand([BATCH, SEQ_LEN, D_MODEL], requires_grad=True)
mha = MultiHeadAttention(D_MODEL, 2, DROPOUT, allow_fusion=False, clip=False)
check_gradient(mha, (tensor, tensor, tensor, None))
'''