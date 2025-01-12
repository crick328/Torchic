import warnings
#import sys
#sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
from torchic import nn, set_default_precision
from torchic.cuda_tensor import Tensor
from torchic.nn.models import transformer as transformer
from torchic.nn.models.transformer import MultiHeadAttention, LayerNormalization
from torchic.tools.helpers import rand, randint
from numpy.linalg import norm
import numpy as np

# il gradient check richiede float64
set_default_precision(64)



D_MODEL = 20
VOCAB_SIZE = 30
SEQ_LEN = 30
BATCH = 1
DROPOUT = 0.0
D_FF = 20




def check_gradient(layer: nn.Module, inputs: tuple[Tensor], eps=1e-7, tolerance=1e-3) -> None:
    '''
    Evaluate the gradients of the trainable leaf parameters.
    If the test fails with very small numerical grads, there are two
    possibilities:
    1) the epsilon is too small (1e-4 to 1e-8 is the ideal range);
    2) the layer contains too many tensors, which nullifies the perturbation.

    NOTE: disable all randomic layers and functions! Otherwise, the relative
    difference will be too big and results in enormous numerical gradients!

    Args:
        layer (nn.Module): can be any component that inherits from `nn.Module`
            such as a layer or a block of layers.
        inputs (tuple): collection of tensors that the layer should receive.
            If it's only one, use the notation (tensor, ) for correct behavior.
    '''
    # gradienti ottenuti dalla mia backward
    analytical_grads = []
    # gradienti calcolati con il rapporto incrementale
    numerical_grads = []
    smallest_magnitude = eps * tolerance

    # 1 step: calcolo del gradiente analitico (backprop)
    layer_output = layer(*inputs)
    layer_output.sum().backward()

    # 2 step: calcolo del gradiente numerico (gradiente approssimato)
    # bisogna iterare solo sui tensor che chiedono il grad
    for name, param in layer.parameters(with_names=True):
        analytical_grads.append(param.grad)
        approx_grad = []

        # bisogna perturbare un singolo ingresso alla volta
        with param._device.nditer(param.data, flags=['multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                x[...] += eps
                out1 = layer(*inputs).sum()
                x[...] -= 2*eps
                out2 = layer(*inputs).sum()
                delta = (out1 - out2).data
                # ∂f/∂θₙ = ( f(x)_θₙ⁺ - f(x)_θₙ⁻ ) / 2*eps
                approx_grad.append( delta / (2 * eps) )
                x[...] += eps  # ripristino il valore originale

                # se il calcolo e' sano, il risultato di questa
                # sottrazione dovrebbe essere attorno all'ordine
                # di grandezza di epsilon
                #if np.abs(delta) < smallest_magnitude:
                if delta == 0.0:
                    index = it.multi_index

                    warnings.warn(
                        "Attention: some numerical gradients are too "
                        "small. Choose a bigger eps or lower the tensor "
                        f"count of the given layer. "
                        f"Name: {name}, shape: {param.shape()}, index: {index}",
                        category=UserWarning
                    )

        # prendo i grad dei singoli scalari e li uso per creare
        # un tensor avente la stessa shape del parametro
        numerical_grads.append(param._device.array(approx_grad).reshape(param.shape()))

    # 3 step: confronto con norma L2
    for grad_approx, grad in zip(numerical_grads, analytical_grads):
        relative_diff = norm(grad - grad_approx, 2) / (norm(grad, 2) + norm(grad_approx, 2))

        # la differenza tra i due deve essere molto piccola
        if relative_diff > 1e-7:
            print(f"Gradient check failed! {relative_diff}. Shape: {grad.shape}")
            #print("Analytical Gradient:", grad)
            #print("Numerical Gradient:", grad_approx)
            #break
        else:
            print(f"Gradient check successfull! {relative_diff} (> 1e-12?)")


# Linear OK!
#tensor = rand([5,], requires_grad=False)
#linear = torchic.nn.Linear(5,5)
#check_gradient(linear, (tensor, ))

# MHA OK!
#tensor = rand([BATCH, SEQ_LEN, D_MODEL], requires_grad=False)
#mha = MultiHeadAttention(D_MODEL, 2, DROPOUT)
#check_gradient(mha, (tensor, tensor, tensor, None))

# LNORM OK!
#tensor = rand([BATCH, SEQ_LEN, D_MODEL], requires_grad=False)
#lnorm = LayerNormalization(D_MODEL)
#check_gradient(lnorm, (tensor, ))

# FFB alcune volte crea nan perche' out1-out2 = 0.0!
#tensor = rand([BATCH, SEQ_LEN, D_MODEL], requires_grad=True)
#ffb = transformer.FeedForwardBlock(D_MODEL, D_FF, DROPOUT)
#check_gradient(ffb, (tensor, ))

# EMB + PE genera di nuovo out1-out2 = 0.0!
#tensor = randint([BATCH, SEQ_LEN], high=VOCAB_SIZE, requires_grad=True)
#temb = transformer.TransformerEmbeddings(D_MODEL, VOCAB_SIZE, SEQ_LEN, DROPOUT)
#check_gradient(temb, (tensor, ), eps=1e-7)

# EMB idem, delta = 0.0
#tensor = randint([BATCH, SEQ_LEN], high=VOCAB_SIZE, requires_grad=True)
#temb = nn.Embedding(VOCAB_SIZE, D_MODEL, _init_funct=nn.xavier_uniform)
#check_gradient(temb, (tensor, ))

# Encoder non ok...?
#tensor = rand([BATCH, SEQ_LEN, D_MODEL])
#enc = transformer.EncoderLayer(D_MODEL, 2, D_FF, DROPOUT)
#check_gradient(enc, (tensor, None), eps=1e-7)


'''
torchic.set_default_precision(32)
sys.setrecursionlimit(1500)

enc_input = randint([BATCH, SEQ_LEN], high=VOCAB_SIZE, requires_grad=True)
dec_input = randint([BATCH, SEQ_LEN], high=VOCAB_SIZE, requires_grad=True)
target = randint([BATCH, SEQ_LEN], high=VOCAB_SIZE, requires_grad=True)
model = transformer.Transformer(D_MODEL, VOCAB_SIZE, VOCAB_SIZE, SEQ_LEN, DROPOUT, 6, 2, D_FF)
#check_gradient(model, (enc_input, dec_input, None, None), eps=1e-7)

encoder_mask = Tensor(enc_input.data != 1, dtype=DTYPE)
encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)

decoder_mask = Tensor(dec_input.data != 1, dtype=DTYPE)
decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)
decoder_mask.data = decoder_mask.data & Tensor(np.triu( np.ones([1, dec_input.size(0), dec_input.size(0),]) ), dtype=DTYPE).data == 0

loss_fn = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)

pred = model(enc_input, dec_input, encoder_mask, decoder_mask)
pred.mean().backward()
#print(pred.view(-1, VOCAB_SIZE))
#print(target)
#loss = loss_fn(pred.view(-1, VOCAB_SIZE), target.view(-1))
#loss.backward()
def param_state(module: nn.Module, _name: str=""):
    if isinstance(module, nn.ModuleList):
        for layer in module:
            param_state(layer)
    else:
        for name, param in module:
            if isinstance(param, Tensor) and param.requires_grad:
                print(f"{_name}.{name}:\n{param.grad}")
            elif isinstance(param, nn.Module):
                param_state(param, name)



for name, param in model.__dict__.items():
    if isinstance(param, nn.Module):
        param_state(param, name)
    elif isinstance(param, Tensor) and param.requires_grad:
        print(f"{name}:\n{param.grad}")

print(model)
'''