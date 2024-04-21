import torch
from replica import nn, CudaTensor

torch.set_printoptions(precision=8)

class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        # Calcola il softmax lungo la dimensione specificata
        max_vals, _ = torch.max(input, dim=dim, keepdim=True)
        input_exp = torch.exp(input - max_vals)
        softmax = input_exp / torch.sum(input_exp, dim=dim, keepdim=True)
        
        # Memorizza le variabili utili per il backward pass
        ctx.save_for_backward(softmax)
        ctx.dim = dim
        
        return softmax

    @staticmethod
    def backward(ctx, grad_output):
        # Estrai le variabili memorizzate durante il forward pass
        softmax, = ctx.saved_tensors
        dim = ctx.dim

        # Calcola il vettore gradiente della softmax
        #print(softmax)
        #print(grad_output * softmax)
        #print(torch.sum(grad_output * softmax, dim=dim, keepdim=True))
        print("Torch:")
        print(grad_output - torch.sum(grad_output * softmax, dim=dim, keepdim=True))
        softmax_grad = softmax * (grad_output - torch.sum(grad_output * softmax, dim=dim, keepdim=True))

        return softmax_grad, None  # None corrisponde al gradiente rispetto a dim, che non è un parametro della funzione softmax



# Wrapper per l'utilizzo più semplice
def GPTsoftmax(input, dim):
    return SoftmaxFunction.apply(input, dim)

# Utilizzo
n_dim = 1

# GRAZIE https://discuss.pytorch.org/t/numerical-differences-between-numpy-and-pytorch/89607
# il problema era solo nel float32 di torch
input_tensor = torch.randn(2, 3, 4, requires_grad=True, dtype=torch.float64)
mio_tensor = CudaTensor(input_tensor.data, requires_grad=True, device='cpu')

pytorch_softmax = torch.nn.Softmax(n_dim)
mia_softmax = nn.Softmax(n_dim)

out1 = GPTsoftmax(input_tensor, dim=n_dim)
out2 = pytorch_softmax(input_tensor)
out3 = mia_softmax(mio_tensor)
out1.retain_grad()
out2.retain_grad()

# IDENTICI
#print(input_tensor)
#print(mio_tensor)


out1.mean().backward()
#print(input_tensor.grad)

input_tensor.grad = None

out2.mean().backward()
#print(input_tensor.grad)

out3.mean().backward()
#print(mio_tensor.grad.data)

import numpy as np
print("All close:")
try:
    print(torch.allclose(input_tensor.grad, torch.tensor(np.array(mio_tensor.grad), dtype=torch.float32)))
except:
    print(torch.allclose(input_tensor.grad, torch.tensor(np.array(mio_tensor.grad.data.get()), dtype=torch.float32)))