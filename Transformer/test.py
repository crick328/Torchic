from transformer import Transformer
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 1.3")
from torchic import Tensor
import numpy as np
import cupy as cp
import torch


data = [[1.,2.,0.],[1.,2.,3.]]
dim = 1


a = torch.tensor(data, requires_grad=True)
#print(a.var(dim))
#a.var(dim).mean().backward()
#print(a.grad)

ten = Tensor(data, requires_grad=True)
#print(ten.var(dim))
#tmp=ten.var(dim)
#tmp.mean().backward()
#print(ten.grad)


# NOTE: l'implementazione del LayerNorm di PyTorch che utilizzano
# nel transformer da risultati differenti da quella normale.
# La mia torna con quella del transformer e di gpt4
# DIFFERENZA TROVATA! La formula e' identica, ma il default
# LayerNorm di PyTorch opera su dim=None, mentre il transformer
# deve operare su embedding_size (dim=-1)
from transformer import LayerNormalization
class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


d_model=a.shape
torch_norm = torch.nn.LayerNorm(d_model, eps=1e-12)
mean = a.mean(dim=None, keepdim=True)
var = a.var(dim=None, unbiased=False, keepdim=True)
print((a-mean)/(torch.sqrt(var+1e-12)))
t_norm = LayerNorm(d_model)
my_norm = LayerNormalization(d_model)

print(torch_norm(a))
print(t_norm(a))
print(my_norm(ten))





model = Transformer(
    d_model=512,
    # length OHE src vector
    encoder_vocab_size=5,
    # length OHE trg vector
    dec_vocab_size=10,
    # seq_len
    max_len=20,
    drop_prob=0.5,
    # enc e dec n layers
    n_layers=6,
    num_heads=8,
    dim_feed_forward=2048
)
