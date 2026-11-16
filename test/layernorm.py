import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic.nn.models.transformer import LayerNormalization
import torchic
import torch


data = torchic.rand([2,3,4], dtype=torchic.float32)
a = torch.tensor(data.data, requires_grad=True)
ten = torchic.Tensor(data, requires_grad=True)


# NOTE: l'implementazione del LayerNorm di PyTorch che utilizzano
# nel transformer da risultati differenti da quella normale.
# La mia torna con quella del transformer e di gpt4
# DIFFERENZA TROVATA! La formula e' identica, ma il default
# LayerNorm di PyTorch opera su dim=None, mentre il transformer
# deve operare su embedding_size (dim=-1)
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



d_model=a.shape[-1]
torch_norm = torch.nn.LayerNorm(d_model, eps=1e-12)
t_norm = LayerNorm(d_model)
my_norm = LayerNormalization(d_model)


mean = a.mean(dim=None, keepdim=True)
var = a.var(dim=None, unbiased=False, keepdim=True)


print(torch_norm(a))
#print((a-mean)/(torch.sqrt(var+1e-12)))
#print("========")
print(t_norm(a))
print(my_norm(ten))