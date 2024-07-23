import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.0")

from torchic import Tensor, int64, nn

import torch
from torch import tensor
from torch.nn import CrossEntropyLoss


values = [
    [0.2, 0.3, 0.5],  # Logits per le tre classi del primo campione
    [0.1, 0.6, 0.3],  # Logits per le tre classi del secondo campione
    [0.4, 0.2, 0.4]   # Logits per le tre classi del terzo campione
]
#values = [0.2, 0.3, 0.5]
true_val = [2, 1, 1] # [3, 1, 1] da' giustamente errore per entrambi
#true_val = [
#    [0., 1., 2.],  # Logits per le tre classi del primo campione
#    [1., 0., 0.],  # Logits per le tre classi del secondo campione
#    [1., 2., 1.]   # Logits per le tre classi del terzo campione
#]
reduction = "mean"
ignore_index = 1
smooth=0.3
#values = [10., 5., 3., 1., 4.]
#true_val = [1., 0., 0., 0., 0.]


my_ten = Tensor(values, requires_grad=True)
my_tar = Tensor(true_val, dtype=int64)
my_loss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, label_smoothing=smooth)

tor_ten = tensor(values, requires_grad=True)
tor_tar = tensor(true_val)
print(tor_tar.dtype)
torch_loss = CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, label_smoothing=smooth)

my_res = my_loss(my_ten, my_tar)
tor_res = torch_loss(tor_ten, tor_tar)

my_res.mean().backward()
tor_res.mean().backward()


print(my_res)
print(tor_res)
print("----")
print(my_ten.grad)
print(tor_ten.grad)