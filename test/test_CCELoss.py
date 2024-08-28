import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.3")

from torchic import Tensor, int64, nn, rand, randint, draw_graph
import torch
from torch import tensor
from torch.nn import CrossEntropyLoss



REDUCTION = "mean"
IGNORE_INDEX = 1
SMOOTH=0.1
INPUT_SHAPE = (20, 10)
TARGET_SHAPE = (20, )
DEC_VOCAB_SIZE = 5


my_ten = rand(INPUT_SHAPE, requires_grad=True)
my_tar = randint(TARGET_SHAPE, high=DEC_VOCAB_SIZE)
my_loss = nn.CrossEntropyLoss(reduction=REDUCTION, ignore_index=IGNORE_INDEX, label_smoothing=SMOOTH)

tor_ten = tensor(my_ten.data, requires_grad=True)
tor_tar = tensor(my_tar.data, dtype=torch.int64)
torch_loss = CrossEntropyLoss(reduction=REDUCTION, ignore_index=IGNORE_INDEX, label_smoothing=SMOOTH)

my_res = my_loss(my_ten, my_tar)
tor_res = torch_loss(tor_ten, tor_tar)

draw_graph(my_res)

my_res.backward()
tor_res.backward()


print(my_res)
print(tor_res)
print("----")
print(my_ten.grad)
print(tor_ten.grad)