import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torchic
from torchic.cuda_tensor import Tensor
import torch
from torch import tensor
import numpy as np

import tempfile

temp_dir = tempfile.gettempdir()
print(f"La cartella temporanea di sistema è: {temp_dir}")



my_relu = torchic.nn.ReLU()
my_sigmoid = torchic.nn.Sigmoid()
my_soft0 = torchic.nn.Softmax(0)
my_soft1 = torchic.nn.Softmax(1)
my_soft2 = torchic.nn.Softmax(2)
my_logsoft0 = torchic.nn.LogSoftmax(0)
my_logsoft1 = torchic.nn.LogSoftmax(1)
my_logsoft2 = torchic.nn.LogSoftmax(2)

torch_relu = torch.nn.ReLU()
torch_sigmoid = torch.nn.Sigmoid()
torch_soft0 = torch.nn.Softmax(0)
torch_soft1 = torch.nn.Softmax(1)
torch_soft2 = torch.nn.Softmax(2)
torch_logsoft0 = torch.nn.LogSoftmax(0)
torch_logsoft1 = torch.nn.LogSoftmax(1)
torch_logsoft2 = torch.nn.LogSoftmax(2)
