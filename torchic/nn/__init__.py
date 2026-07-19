# necessario per poter importare solo nn e fare nn.Embedding e non dover fare
# from torchic.nn.networks import Embedding
from torchic.nn.activations import Sigmoid, Tanh, ReLU, Softmax, LogSoftmax
from torchic.nn.initializers import naive_init, naive_init_v2, calculate_gain, xavier_uniform, xavier_norm, kaiming_uniform, kaiming_norm
from torchic.nn.loss import LossFunction, MSELoss, CrossEntropyLoss
from torchic.nn.layers.linears import Linear, Embedding, Dropout
from torchic.nn.models.sequential import Sequential
from torchic.nn.optim import SGD, AdamV3
from torchic.nn.module import Module, ModuleList
from torchic.nn.models.sequential import Sequential
from torchic.nn.models.transformer import *
from torchic.nn.models import transformer
from torchic.nn import models