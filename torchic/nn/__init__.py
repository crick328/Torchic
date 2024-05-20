# necessario per poter importare solo nn e fare nn.Embedding e non dover fare
# from nn.networks import Embedding
from torchic.nn.activation import Sigmoid, ReLU, Softmax, LogSoftmax
from torchic.nn.initializers import naive_init, naive_init_v2, xavier_uniform, xavier_norm
from torchic.nn.loss import SELoss, CrossEntropyLoss
from torchic.nn.networks import Linear, Embedding, Dropout, Sequential