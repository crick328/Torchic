# necessario per poter importare solo nn e fare nn.Embedding e non dover fare
# from nn.networks import Embedding
from replica.nn.activation import Sigmoid, ReLU, Softmax
from replica.nn.initializers import naive_init, naive_init_v2, xavier_uniform, xavier_norm
from replica.nn.loss import SELoss
from replica.nn.networks import LLayer, Embedding, Dropout, NNetwork