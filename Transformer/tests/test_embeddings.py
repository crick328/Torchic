import torch
from torch import nn
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.0")
import torchic

vocab_size = 5
d_model = 2

embedding = nn.Embedding(vocab_size, d_model)
x = embedding(torch.LongTensor([3,4]))
print(x.shape)
x.mean().backward()
print(embedding.weight.grad)

emb = torchic.nn.Embedding(5, 3)
print(emb.weights)
y = emb(torchic.Tensor([1,4]))
print(y)
y.mean().backward()
print(emb.weights.grad)