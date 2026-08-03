from preset import torchic, create_random_tensors,compare_tensors
from torchic import Tensor, draw_graph
from torch import tensor
import numpy as np


'''
a=Tensor([1.2,1.2, 99, 90], requires_grad=True)
s=a.clone()
b=a.clamp(0, 90)
c=s.clamp_(0, 90)

#print(b, c)

b.mean().backward()
print(a.grad)
a._grad=None
c.mean().backward()
print(a.grad)
'''

tt, myt = create_random_tensors((10, 20))

mya = myt.clone()+Tensor(-8)
myb = myt.clone()+Tensor(-3)
myc = mya.exp()#/Tensor(8)
myd = myb.exp()#/Tensor(3)
mye = myc+myd
#draw_graph(mye)
mye.mean().backward()

ta = tt.clone()-8
tb = tt.clone()-3
tc = ta.exp()#/8
td = tb.exp()#/3
te = tc+td
te.mean().backward()

print(myt.grad.sum(), tt.grad.sum())

compare_tensors(tt, myt)