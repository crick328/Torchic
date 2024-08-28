import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.2")
from torchic import nn, Tensor

a = Tensor([1,2],True)
b = Tensor([1,2])
c = a+b

print(sys.getrefcount(a)) #2
print(a._get_ref_count()) #3
print(sys.getrefcount(b)) #2
print(a._get_ref_count()) #3
print(b._get_ref_count()) #3