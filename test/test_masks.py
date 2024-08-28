import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.1")
import torchic
import numpy as np
import torch

# Array di esempio
array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
# Maschera composta da 0 e 1
mask = np.array([1, 0, 1, 0, 1], dtype=np.float32)


arr = torchic.Tensor(array, dtype=np.float32, requires_grad=True)
imask = torchic.Tensor(mask, dtype=np.int64)
#arr.data[imask.data == 0] = -1e9
#arr.data = np.where(imask.data == 0, -1e9, arr.data)
masked_arr = arr.masked_fill(imask, -1e9)
print(masked_arr)
output = masked_arr.sum()
output.backward()
print(arr.grad)



tensor = torch.tensor(array, dtype=torch.float32, requires_grad=True)
tmask = torch.tensor(mask, dtype=torch.float32)
# Applica masked_fill: dove mask è 0, imposta il valore a -1e9
masked_tensor = tensor.masked_fill(tmask == 0, -1e9)
print(masked_tensor)
output = masked_tensor.sum()
output.backward()
print(tensor.grad)