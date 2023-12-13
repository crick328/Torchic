import torch
from test6 import Tensor


# MY
my_x = Tensor([0.,1.,3.], requires_grad=False) # True per testing
my_W = Tensor([
    [1.,-1.],
    [5.,5.],
    [-5.,-1.]
    ], requires_grad=True)
my_b = Tensor([-8.,4.], requires_grad=True)

# TORCH
torch_x = torch.tensor([0.,1.,3.], requires_grad=False)
torch_W = torch.tensor([
        [1.,-1.],
        [5.,5.],
        [-5.,-1.]
    ], requires_grad=True)
torch_b = torch.tensor([-8.,4.], requires_grad=True)


# OPERAZIONI
my_k = my_x.matmul(my_W)     #[-10, 2]
my_f = my_b+my_k
my_f.backward(Tensor([1.,1.]))


torch_k = torch_x.matmul(torch_W)
torch_f=torch_k+torch_b

torch_k.retain_grad()
torch_f.retain_grad()
print(torch_W._grad)
torch_W.grad = torch.tensor([[-235.,-34.],[-235.,-34.],[-235.,-34.]])
torch_f.backward(torch.tensor([1.,1.]))


print(f"Gradient matrice. MY: {my_W.grad}, TORCH: {torch_W.grad}")
print(f"Gradient bias.  MY: {my_b.grad}, TORCH: {torch_b.grad}")
print(f"Gradient risultato matmul.  MY: {my_k.grad}, TORCH: {torch_k.grad}")


import numpy as np

np_x = np.array([0.,1.,3.])
np_W = np.array([
        [1.,-1.],
        [5.,5.],
        [-5.,-1.]
    ])
torch_A = torch.tensor([
    [2],
    [2],
    [2]
])
np_A = np.array([
    [[2,3]],
    [[2,3]],
    [[2,3]]
])

'''
torch_y=torch_x.pow(3)
torch_y.backward(torch.tensor([1.,1.,1.]))
print(torch_y)
print(torch_x.grad)
'''