from test_NNet import NNetwork, LLayer, SELoss, Sigmoid, load_model_type1, save_model_type1, ReLU
from test10_tensor import Tensor
from optimizers import SGD

W1 = Tensor([[0.5794, 0.6730, 0.8875, 0.9021, 0.9662, 0.8482],
        [0.5435, 0.7482, 0.7800, 0.9295, 0.6639, 0.9301],
        [0.2938, 0.0957, 0.4389, 0.5478, 0.2955, 0.3213],
        [0.3429, 0.4262, 0.1190, 0.1769, 0.3686, 0.7412],
        [0.6553, 0.2470, 0.0884, 0.1499, 0.8788, 0.7797],
        [0.3195, 0.6272, 0.6759, 0.3161, 0.8730, 0.8096],
        [0.0216, 0.4054, 0.3292, 0.8136, 0.7822, 0.6545],
        [0.7702, 0.4767, 0.7654, 0.8625, 0.6182, 0.6024],
        [0.4077, 0.1423, 0.0370, 0.8075, 0.4355, 0.0881],
        [0.0561, 0.0848, 0.2390, 0.1369, 0.3916, 0.9968],
        [0.5500, 0.7078, 0.2318, 0.1214, 0.8642, 0.1792],
        [0.0261, 0.6829, 0.4548, 0.6723, 0.8068, 0.5544]], requires_grad=True)
B1 = Tensor([0.8918, 0.3249, 0.6569, 0.0247, 0.9285, 0.0806], requires_grad=True)
W2 = Tensor([[0.5671, 0.5108, 0.6357],
        [0.7820, 0.9338, 0.9623],
        [0.8650, 0.4352, 0.3510],
        [0.4005, 0.7835, 0.2309],
        [0.9725, 0.4179, 0.6448],
        [0.7971, 0.2383, 0.8744]], requires_grad=True)
B2 = Tensor([0.8601, 0.5030, 0.2870], requires_grad=True)


model = NNetwork((
            LLayer(12, 6,
                   _w_matrix=W1,
                   _b_vector=B1),
            Sigmoid(),
            LLayer(6, 3,
                   _w_matrix=W2,
                   _b_vector=B2),
            Sigmoid()
        ))
optimizer = SGD(model.parameters())
output = model([0,4,3,4,5,6,4,3,56,6,3,4])
loss = SELoss(output, [7.,-10.,3.])
loss.mean().backward()


# NOTA: c'è differenza tra torch.Tensor e torch.tensor:
# https://stackoverflow.com/questions/53201921/torch-tensor-with-requires-grad-parameter
# NOTA: RuntimeError: Only Tensors of floating point and complex dtype can require gradients
# https://discuss.pytorch.org/t/dont-understand-why-only-tensors-of-floating-point-dtype-can-require-gradients/158282

import torch
sigmoid=torch.nn.Sigmoid()


W1 = torch.tensor([[0.5794, 0.6730, 0.8875, 0.9021, 0.9662, 0.8482],
        [0.5435, 0.7482, 0.7800, 0.9295, 0.6639, 0.9301],
        [0.2938, 0.0957, 0.4389, 0.5478, 0.2955, 0.3213],
        [0.3429, 0.4262, 0.1190, 0.1769, 0.3686, 0.7412],
        [0.6553, 0.2470, 0.0884, 0.1499, 0.8788, 0.7797],
        [0.3195, 0.6272, 0.6759, 0.3161, 0.8730, 0.8096],
        [0.0216, 0.4054, 0.3292, 0.8136, 0.7822, 0.6545],
        [0.7702, 0.4767, 0.7654, 0.8625, 0.6182, 0.6024],
        [0.4077, 0.1423, 0.0370, 0.8075, 0.4355, 0.0881],
        [0.0561, 0.0848, 0.2390, 0.1369, 0.3916, 0.9968],
        [0.5500, 0.7078, 0.2318, 0.1214, 0.8642, 0.1792],
        [0.0261, 0.6829, 0.4548, 0.6723, 0.8068, 0.5544]], requires_grad=True)
B1 = torch.tensor([0.8918, 0.3249, 0.6569, 0.0247, 0.9285, 0.0806], requires_grad=True)
W2 = torch.tensor([[0.5671, 0.5108, 0.6357],
        [0.7820, 0.9338, 0.9623],
        [0.8650, 0.4352, 0.3510],
        [0.4005, 0.7835, 0.2309],
        [0.9725, 0.4179, 0.6448],
        [0.7971, 0.2383, 0.8744]], requires_grad=True)
B2 = torch.tensor([0.8601, 0.5030, 0.2870], requires_grad=True)


x = torch.tensor([0.,4.,3.,4.,5.,6.,4.,3.,56.,6.,3.,4.], requires_grad=True)


k1 = x.matmul(W1)
f1=k1+B1
h1=sigmoid(f1)

k2 = h1.matmul(W2)
f2=k2+B2
h2=sigmoid(f2)

h1.retain_grad()

loss_fn = torch.nn.MSELoss(reduction='none')
loss = loss_fn(h2, torch.tensor([7.,-10., 3.]))
loss.mean().backward()

# NOTA: l'assert dà errore perché torch azzera le componenti con numeri troppo piccoli
print(B1.grad)
print(model.structure[0].b_vector.grad)

assert W1.grad == model.structure[0].w_matrix.grad



#save_model_type1(model)

