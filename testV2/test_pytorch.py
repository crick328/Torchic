import torch

# Creiamo un tensor iniziale
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device="cpu", dtype=torch.float32)

# SCENARIO 1: Chiamata a vuoto (Stesso device e stesso dtype)
y = x.to(device="cpu", dtype=torch.float32)
print(x is y)  # Output: True (È lo stesso oggetto, NON ha clonato)

# SCENARIO 2: Cambio di Dtype
z = x.to(dtype=torch.int32, copy=False)
print(x is z)  # Output: False (Ha dovuto clonare e allocare nuova memoria)

# SCENARIO 3: Forzare la copia con copy=True
w = x.to(copy=True)
print(x is w)  # Output: False (Forzato a clonare anche se device/dtype sono identici)

print(y.requires_grad, z.requires_grad, w.requires_grad)
print(y.grad_fn, z.grad_fn, w.grad_fn)