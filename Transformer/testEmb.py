import torch

# Creiamo un tensor a 4 dimensioni
tensor = torch.randn(2, 3, 4, 5)

# Scambiare la prima e la seconda dimensione
transposed_tensor = tensor.transpose(0, 1)
print(transposed_tensor.shape)
