import torch

# Input simulato
batch_size, seq_len, d_model, head_num = 2, 4, 8, 2
qkv = torch.randn(batch_size, seq_len, 3 * d_model)

# Prima divisione
original_sum = torch.sum(qkv[:, :, :d_model])

# Forza la contiguità del tensore prima del reshape
qkv = qkv.contiguous()

# Reshape e split
head_dim = d_model // head_num
print("Original is contiguous:", qkv.is_contiguous())
reshaped = qkv.reshape(batch_size, seq_len, head_num, 3 * head_dim).contiguous()
print("Reshaped is contiguous:", reshaped.is_contiguous())
split_sum = torch.sum(reshaped[:, :, :, :head_dim])

# Somma totale (flattened)
orig_sum_flat = torch.sum(qkv.flatten())
reshaped_flat = torch.sum(reshaped.flatten())
print("Original sum (flattened):", orig_sum_flat)
print("Reshaped sum (flattened):", reshaped_flat)

# Slicing
original_slice = qkv[:, :, :d_model]
reshaped_slice = reshaped[:, :, :, :head_dim]
print(f"Original stride: {original_slice.stride()}")
print(f"Reshaped stride: {reshaped_slice.stride()}")
print("Original slice sum (manual):", torch.sum(original_slice.flatten()))
print("Reshaped slice sum (manual):", torch.sum(reshaped_slice.flatten()))

# Media
original_mean = torch.mean(qkv[:, :, :d_model])
reshaped_mean = torch.mean(reshaped[:, :, :, :head_dim])
print("Original mean:", original_mean)
print("Reshaped mean:", reshaped_mean)

# Differenza tra slice
diff = torch.abs(original_slice.flatten() - reshaped_slice.flatten())
print("Max difference between slices:", torch.max(diff))

# Somma parziale
original_partial_sum = torch.cumsum(original_slice.flatten(), dim=0)
reshaped_partial_sum = torch.cumsum(reshaped_slice.flatten(), dim=0)
print("Original partial sum (last):", original_partial_sum[-1])
print("Reshaped partial sum (last):", reshaped_partial_sum[-1])

# Somma manuale
manual_sum_original = 0.0
for val in original_slice.flatten():
    manual_sum_original += val.item()

manual_sum_reshaped = 0.0
for val in reshaped_slice.flatten():
    manual_sum_reshaped += val.item()

print(f"Manual sum original: {manual_sum_original}")
print(f"Manual sum reshaped: {manual_sum_reshaped}")
