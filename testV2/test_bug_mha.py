import numpy as np
import cupy as cp



xp = cp

# Input simulato
batch_size, seq_len, d_model, head_num = 2, 4, 8, 2
qkv = xp.random.randn(batch_size, seq_len, 3 * d_model)

# Prima divisione
original_sum = xp.sum(qkv[:, :, :d_model])
og_qkv = qkv

# Reshape e split
head_dim = d_model // head_num
print("Original is contiguous:", qkv.flags['C_CONTIGUOUS'])
reshaped = qkv.reshape(batch_size, seq_len, head_num, 3 * head_dim)
print("Reshaped is contiguous:", reshaped.flags['C_CONTIGUOUS'])
print(og_qkv.sum(), reshaped.sum())


split_sum = xp.sum(reshaped[:, :, :, :head_dim])
print(og_qkv.sum(), reshaped.sum())


# questi fanno piu' o meno cagare
print("Original sum:", original_sum)
print("Split sum:", split_sum)

# questi sono perfetti
orig_sum_flat = xp.sum(qkv.flatten())
reshaped_flat = xp.sum(reshaped.flatten())
print("Original sum (flattened):", orig_sum_flat)
print("Reshaped sum (flattened):", reshaped_flat)

# questi fanno di nuovo cagare
original_slice = qkv[:, :, :d_model]
reshaped_slice = reshaped[:, :, :, :head_dim]
original_slice = original_slice.copy()  # Force a deep copy
reshaped_slice = reshaped_slice.copy()  # Force a deep copy
print(f"Original stride: {original_slice.strides}")
print(f"Reshaped stride: {reshaped_slice.strides}")
print("Original slice sum (manual):", xp.sum(original_slice.ravel()))
print("Reshaped slice sum (manual):", xp.sum(reshaped_slice.ravel()))

# fa cagare anche questa
original_mean = np.mean(qkv[:, :, :d_model])
reshaped_mean = np.mean(reshaped[:, :, :, :head_dim])
print("Original mean:", original_mean)
print("Reshaped mean:", reshaped_mean)

# schifo
diff = np.abs(original_slice.ravel() - reshaped_slice.ravel())
print("Max difference between slices:", np.max(diff))

# differente
original_partial_sum = np.cumsum(original_slice.ravel())
reshaped_partial_sum = np.cumsum(reshaped_slice.ravel())
print("Original partial sum (last):", original_partial_sum[-1])
print("Reshaped partial sum (last):", reshaped_partial_sum[-1])

# Esegui la somma manuale (simulando l'operazione che sum() fa internamente)
# Nada
manual_sum_original = 0.0
for val in original_slice.ravel():
    manual_sum_original += val

manual_sum_reshaped = 0.0
for val in reshaped_slice.ravel():
    manual_sum_reshaped += val

print(f"Manual sum original: {manual_sum_original}")
print(f"Manual sum reshaped: {manual_sum_reshaped}")