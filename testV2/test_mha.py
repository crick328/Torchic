import torch
import numpy as np


D_MODEL = 4
BATCH_SIZE = 2
SEQ_LEN = 4
HEAD_NUM = 2

np_qkv_proj = np.random.rand(BATCH_SIZE, SEQ_LEN, 3*D_MODEL)
torch_qkv_proj = torch.from_numpy(np_qkv_proj)

og_np_qkv = np_qkv_proj
og_torch_qkv = torch_qkv_proj

head_dim = 3*D_MODEL // HEAD_NUM

np_qkv = np_qkv_proj.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, head_dim).swapaxes(1, 2)
t_qkv = torch_qkv_proj.view(BATCH_SIZE, SEQ_LEN, HEAD_NUM, head_dim).transpose(1, 2)

print(og_np_qkv.sum(), np_qkv.sum())
print(og_torch_qkv.sum(), torch_qkv_proj.sum())
