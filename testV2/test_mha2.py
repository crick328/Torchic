import numpy as np


D_MODEL = 4
BATCH_SIZE = 2
SEQ_LEN = 4
HEAD_NUM = 2

qkv_proj = np.random.rand(BATCH_SIZE, SEQ_LEN, 3*D_MODEL)
q_proj = qkv_proj[:, :, :D_MODEL]

og_qkv_proj = qkv_proj
og_q_proj = q_proj

head_dim = D_MODEL // HEAD_NUM

qkv = qkv_proj.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, 3*head_dim).swapaxes(1, 2)
q = q_proj.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, head_dim).swapaxes(1, 2)

print(f"q rimane invariato? {qkv[:, :, :, :head_dim].sum() == q.sum()}: {qkv[:, :, :, :head_dim].sum()}, {q.sum()}")
print("Il reshape ha modificato i tensori?")
print(f"-> qkv: {og_qkv_proj.sum() != qkv.sum()}")
print(f"-> q: {og_q_proj.sum() != q.sum()}")
