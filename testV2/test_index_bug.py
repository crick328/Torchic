import numpy as np
import cupy as cp
from preset import torchic
from torchic import Tensor, ones

BATCH_SIZE = 2
MAX_LEN = 8
PAD_TOKEN = Tensor(1)


# creo gli input
np_enc_input = []
cp_enc_input = []
for i in range(0, BATCH_SIZE):
    data = np.random.rand(MAX_LEN-3,)
    np_enc_input.append(Tensor(data, device="cpu"))
    cp_enc_input.append(Tensor(data, device="cuda"))

# creo le maschere di [PAD]
np_pad_arr = ones((BATCH_SIZE, MAX_LEN)) * PAD_TOKEN
cp_pad_arr = ones((BATCH_SIZE, MAX_LEN), device="cuda") * PAD_TOKEN.to("cuda")

for i, (np_enc_seq, cp_enc_seq) in enumerate(zip(np_enc_input, cp_enc_input)):
    np_pad_arr[0, :len(np_enc_seq)] = np_enc_seq
    cp_pad_arr[0, :len(cp_enc_seq)] = cp_enc_seq

print(np_pad_arr[0])
print(cp_pad_arr[0])



'''
# creo gli input
np_enc_input = []
cp_enc_input = []
for i in range(0, BATCH_SIZE):
    data = np.random.rand(MAX_LEN-3,)
    np_enc_input.append(data)
    cp_enc_input.append(cp.array(data))

# creo le maschere di [PAD]
np_pad_arr = np.ones((BATCH_SIZE, MAX_LEN)) * PAD_TOKEN
cp_pad_arr = cp.ones((BATCH_SIZE, MAX_LEN)) * PAD_TOKEN

# Test
np_pad_arr[0, :len(np_enc_input[i])] = np_enc_input[i]
cp_pad_arr[0, :len(cp_enc_input[i])] = cp_enc_input[i]

print(np_pad_arr[0])
print(cp_pad_arr[0])
'''