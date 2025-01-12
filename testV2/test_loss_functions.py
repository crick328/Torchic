from preset import torchic, compare_tensors, create_random_tensors, backward_and_compare
from torch import zeros, nn
from torchic import randint, int64
import torch


# ===== MSELoss =====
torch_mse = nn.MSELoss()
my_mse = torchic.nn.MSELoss()

torch_t, my_t = create_random_tensors((10,))
torch_trg, my_trg = zeros((10,)), torchic.zeros((10,))

torch_loss = torch_mse(torch_t, torch_trg)
my_loss = my_mse(my_t, my_trg).mean()

backward_and_compare(
    [torch_loss, torch_t],
    [my_loss, my_t]
    )


# ===== CrossEntropyLoss =====
PAD = 1
LABEL_SMOOTHING = 0.1
BATCH = 2
SEQ_LEN = 8
VOCAB_SIZE = 20
torch_CCL = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=LABEL_SMOOTHING)
my_CCL = torchic.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=LABEL_SMOOTHING)

torch_t, my_t = create_random_tensors((BATCH, SEQ_LEN, VOCAB_SIZE))
my_trg = randint((BATCH, SEQ_LEN), high=VOCAB_SIZE, dtype=int64)
torch_trg = torch.from_numpy(my_trg.data)

torch_loss = torch_CCL(torch_t.view(-1, VOCAB_SIZE), torch_trg.view(-1))
my_loss = my_CCL(my_t.view(-1, VOCAB_SIZE), my_trg.view(-1))

backward_and_compare(
    [torch_loss, torch_t],
    [my_loss, my_t]
    )