# Riduzioni (sum, mean, max, min, std)

import pytest
from utils import create_random_tensors, backward_and_compare


# ===== MEAN =====
@pytest.mark.parametrize("dim", (0, 1, 2, -1))
@pytest.mark.parametrize("shape", [
    (2, 3, 4), (10, 3, 4, 2, 6)
])
def test_mean(shape, dim):
    torch_t, my_t = create_random_tensors(shape)

    torch_res = torch_t.mean(dim)
    my_res = my_t.mean(dim)

    backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )
    

# ==== STD ====
@pytest.mark.parametrize("dim", (0, 1, 2, -1))
@pytest.mark.parametrize("keepdim", (True, False))
@pytest.mark.parametrize("shape", [
    (2, 3, 4), (10, 3, 4, 2, 6)
])
def test_std(shape, dim, keepdim):
    # test delle dimensioni
    torch_t, my_t = create_random_tensors(shape)

    # meglio specificare i params perche' PyTorch puo'
    # averli in ordine diverso (come in questo caso)
    torch_res = torch_t.std(dim, keepdim=keepdim)
    my_res = my_t.std(dim, keepdim=keepdim)

    backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )
