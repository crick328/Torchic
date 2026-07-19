# Tutte le ops che usano gli indici (Slicing, indexing, masking, concatenazioni)
import pytest
from utils import create_random_tensors, backward_and_compare


# ==== PERMUTE ====
@pytest.mark.parametrize("shape, target", [
    ((2,3,4,5), (0,3,1,2))
])
def test_permute(shape, target):
    torch_t, my_t = create_random_tensors(shape)

    torch_res = torch_t.permute(target)
    my_res = my_t.permute(*target)

    backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )