import pytest
#from utils import torchic, create_random_tensors
from torchic.settings import CUPY_AVAILABLE, BRAIN_FLOAT_AVAILABLE, set_default_device
from torchic.amp.autocast_ctx import autocast
from torchic import rand, bfloat16


#@pytest.mark.parametrize
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="Flag CUPY_AVAILABLE è False")
def test_autocast_ctx():
    # NOTE: non basta definire entrambi i tensori su cuda per avere
    # un child su cuda perche' prevale il default_device e il figlio
    # viene messo su CPU
    set_default_device("cuda")

    a = rand((200, 200))
    b = rand((200, 200))

    with autocast():
        c = a @ b
        d = a @ c
    # if brain float isn't available, bfloat16 == cp.float16
    assert c.dtype == bfloat16
    assert d.dtype == bfloat16
    



