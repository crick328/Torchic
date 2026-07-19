import sys
from preset import torchic
from torchic import Tensor
from torchic.tools.code_analysis.mem import get_nbytes
import numpy as np
from torchic.nn import ModuleList, Linear

a = ModuleList(Linear(1,1,False))


print(get_nbytes(a))
print(get_nbytes(Linear(1,1,False)))
print(np.array([[1]]).nbytes)


