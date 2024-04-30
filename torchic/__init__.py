#import replica.nn
#import replica.optim
#from replica.tensor13 import Tensor
from torchic.cuda_tensor import Tensor
from torchic.helper import create_ohe_table, get_classes, labels_to_OHE


#dc = DeviceContext("cpu")