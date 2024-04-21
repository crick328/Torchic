#import replica.nn
#import replica.optim
#from replica.tensor13 import Tensor
from replica.cuda_tensor import CudaTensor
from replica.helper import create_ohe_table, get_classes, labels_to_OHE


#dc = DeviceContext("cpu")