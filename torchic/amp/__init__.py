'''
Automatic Mixed Precision Module.

'''


from torchic.amp.amp_stability import autocast_operation
#from torchic.amp.grad_scaler import GradScaler
from torchic.amp.autocast_ctx import autocast