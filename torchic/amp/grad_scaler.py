# Gradient scaling is a methon employed to avoid underflow.
# When using mixed precisions, the elegible forward operations


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor


class GradScaler:
    '''
    Gradient scaling is a method employed to avoid underflow.
    When using mixed precision, the elegible forward operations are done in float16
    or bfloat16. This implies that
    '''
    def __init__(self, scale_factor: int):
        self.s = scale_factor
        self.scale_correction = 2   # richiede logica per init e andamento
        self.overflow_detected = False
        # non ho flag per l'underflow perche' cerco gia'
        # di prevenirlo con il rescaling

    def scale(self, loss: 'Tensor') -> 'Tensor':
        return loss*self.s
    
    def step(self, optim):
        # effettuo un rescaling prima di aggiornare i gradienti accumulati
        for param in optim.params:
            if not param._device.isfinite(param._grad):
                self.overflow_detected = True
                break

            param._grad /= self.s
        
        if not self.overflow_detected:
            optim.step()
        else:
            # devo scartare tutti i gradienti della batch
            # se overflow, divido s
            pass

    def update(self, optim):
        if not self.overflow_detected:
            optim.update()