from typing import TYPE_CHECKING
from torchic.amp import wrappers
if TYPE_CHECKING:
    from torchic.cuda_tensor import Tensor


# https://github.com/pytorch/pytorch/blob/v2.13.0/torch/amp/grad_scaler.py


class GradScaler:
    '''
    Gradient scaling is a method employed to avoid underflow.
    When using mixed precision, the elegible forward operations are done in float16
    (or bfloat16), which has only 5 bit dedicated to the exponent. In this
    range, underflow are very common given the small magnitude of the gradients.

    To avoid this problem, GradScaler is introduced.
    At the start of the backpropagation it multiplies the grad (loss -> 
    original name was Loss Scaling) for a
    scale_factor (which usually is a big number). This shift propagates
    to all other gradients thanks to the products of the chain rule, so
    every Tensor is shifted by the same amount.

    Before calling optim.step(), gradients must be scaled back to
    their original magnitude (otherwise the optimizer would do enormous
    steps breaking everything).
    scaler.step(optim) checks for Nan/inf gradients. If detected,
    optim.step() is skipped and the scale_factor is adjusted accordingly
    with .update()
    (halved in case of overflow and doubled if there aren't any issues
    for a number of steps).

    How to use it:
    scaler = GradScaler()
    
    for batch in epoch:
        ...
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # updates scaling factor
        scaler.update()
        # the order of this one isn't important, just remember to call it
        optim.zero_grad()

    '''
    def __init__(self, scale_factor: int = 2.0**14):
        self.scale_factor = scale_factor  # da rendere Tensor
        self.overflow_detected = False
        self.growth_factor = 2.0
        self.backoff_factor = 0.5 # used when nan/inf are detected

        self.growth_interval = 2000 # every 2000 .step()s it will double scale_factor
        self._num_steps = 0

        # non ho flag per l'underflow perche' cerco gia'
        # di prevenirlo con il rescaling

    def scale(self, loss: 'Tensor') -> 'Tensor':
        # d/dL = 1*scale_factor => rescale of all backward gradients
        return loss*self.scale_factor
    
    def step(self, optim):
        # effettuo il rescaling inverso prima di aggiornare i gradienti accumulati

        # NOTE: QUA e' dove voglio resettare la amp_cache
        wrappers.amp_cache = {}

        for param in optim.params:
            if not param._device.isfinite(param._grad):
                self.overflow_detected = True
                break

            param._grad /= self.scale_factor
        
        # aggiorno solo se tutto ok, senno' skippo lo step e butto i grads
        if not self.overflow_detected:
            optim.step()
            self._num_steps += 1
        #va fatto fuori, come sempre
        #optim.zero_grad()

    def update(self):
        '''
        Updates the scale factor
        '''
        if self.overflow_detected:
            # riduco lo scale factor
            self.scale_factor *= self.backoff_factor
            # resetto variabili utils
            self._num_steps = 0
            self.overflow_detected = False
        
        elif self._num_steps == self.growth_interval:
            # aumento lo scale factor
            self.scale_factor *= self.growth_factor
            self._num_steps = 0

    def state_dict(self):
        # memo per dire che devo scrivere un modo per save/load e 
        # integrarlo in tutto il sistema
        pass

# pip install --upgrade pip setuptools wheel
# pip install ml_dtypes