from torchic.cuda_tensor import Tensor
import numpy as np
import cupy as cp


# posso fare in modo che self._xp sia definito in base al tensor che riceve?
# Potrei assegnarlo ad ogni call, ma sarebbe molto inefficiente ed overkill.
# Potrei metterlo come arg, ma sarebbe fragile e controintuitivo

# Bisogna aggiustare la loss function. Non può essere SE, ma così:
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
class SELoss:
    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
        # per ora è solo una potenza, basta solo fare il return dell'operazione con **
        '''
        Squared Error
        Input:
        - x: Tensor object containing the output of the network (predicted)
        - target: Tensor.data object containing the desired output values
        '''
        #target = Tensor(target) if type(target) != Tensor else target
        loss = Tensor(x._device.power(x.data - target.data, 2))
        
        def _SELoss_backward(gradient: Tensor) -> None:
            # non aggiorno il grad di loss perché la derivata di loss w.r.t. loss è 1
            x.grad_fn(2*(x.data - target.data) * gradient)
            # rilascio il grafo computazionale in modo tale che il garbage collector possa cancellare le variabili inutili
            x.grad_fn = None

        # scritto per uniformità e flessibilità, si può valutare di rimuoverlo
        if x.requires_grad:
            loss.grad_fn=_SELoss_backward
        return loss