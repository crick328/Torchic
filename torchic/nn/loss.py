from torchic.cuda_tensor import Tensor
from torchic.nn import Softmax, LogSoftmax


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
        #print(x.data.dtype)
        loss = Tensor(x._device.power(x.data - target.data, 2))
        #print(x._device.power(x.data - target.data, 2).dtype)
        # il config e' giusto, ma questo tensore si istanzia come float32...
        #print(loss.data.dtype)
        #print(DEFAULT['dtype'])
        
        def _SELoss_backward(gradient: Tensor) -> None:
            # non aggiorno il grad di loss perché la derivata di loss w.r.t. loss è 1
            x.grad_fn(2*(x.data - target.data) * gradient)
            # rilascio il grafo computazionale in modo tale che il garbage collector possa cancellare le variabili inutili
            x.grad_fn = None

        # scritto per uniformità e flessibilità, si può valutare di rimuoverlo
        if x.requires_grad:
            loss.grad_fn=_SELoss_backward
        return loss


class NegativeLogLikehoodLoss:
    '''
    Implementazione alternativa (naive) di (categorical) `torchic.nn.CrossEntropyLoss`.
    Questa variante assume che `x_pred` siano  una serie di probabilita'
    che si sommano ad uno (softmax & co.).
    Possono presentarsi dei rounding errors se si usa la softmax e la derivata e'
    piu' lenta rispetto a quella riscritta ad hoc della cross entropy.
    Si usi questa loss se si desidera scegliere il tipo di softmax da usare e se
    si vuole che faccia parte dei layer del modello, altrimenti optare per la cross entropy. 
    '''
    def __init__(self):
        pass

    def __call__(self, x_pred: Tensor, y_trg: Tensor, eps: int=1e-10) -> Tensor:
        '''
        pred_p: output tensor of model containing all out values
        eps: small offset value to avoid log(0)

        returns: scalar tensor
        '''
        loss = -Tensor(x_pred.data * y_trg._device.log(y_trg.data + eps)).sum()#.mean()

        def _NLLLoss_backward(cr_grad):
            x_pred._run_backward(
                # d/dx(y*log(x + eps)) = y * d/dx(log(x + eps)) = y * 1/(x + eps)
                self_grad=-y_trg.data/(x_pred.data + eps) * cr_grad
            )

        if x_pred.requires_grad:
            loss.grad_fn = _NLLLoss_backward
        return loss


class CrossEntropyLoss:
    '''
    Loss function che e' una combinazione ottimizzata di `torch.nn.LogSoftmax` e
    `torch.nn.NegativeLogLikehoodLoss`. Viene eseguita la log softmax sul tensore
    `x` e poi viene calcolata la loss. Per questo motivo, la funzione vuole i raw
    values dell'ultimo layer del modello (quindi senza softmax & co. come activation
    function finale).
    '''
    def __init__(self) -> None:
        self.log_soft = LogSoftmax()
        self.softmax = Softmax()

    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
        '''
        Returns a scalar `Tensor`.

        Args:
            x (Tensor): raw output (logits) of the model (1-dim only).
            target (Tensor): one hot encoded label vector of the correct class (1-dim only).
        '''
        lsoft_p = self.log_soft(x)
        #tmp = -Tensor(target_p.data * pred_p._device.log(soft_p.data + eps), True)
        loss = -Tensor(x.data * lsoft_p.data).sum()
        #loss = -Tensor(target_p.data * pred_p._device.log(tmp.data + eps)).sum()#.mean()
        #loss = -Tensor(pred_p._device.log(tmp.data[target_p.data.argmax()] + eps))

        def _CELoss_backward(cr_grad):
            #pred_p.grad_fn(-target_p.data/(soft_p.data + eps) * cr_grad)
            #pred_p.grad_fn = None
            x._run_backward(
                # self_grad= -target_p.data/(soft_p.data + eps) * cr_grad
                self_grad=(self.softmax(x)-target).data * cr_grad
            )

        if x.requires_grad:
            loss.grad_fn =  _CELoss_backward
        return loss