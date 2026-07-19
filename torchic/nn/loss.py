from torchic.cuda_tensor import Tensor
from torchic.nn.activations import Softmax, LogSoftmax
import torchic.autograd.functions as back_op
from torchic.autograd.functions import register_backward_if_needed
from torchic.dtypes import int64

# check derivata


# posso fare in modo che self._xp sia definito in base al tensor che riceve?
# Potrei assegnarlo ad ogni call, ma sarebbe molto inefficiente ed overkill.
# Potrei metterlo come arg, ma sarebbe fragile e controintuitivo


class LossFunction:
    pass


class MSELoss:
    __slots__ = ("_reduction")

    def __init__(self, reduction: str="mean") -> None:
        self._check_reduction_value(reduction)
        self._reduction = reduction

    @property
    def reduction(self) -> str:
        return self._reduction
    
    @reduction.setter
    def reduction(self, value: str) -> None:
        self._check_reduction_value(value)
        self._reduction = value

    def _check_reduction_value(self, value):
        if value not in ("none", "sum", "mean"):
            raise ValueError(
                "Reduction can be 'none', 'sum' or 'mean', received "
                f"{value} instead."
            )

    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
        '''
        Squared Error
        Input:
        - x: output of the network (predicted)
        - target: Tensor.data object containing the desired output values
        '''
        loss = Tensor(x._device.power(x.data - target.data, 2))

        register_backward_if_needed(loss, back_op.SELossBackward(x, target.data))
        
        # per compatibilita' con PyTorch
        match self.reduction:
            case "mean":
                loss = loss.mean()
            case "sum":
                loss = loss.sum()
            case "none":
                pass
            case _:
                raise RuntimeError(
                    "Why did you changed the ._reduction attribute? :)"
                )
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
    def __call__(self, x: Tensor, target: Tensor, eps: int=1e-10) -> Tensor:
        '''
        pred_p: output tensor of model containing all out values
        eps: small offset value to avoid log(0)

        returns: scalar tensor
        '''
        loss = -Tensor(x.data * target._device.log(target.data + eps)).sum()#.mean()

        def _NLLLoss_backward(cr_grad):
            x._run_backward(
                # d/dx(y*log(x + eps)) = y * d/dx(log(x + eps)) = y * 1/(x + eps)
                self_grad=-target.data/(x.data + eps) * cr_grad
            )

        register_backward_if_needed(loss, back_op.NLLLossBackward(x, target, eps))
        return loss


class CrossEntropyLoss:
    __slots__ = ("log_soft", "softmax", "reduction",
                 "ignore_index", "label_smoothing")

    '''
    Combinazione ottimizzata di `torch.nn.LogSoftmax` e `torch.nn.NegativeLogLikehoodLoss`.
    Viene eseguita la (log) softmax sul tensore `x` e poi viene calcolata la loss. Per
    questo motivo, la funzione vuole i raw values dell'ultimo layer (logits) del modello
    (quindi senza softmax & co. come activation function finale).
    '''
    def __init__(self, reduction: str="mean", ignore_index: int=-100, label_smoothing: int=0.0) -> None:
        '''
        Args:
            reduction (str, optional): the reduction to apply on the final loss.
                Can be `none`, `sum` or `mean` (default value).
            ignore_index (int, optional): Specifies a target value (which should
                represent a class) that is ignored and does not contribute to the
                input gradient.
                Target has to be an array of integer containing the class labels.
                Default is `-100` (which means not applied).
            label_smoothing (float, optional): redistributes the target probabilities
                to lower the correct class value and increase the other classes.
                This reduce the model confidence, (hopefully) allowing for
                better generalization and less overfitting.
                value can be `[0.0, 1.0]`. Default is `0.0`.
        '''
        self.log_soft = LogSoftmax(-1)
        self.softmax = Softmax(-1)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
        '''
        La versione 2D da' per scontato che il *vettore* target ricevuto siano
        *le labels* delle classi giuste!
        La con x 1D vuole che target sia un vettore one hot encoded, la versione
        2D vuole che target sia sempre un vettore, percio', per contenere
        le risposte corrette di ogni riga di x, deve far si' che ogni label
        sia convertito nel suo corrispettivo OHE!
        Il [target.data] serve proprio per risistemare le righe della
        matrice identita' in modo che la riga della matrice sia la
        OHE del valore corrispondente di target!
        Es:
        target = [2, 1, 1]
        one_hot_targets = eye(3)[target]
           |0 0 1|   |2|
        => |0 1 0| = |1|
           |0 1 0|   |1|
        Questo spiega perche' la versione 2D richiede che target sia
        composto da interi: i suoi valori sono usati come indici! E
        grazie a loro che si ottiene la matrice "rimescolata"!
        Questo spiega anche perche' ignore_index e' accettato solo
        quando x e' 2D. Infatti, se x e' 1D, target e' un vettore
        one hot encoded, quindi l'unico valore ignorabile sarebbe
        1, che non ha senso!
        Infine, questo spiega perche', in caso di x bidimensionale,
        target non puo' avere un valore maggiore dell'ultima
        dimensione di x (cioe' delle singole prediction): rappresenterebbe
        una classe che non esiste sul lato logico, mentre sul pratico
        il codice si rompe perche' quel valore non puo' essere usato
        come index per "riordinare" la matrice identita', che ha
        perfettamente senso.

        NOTA: non so come si comporta per x.dim() == 3!
        '''
        #- se x e target hanno le stesse dim, target puo' essere
        #  di qualsiasi dtype
        #- se x.dim() == 2 && target.dim() == 1, il dtype di
        #  target deve essere int e deve rappresentare le labels non ohe
        

        #num_classes = x.data.shape[-1]
        num_classes = x.size(-1)
        
        if x.data.ndim == target.data.ndim:
            # Questo funziona per prediction 1D
            smoothed_targets = Tensor(target.data * (1 - self.label_smoothing) + self.label_smoothing / num_classes)
        else:
            # Questo funziona per prediction 2D
            one_hot_targets = x._device.eye(num_classes)[target.data]
            smoothed_targets = Tensor(one_hot_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes)

        log_prob_x = self.log_soft(x)
        loss = -(smoothed_targets * log_prob_x).sum(-1)
        
        if self.ignore_index != -100:
            if not target._device.issubdtype(target.dtype, target._device.integer):
                #target.data = Tensor._to_dtype(target.data, int64)
                raise RuntimeError(
                    f"'ignore_index' supports only integer target dtype, "
                    f"received {target.dtype} dtype instead "
                    "(Note that you have to specify the dtype when creating "
                    "the target tensor or else will be casted to the default "
                    "dtype of the library)."
                )
            mask = Tensor(target.data != self.ignore_index, device=target.device, dtype=int64)
            loss *= mask
        

        match self.reduction:
            case "none":
                pass
            case "sum":
                loss = loss.sum()
            case "mean":
                if self.ignore_index != -100:
                    loss = (loss * mask).sum() / mask.sum()
                else:
                    loss = loss.mean()
            case _:
                raise ValueError(
                    f"Received {self.reduction} as reduction value, but only "
                    "'none', 'sum' or 'mean' is valid."
                    )

        # eh, la backward ad hoc non sembra funzionare
        # con l'aggiunta dei vari parametri... Percio'
        # ho scritto tutta la __call__ con il framework
        # di autograd in modo che tutto sia backpropagabile
        #if x.requires_grad:
            #loss.grad_fn = back_op.CELossBackward(x, self.softmax(x), target)
        return loss