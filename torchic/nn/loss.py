from torchic.cuda_tensor import Tensor
from torchic.nn.activations import Softmax, LogSoftmax
import torchic.autograd.functions as back_op
from torchic.dtypes import int32, int64
from torchic.autograd.grad_context import no_grad
from torchic.amp.amp_stability import autocast_operation

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

        loss.register_backward_if_needed(back_op.SELossBackward, x, target.data)
        
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

        loss.register_backward_if_needed(back_op.NLLLossBackward, x, target, eps)
        return loss


class CrossEntropyLoss:
    __slots__ = ("log_soft", "softmax", "reduction",
                 "ignore_index", "eps")

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
                Target has to be an array of integer containing the class labels,
                NOT one hot encoded.
                Default is `-100` (which means not applied).
            label_smoothing (float, optional): redistributes the target probabilities
                to lower the correct class value and increase the other classes.
                The epsilon removed from the correct class is evenly distributed
                to the others.
                It reduces the model confidence, (hopefully) allowing for
                better generalization and less overfitting.
                value can be `[0.0, 1.0]`. Default is `0.0`.
        '''
        self.log_soft = LogSoftmax(-1)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.eps = label_smoothing

    #@autocast_operation
    def __call__(self, x: Tensor, target: Tensor) -> Tensor:
        '''
        This version accepts only a `target` which is an integer
        `Tensor` of labels. One hot encoding is no longed supported
        because it complicates the backward logic (and also the forward)
        for nothing: they occupy a lot of memory and you can't even
        use the `ignore_index` param.
        If you need OHE support, use the legacy version with `.funzione()`
        '''
        # questa NON puo' generare backprop, altrimenti creo un'altra
        # reference alla backward di x, creando quindi un ref_count
        # di 2 che non verra' MAI esaurito! In piu' perdi memoria.
        # Potrei fare copia-incolla delle due righe della softmax, ma
        # non mi piace per niente come idea. La logica deve essere
        # centralizzata
        #print(x.dtype)
        with no_grad():
            #print(self.log_soft.__call__)
            log_probs = self.log_soft(x).data
            #print(log_probs.dtype)

        # se l'utente vuole azzerare il gradiente associato ad una classe
        if self.ignore_index != -100:
            # gli indici (labels delle classi) devono essere interi
            if not target.dtype.kind != 'i':
                raise RuntimeError(
                    f"'ignore_index' supports only integer target dtype, "
                    f"received {target.dtype} dtype instead "
                    "(Note that you have to specify the dtype when creating "
                    "the target tensor or else will be casted to the default "
                    "dtype of the library)."
                )
            
            mask = Tensor(target.data != self.ignore_index, device=target.device, dtype=int32)
            # sostituisco gli indici da ignorare con 0 per evitare 
            # IndexErrors o errori silenti (dovuti all'autogestione di np/cp)
            # durante l'indexing su log_prob_x
            # occhio: qua puo' partire type promotion se passi dei target
            # float per svista
            safe_target = target.data * mask.data
        else:
            mask=None
            safe_target = target.data
        
        # prendo le log-probs del modello associate alla classe corretta
        batch_indices = x._device.arange(len(target.data))
        nll_loss = -log_probs[batch_indices, safe_target]

        # termine di smoothing (media su tutte le classi)
        if self.eps > 0.0:
            #smooth_loss = -log_probs.mean(dim=-1)
            smooth_loss = -x._device.mean(log_probs, axis=-1)
            loss = Tensor(nll_loss*(1 - self.eps) + self.eps*smooth_loss)
        else:
            loss = Tensor(nll_loss)

        if mask is not None:
            loss *= mask


        loss.register_backward_if_needed(
            back_op.CrossEntropyLossBackward,
            x, log_probs, safe_target,
            mask.data if mask is not None else None, self.eps
        )

        match self.reduction:
            case "none":
                pass
            case "sum":
                loss = loss.sum()
            case "mean":
                if self.ignore_index != -100:
                    loss = loss.sum() / mask.sum()
                else:
                    loss = loss.mean()
            case _:
                raise ValueError(
                    f"Received {self.reduction} as reduction value, but only "
                    "'none', 'sum' or 'mean' is valid."
                    )

        return loss

    def __full_call__(self, x: Tensor, target: Tensor) -> Tensor:
            num_classes = x.size(-1)
            mask = None
            log_probs = self.log_soft(x).data
    
            # se il target sono OHE
            if x.data.ndim == target.data.ndim:
                # tutte le classi ricevono un +eps/n_classes, mentre solo la classe target
                # riceve anche un termine +(1-eps)
                smoothed_target = target * Tensor(1.0 - self.eps) + Tensor(self.eps / num_classes)
                loss = -(smoothed_target * log_probs).sum(-1)
            # se il target e' un vettore di labels (quindi indici interi)
            else:
                # se l'utente vuole azzerare il gradiente associato ad una classe
                if self.ignore_index != -100:
                    # gli indici (labels delle classi) devono essere interi
                    if not target._device.issubdtype(target.dtype, target._device.integer):
                        raise RuntimeError(
                            f"'ignore_index' supports only integer target dtype, "
                            f"received {target.dtype} dtype instead "
                            "(Note that you have to specify the dtype when creating "
                            "the target tensor or else will be casted to the default "
                            "dtype of the library)."
                        )
                    mask = Tensor(target.data != self.ignore_index, device=target.device, dtype=int64)
                    # sostituisco gli indici da ignorare con 0 per evitare 
                    # IndexErrors o errori silenti (dovuti all'autogestione di np/cp)
                    # durante l'indexing su log_prob_x
                    # occhio: qua puo' partire type promotion se passi dei target
                    # float per svista
                    safe_target = target.data * mask.data
                else:
                    safe_target = target.data
                
                # prendo le log-probs del modello associate alla classe corretta
                batch_indices = x._device.arange(len(target.data))
                print(batch_indices.dtype, safe_target.dtype)
                nll_loss = -log_probs[batch_indices, safe_target]
    
                # termine di smoothing (media su tutte le classi)
                if self.eps > 0.0:
                    #smooth_loss = -log_probs.mean(dim=-1)
                    smooth_loss = -x._device.mean(log_probs, axis=-1)
                    loss = Tensor(nll_loss*(1 - self.eps) + self.eps*smooth_loss)
                else:
                    loss = Tensor(nll_loss)
    
                if mask is not None:
                    loss *= mask
    
    
            loss.register_backward_if_needed(
                back_op.CrossEntropyLossBackward,
                x, log_probs, target.data, mask.data if mask is not None else None, self.eps
            )
    
            match self.reduction:
                case "none":
                    pass
                case "sum":
                    loss = loss.sum()
                case "mean":
                    if self.ignore_index != -100:
                        loss = loss.sum() / mask.sum()
                    else:
                        loss = loss.mean()
                case _:
                    raise ValueError(
                        f"Received {self.reduction} as reduction value, but only "
                        "'none', 'sum' or 'mean' is valid."
                        )
    
            return loss

    def __ecall__(self, x: Tensor, target: Tensor) -> Tensor:
        num_classes = x.size(-1)
        log_prob_x = self.log_soft(x)

        # se il target sono OHE
        if x.data.ndim == target.data.ndim:
            # tutte le classi ricevono un +eps/n_classes, mentre solo la classe target
            # riceve anche un termine +(1-eps)
            smoothed_target = target * Tensor(1.0 - self.eps) + Tensor(self.eps / num_classes)
            loss = -(smoothed_target * log_prob_x).sum(-1)
        # se il target e' un vettore di labels (quindi indici interi)
        else:
            # se l'utente vuole azzerare il gradiente associato ad una classe
            if self.ignore_index != -100:
                # gli indici (labels delle classi) devono essere interi
                if not target._device.issubdtype(target.dtype, target._device.integer):
                    raise RuntimeError(
                        f"'ignore_index' supports only integer target dtype, "
                        f"received {target.dtype} dtype instead "
                        "(Note that you have to specify the dtype when creating "
                        "the target tensor or else will be casted to the default "
                        "dtype of the library)."
                    )
                mask = Tensor(target.data != self.ignore_index, device=target.device, dtype=int64)
                # sostituisco gli indici da ignorare con 0 per evitare 
                # IndexErrors o errori silenti (dovuti all'autogestione di np/cp)
                # durante l'indexing su log_prob_x
                safe_target = target.data * mask.data
            else:
                mask = None
                safe_target = target.data

            # prendo le log-probs del modello associate alla classe corretta
            batch_indices = x._device.arange(len(target.data))
            nll_loss = -log_prob_x[batch_indices, safe_target]

            # termine di smoothing (media su tutte le classi)
            # if self.label_smoothing > 0.0:
            smooth_loss = -log_prob_x.mean(dim=-1)
            loss = nll_loss*Tensor(1 - self.eps) + self.label_smoothing*smooth_loss
            #else:
                #loss = nll_loss

            if mask is not None:
                loss *= mask

        loss.register_backward_if_needed(x, )

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
        return loss

    def __ogcall__(self, x: Tensor, target: Tensor) -> Tensor:
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
            loss = -(smoothed_targets * log_prob_x).sum(-1)
        else:
            # Questo funziona per prediction 2D
            one_hot_targets = x._device.eye(num_classes)[target.data]
            # perche' non da' IndexError? Perche' numpy gestisce
            # silenziosamente arr[-100] partendo dal fondo! Potrebbe
            # causare bug silenti! BRUZZOH!
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
            # SUS!
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