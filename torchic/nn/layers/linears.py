from torchic.cuda_tensor import Tensor
from torchic.autograd.functions import EmbeddingBackward
import torchic.nn.initializers as inits
from torchic.nn.module import Module
from torchic.tools.helpers import zeros
from torchic.autograd.grad_context import no_grad
from torchic.amp.amp_stability import autocast_operation


# Linear Layer (corrisponde al Dense layer di Tensorflow se quest'ultimo è settato con activation=None)
# sara' la classe da cui tutti i layer di tipo linear erediteranno

# NOTE: al momento tutta la logica dei _weights e _bias e' inutilizzata perche' prima 
# istanzi la classe e poi fai linear.weights = array. Scegli quale metodo preferisci
# e cancella l'escluso. Secondo me quello attuale e' piu' flessibile (supporta
# caricamenti separati e unici insieme al modello) ed e' sostanzialmente la stessa cosa.
# O istanzi e assegni tramite args o prima istanzi e poi assegni, ma le tempistiche
# sono quelle...
# NOTE: serve type hint in modo che si capisca che le init restituiscono Tensor!
class Linear(Module):
    __slots__ = ("has_bias", "weights", "bias")

    def __init__(self, n_input: int=None, n_output: int=None, bias=True, _init_funct=inits.xavier_norm, **_init_kwargs):
        '''
        n_input: number of values received
        n_output: number of values returned
        _weights: sets the value of the weight matrix. Used when loading a model and it's not intended
            for user usage.
        _bias: sets the value of the bias vector. Used when loading a model and it's not intended
            for user usage.
        '''
        if n_input < 1 or n_output < 1:
            raise ValueError(
                "n_input and n_output have to be strictly positive, received"
                f"n_input={n_input} and n_output={n_output} instead."
                )

        self.has_bias = bias
        self.weights = _init_funct(n_input, n_output, **_init_kwargs)
        self.bias = zeros((n_output,), requires_grad=True) if bias else None

    #@autocast_operation
    def __call__(self, x: Tensor) -> Tensor:
        x = x.matmul(self.weights)
        # NOTE: l'eventuale broadcasting del bias viene gestito in autonomia dalle lib
        #return x + self.bias if self.has_bias else x
        return x if not self.has_bias else x + self.bias

    # metodo chiamato con print()
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        shape = self.weights.shape()
        return f"{self.__class__.__name__}(n_input={shape[0]}, n_output={shape[1]}, bias={self.has_bias})"
        

class Embedding(Linear):
    '''
    Contiene una singola matrice di dimensioni `(vocab_size, embedding_size)`
    che viene usata come look-up table: il vettore di input viene usato
    per accedere alle righe della matrice, che saranno restituite in
    output.
    Questo richiede che il vettore di input sia di interi e che i suoi valori
    siano essere compresi tra `[0, vocab_size)` per la stessa ragione. Puo'
    Avere lunghezza arbitraria.

    Esempio::

        >>> x = [1, 4]
        >>> emb = Embedding(5, 3)
        >>> emb.weights
        Tensor([
            [ 0.0587, -0.1269,  0.4763],
            [ 1.1524, -0.369 , -0.1064],
            [-0.0346,  0.4126,  0.3307],
            [-0.4452,  0.2931,  0.039 ],
            [ 0.9947,  0.6848,  1.0199]
        ],requires_grad=True)
        >>> emb(x)
        Tensor([
            [ 1.1524, -0.369 , -0.1064],
            [ 0.9947,  0.6848,  1.0199]
        ], grad_fn=EmbeddingBackward)
    '''
    # non necessario perche' utilizza quella di Linear
    # Si, usa quelli di linear, ma quindi ha slots non usati!
    __slots__ = ("weights",)

    def __init__(self, vocab_size: int, embedding_size: int, *args, **kwargs):
        # questa matrix pare che giovi sempre dell'init della Xavier norm
        super().__init__(vocab_size, embedding_size, *args, bias=False, **kwargs)
        #super().__init__(embedding_size, vocab_size, *args, bias=False, **kwargs)
    
    def __call__(self, x: Tensor) -> Tensor:
        '''
        Args:
            x (Tensor): integer 1D tensor of shape (1, N)

        Returns:
            Tensor of shape (N, embedding_size)
        '''
        # NOTE: potrei usaree torch.no_grad(), ma ha solo piu' overhead.
        # Usando .data sono in linea con le intenzioni (non voglio backward)
        # e posso dare la mia
        #with no_grad():
        #   output = self.weights[x.data]
        output = Tensor(self.weights.data[x.data])

        # NOTE: non riesco ad implementarla in modo pulito. Aggiunge
        # la grad_fn solamente se hai parents 
        output.register_backward_if_needed(EmbeddingBackward, x, self.weights)

        # x non puo' richiedere il grad perche' e' un tensore di interi
        #if self.weights.requires_grad:
            # rimuovo IndexBackward, che in questo caso non serve
            #output._grad_fn = None
            #output.grad_fn = EmbeddingBackward(x, self.weights)
        return output
    

class Dropout(Module):
    '''
    Setta a 0 randomicamente alcuni valori dell'input secondo la probabilità data in input.
    Sono attivi solo durante la fase di training (cosa che devi implementare).
    Le operazioni sono svolte inplace ma supportano backward corretta

    Args:
        p (float): probability of an element to be zeroed. It has to be
                 in range [0, 1]
    '''
    __slots__ = ("p", "training", "scale")

    def __init__(self, p: float=0.5):
        self.p = p
        # NOTE: nome importante per essere disattivato con .eval()
        self.training = True
        self.scale = Tensor(1.0 / (1.0 - self.p))

    def __call__(self, x):
        if self.training:
            # Bernoulli (bilinear) distribution
            mask = Tensor(x._device.random.rand(*x.shape()) > self.p)
            # "inverted dropout"
            x *= mask
            x *= self.scale
        return x