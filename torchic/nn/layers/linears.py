from torchic.cuda_tensor import Tensor
from torchic.backward import EmbeddingBackward
import torchic.nn.initializers as inits
from torchic.nn.module import Module


# Linear Layer (corrisponde al Dense layer di Tensorflow se quest'ultimo è settato con activation=None)
# sara' la classe da cui tutti i layer di tipo linear erediteranno

# NOTE: al momento tutta la logica dei _weights e _bias e' inutilizzata perche' prima 
# istanzi la classe e poi fai linear.weights = array. Scegli quale metodo preferisci
# e cancella l'escluso. Secondo me quello attuale e' piu' flessibile (supporta
# caricamenti separati e unici insieme al modello) ed e' sostanzialmente la stessa cosa.
# O istanzi e assegni tramite args o prima istanzi e poi assegni, ma le tempistiche
# sono quelle...
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
        if bias:
            self.bias = _init_funct(n_input, n_output, is_bias=True, **_init_kwargs)
    
    # si potrebbe anche fare in modo che la logica sia scritta nel .forward()
    # in modo tale che si sovrascrive solo quello quando si eredita e di
    # default il __call__lo chiama (che e' cio' che hai fatto con Sequential)
    def forward(self, x: Tensor) -> Tensor:
        x = x.matmul(self.weights)
        
        return x + self.bias if self.has_bias else x
    
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

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
        #self.weights = inits.xavier_norm(vocab_size, embedding_size) if _weights is None else _weights
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x (Tensor): integer 1D tensor of shape (1, N)

        Returns:
            Tensor of shape (N, embedding_size)
        '''
        # NOTE: sarebbe carino trovare un modo per dire che solo questa op
        # non deve generare backward, tipo torch.no_grad()
        output = self.weights[x.data]

        # x non puo' richiedere il grad perche' deve essere un intero
        if self.weights.requires_grad:
            # rimuovo IndexBackward, che in questo caso non serve
            output._grad_fn = None
            output.grad_fn = EmbeddingBackward(self.weights, x)
        return output
    

class Dropout(Module):
    '''
    Setta a 0 randomicamente alcuni valori dell'input secondo la probabilità data in input.
    Sono attivi solo durante la fase di training (cosa che devi implementare).

    probability: valore compreso tra (0, 1). Corrisponde alla percentuale di valori
                 settati a zero.
    '''
    __slots__ = ("p", "training")

    def __init__(self, probability: float=0.5):
        self.p = probability
        self.training = True

    def forward(self, x):
        if self.training:
            # Bernoulli (bilinear) distribution
            mask = Tensor(x._device.random.rand(*x.shape()) > self.p)
            # "inverted dropout"
            scale = Tensor(1.0 / (1.0 - self.p))
            x *= mask
            x *= scale
        return x
        
    def eval(self):
        self.training = False

    def train(self):
        self.training = True