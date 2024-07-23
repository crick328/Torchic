from torchic.cuda_tensor import Tensor
import torchic.nn.initializers as inits


# Linear Layer (corrisponde al Dense layer di Tensorflow se quest'ultimo è settato con activation=None)
# sara' la classe da cui tutti i layer di tipo linear erediteranno
# (devo farlo con Embeddings)

# NOTE: al momento tutta la logica dei _weights e _bias e' inutilizzata perche' prima 
# istanzi la classe e poi fai linear.weights = array. Scegli quale metodo preferisci
# e cancella l'escluso. Secondo me quello attuale e' piu' flessibile (supporta
# caricamenti separati e unici insieme al modello) ed e' sostanzialmente la stessa cosa.
# O istanzi e assegni tramite args o prima istanzi e poi assegni, ma le tempistiche
# sono quelle...
class Linear:
    __slots__ = ("has_bias", "weights", "bias")

    def __init__(self, n_input: int=None, n_output: int=None, bias=True, _weights=None, _bias=None, _init_funct=inits.xavier_norm):
        '''
        n_input: number of values received
        n_output: number of values returned
        _weights: sets the value of the weight matrix. Used when loading a model and it's not intended
            for user usage.
        _bias: sets the value of the bias vector. Used when loading a model and it's not intended
            for user usage.
        '''
        self.__value_check(n_input, n_output, _weights, _bias)

        self.has_bias = bias
        self.weights = _init_funct(n_input, n_output) if _weights is None else _weights
        if bias:
            self.bias = _init_funct(n_input, n_output, is_bias=True) if _bias is None else _bias
    
    # si potrebbe anche fare in modo che la logica sia scritta nel .forward()
    # in modo tale che si sovrascrive solo quello quando si eredita e di
    # default il __call__lo chiama (che e' cio' che hai fatto con Sequential)
    def __call__(self, x: Tensor) -> Tensor:
        x = x.matmul(self.weights)
        return x + self.bias if self.has_bias else x
    
    def __value_check(self, n_input=None, n_output=None, weights=None, bias=None):
        if not weights and not bias:
            if n_input is None or n_output is None:
                raise ValueError("You have to pass a value for n_input and n_output.")
            
            if n_input < 1 or n_output < 1:
                raise ValueError(
                    "n_input and n_output have to be strictly positive, received"
                    f"n_input={n_input} and n_output={n_output} instead."
                    )
            
        elif weights and not bias:
            if weights.shape()[1] != bias.shape()[0]:
                raise RuntimeError(
                    "Given weights and bias have mismathed shapes. "
                    f"weights: {weights.shape()}, bias: {bias.shape()}."
                    )
    
    def __str__(self):
        # metodo chiamato con print()
        output = f"{self.__class__.__name__}(weights = {self.weights.shape()}"
        if self.has_bias:
            return output + f", bias = {self.bias.shape()})"
        else:
            return output + ")"

    def __repr__(self):
        shape = self.weights.shape()
        return f"{self.__class__.__name__}({shape[0]}, {shape[1]}, bias={self.has_bias})"
        
    def state_dict(self):
        out_dict = {"class": self.__class__.__name__,
                    "kwargs": {"_weights": self.weights.state_dict()}
                    }
        if self.has_bias:
            out_dict["kwargs"]["has_bias"] = True
            out_dict["kwargs"]["_bias"] = self.bias.state_dict()
        else:
            out_dict["kwargs"]["has_bias"] = False
        return out_dict
        

class Embedding(Linear):
    #__slots__ = ("weights")

    def __init__(self, vocab_size: int, embedding_size: int, _weights=None):
        '''
        Layer avente solo weight matrix e nessun bias. Funge da "lookup table", ovvero, dato in input
        un vettore codificato in one hot encoding (cioè un index), restituisce la rispettiva riga
        della matrice, senza fare calcoli.
        Per il resto, la sua struttura è equivalente a quella di un Linear Layer, con:
        vocab_size = n_input
        embedding_size = n_output

        In realtà, il funzionamento è equivalente a quello di un linear layer senza bias:
        se infatti facciamo la matmul di un vettore con un solo ingresso ad 1 e gli altri a 0
        per una matrice, il risultato sarà proprio l'unica riga moltiplicata per 1 anziché 0.
        Tuttavia, è molto più efficiente leggere direttamente l'index.

        Per la backprop, viene aggiornata una sola riga di weights alla volta, ovvero quella che viene usata.
        Infatti, essa verrà poi moltiplicata per la w_matrix del layer successivo che, una volta differenziata,
        creerà il grad per la riga dell'embedding layer. Un Linear aggiorna la w_matrix in un colpo solo,
        qui invece si creano diversi grad (che però si possono accumulare prima di aggiornare la matrice)

        vocab_size: number of values received
        embedding_size: number of values returned
        _weights: sets the value of the weight matrix. Used when loading a model and it's not intended
            for user usage.
        '''
        # questa matrix pare che giovi sempre dell'init della Xavier norm
        super().__init__(vocab_size, embedding_size, bias=False, _weights=_weights)
        #self.weights = inits.xavier_norm(vocab_size, embedding_size) if _weights is None else _weights
    
    def __call__(self, x: Tensor) -> Tensor:
        # OCCHIO: NON e' UN tensor e cosi' non andrebbe
        return Tensor(self.weights[x._device.nonzero(x)[0]], requires_grad=True)


class Dropout:
    '''
    Setta a 0 randomicamente alcuni valori dell'input secondo la probabilità data in input.
    Sono attivi solo durante la fase di training (cosa che devi implementare).

    probability: valore compreso tra (0, 1). Corrisponde alla percentuale di valori
                 settati a zero.
    '''
    __slots__ = ("p")

    def __init__(self, probability: float):
        self.p = probability

    def __call__(self, x: Tensor) -> Tensor:
        # ho due possibilità:
        # - genero una lista di indici delle entrate da settare a 0
        # - genero un vettore con 0 e 1 e moltiplico
        # per ora opto per il primo approccio che occupa meno memoria e dovrebbe
        # richiedere meno risorse

        # spiegazione parametri np.random.choice:
        # - range: intervallo in cui può estrarre i numeri. Settato a [0, len(x))
        # - size: len dell'output, quindi quanti indici restituisce. Settato a
        #         len(x) * probabilità, che equivale a dire 100% * 0.n = n%
        indices = x._device.random.choice(x._device.arange(x.data.size), replace=False,
                           size=int(x.data.size * self.p))
        x.data[indices] = 0
        # scalo l'output
        return x * 1/(1 - self.p)