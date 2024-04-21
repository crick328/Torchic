from replica import CudaTensor, nn
import numpy as np
import cupy as cp 

# Linear Layer (corrisponde al Dense layer di Tensorflow se quest'ultimo è settato con activation=None)
class LLayer:
    def __init__(self, n_input: int, n_output: int, _weights=None, _bias=None):
        '''
        n_input: number of values received
        n_output: number of values returned
        _weights: sets the value of the weight matrix. Used when loading a model and it's not intended
            for user usage.
        _bias: sets the value of the bias vector. Used when loading a model and it's not intended
            for user usage.
        '''
        self.weights = nn.xavier_norm(n_input, n_output) if _weights is None else _weights
        self.bias = nn.xavier_norm(n_input, n_output, is_bias=True) if _bias is None else _bias
    
    def __call__(self, x: CudaTensor) -> CudaTensor:
        return x.matmul(self.weights) + self.bias
    
    def __str__(self):
        # metodo chiamato con print()
        return f"LLayer(weights = {self.weights.shape()}, bias = {self.bias.shape()})"
    
    def __repr__(self):
        return f"LLayer(n_input={self.weights.shape()[0]}, n_output={self.bias.shape()[0]}, _weights={repr(self.weights)}, _bias={repr(self.bias)})"

    def state_dict(self):
        return {
            self.name: {
                "a": 3
            }
        }
    

class Embedding:
    def __init__(self, vocab_size: int, embedding_size: int, _weights=None, _bias=None):
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
        creerà il grad per la riga dell'embedding layer. Un LLayer aggiorna la w_matrix in un colpo solo,
        qui invece si creano diversi grad (che però si possono accumulare prima di aggiornare la matrice)

        vocab_size: number of values received
        embedding_size: number of values returned
        _weights: sets the value of the weight matrix. Used when loading a model and it's not intended
            for user usage.
        '''
        # questa matrix pare che giovi sempre dell'init della Xavier norm
        self.weights = nn.xavier_norm(vocab_size, embedding_size) if _weights is None else _weights
    
    def __call__(self, x: CudaTensor) -> CudaTensor:
        return self.weights[np.nonzero(x)[0]]
    
    def __str__(self):
        # metodo chiamato con print()
        return f"Embedding(weights = {self.weights.shape()})"
    
    def __repr__(self):
        return f"Embedding(vocab_size={self.weights.shape()[0]}, embedding_size={self.weights.shape()[1]}, _weights={repr(self.weights)})"

    def state_dict(self):
        return {
            self.name: {
                "a": 3
            }
        }


class Dropout:
    '''
    Setta a 0 randomicamente alcuni valori dell'input secondo la probabilità data in input.
    Sono attivi solo durante la fase di training (cosa che devi implementare).

    probability: valore compreso tra (0, 1). Corrisponde alla percentuale di valori
                 settati a zero.
    '''

    def __init__(self, probability: float):
        self.p = probability

    def forward(self, x: CudaTensor) -> CudaTensor:
        # ho due possibilità:
        # - genero una lista di indici delle entrate da settare a 0
        # - genero un vettore con 0 e 1 e moltiplico
        # per ora opto per il primo approccio che occupa meno memoria e dovrebbe
        # richiedere meno risorse

        # spiegazione parametri np.ranodm.choice:
        # - range: intervallo in cui può estrarre i numeri. Settato a [0, len(x))
        # - size: len dell'output, quindi quanti indici restituisce. Settato a
        #         len(x) * probabilità, che equivale a dire 100% * 0.n = n%
        indices = np.random.choice(np.arange(x.data.size), replace=False,
                           size=int(x.data.size * self.p))
        x.data[indices] = 0
        # scalo l'output
        return x * 1/(1 - self.p)


class NNetwork:
    '''
    Questa classe si può usare in due modi:
    - chiami la classe e gli passi in input una lista/tupla contenente i layer e
      e le activation function in ordine e io gestisco in automatico la scrittura
      della forward;
    - creare una nuova classe che eredita da questa, creare variabili per layer
      e activation function, ma così devi scriverti la forward tu.
    Questo funzionamento mimica Pytorch.
    '''

    def __init__(self, *args):
        for i, element in enumerate(args):
            self.__dict__[f"{element.__class__.__name__.lower()}{i}"] = element

    def forward(self, x) -> CudaTensor:
        #assert type(x) == Tensor, f"Expected Tensor type object, received {type(x)}"

        # itero gli args
        # possibile che il forward avvenga male? i dict non sono ordinati...
        # Update December 2017: dicts retaining insertion order is guaranteed for Python 3.7
        # piace alla gpu? Non credo...
        for key in self.__dict__:
            x = self.__dict__[key](x)
        return x
    
    def to(self, device):
        for key in self.__dict__:
            if type(self.__dict__[key]) == LLayer:
                self.__dict__[key].weights.to(device)
                self.__dict__[key].bias.to(device)
            else:
                self.__dict__[key]._xp = cp if device == "cuda" else np


    def parameters(self):
        """
        returns all trainable tensors of the model
        """
        params = []

        for key in self.__dict__:
            if type(self.__dict__[key]) == LLayer:
                if self.__dict__[key].weights.requires_grad:
                    params.append(self.__dict__[key].weights)

                if self.__dict__[key].bias.requires_grad:
                    params.append(self.__dict__[key].bias)

        return params
    

    def number_of_params(self):
        number_of_params = 0
        for key in self.__dict__:
            if type(self.__dict__[key]) == LLayer:
                number_of_params += self.__dict__[key].weights.data.size + self.__dict__[key].bias.data.size
        return number_of_params
    
    def state_dict(self):
        nnetwork_class = self.__class__.__name__
        output = {nnetwork_class: []}
        for key in self.__dict__:
            output[nnetwork_class].append(self.__dict__[key])
        return
    
    def __repr__(self):
        output = "NNetwork("
        for key in self.__dict__:
            output = output + f"{repr(self.__dict__[key])}, "
        return output[0:-2] + ")"

    def __call__(self, x):
        return self.forward(x)
    

import base64
import gzip
#import ast

def save_model_type1(model, filename="type1"):
    data = "{}".format(repr(model)).encode()
    data = base64.b64encode(data)
    open(f"{filename}.rpl", "wb").write(gzip.compress(data)) # save data

def load_model_type1(filename):
    data = open(f"{filename}.rpl", "rb").read()
    data = gzip.decompress(data)
    data = base64.b64decode(data)
    #return ast.literal_eval(data.decode())[0]#data.decode()
    return eval(data.decode())