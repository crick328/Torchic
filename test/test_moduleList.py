import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.3")
import torchic
from torchic import nn, Tensor
from torchic.nn import Linear
from torchic.nn import Module, ModuleList
from torchic.nn.models.transformer import *


import inspect
from collections.abc import Generator


"""
# Metaclass che forza l'uso di __slots__ e estende automaticamente gli slots
class ForceSlots(type):
    # Aggiunge '_init_params' agli __slots__ esistenti, se presenti
    def __new__(cls, name, bases, classdict):
        classdict['__slots__'] = ('_init_params',) + classdict.get('__slots__', ())
        print(classdict['__slots__'])
        return super().__new__(cls, name, bases, classdict)


# Classe base Module
class Module(metaclass=ForceSlots):
    '''
    Classe base per componenti che hanno `Tensor` trainabili.
    '''
    def parameters(self, trainable_only=True, with_names=False, _layer_name=""):
        for name, param in self:
            if isinstance(param, Tensor) and (not trainable_only or param.requires_grad):
                yield (f"{_layer_name}{name}", param) if with_names else param
            elif isinstance(param, Module):
                yield from param.parameters(trainable_only, with_names, _layer_name=f"{name}.")
            elif isinstance(param, ModuleList):
                for sublayer in param:
                    yield from sublayer.parameters(trainable_only, with_names, _layer_name=f"{name}.")

    def state_dict(self, layer_name: str) -> Generator:
        for name, param in self:
            if isinstance(param, Tensor) and param.requires_grad:
                yield f"{layer_name}.{name}", param.data
            elif isinstance(param, Module):
                yield from param.state_dict(f"{layer_name}.{name}")
            elif isinstance(param, ModuleList):
                for sublayer in param:
                    yield from sublayer.state_dict(f"{layer_name}.{name}")

    def to(self, *args):
        for _, param in self:
            if isinstance(param, (Tensor, Module)):
                param.to(*args)

    def freeze(self):
        self._set_requires_grad(False)

    def unfreeze(self):
        self._set_requires_grad(True)

    def _set_requires_grad(self, value: bool):
        for _, param in self:
            if isinstance(param, Tensor):
                param.requires_grad = value
            elif isinstance(param, Module):
                param._set_requires_grad(value)

    def __new__(cls, *args, **kwargs):
        # in pratica intercetta la creazione dell'oggetto
        # per salvare gli args
        instance = super().__new__(cls)
        # salvo (solo) i parametri di init passati dall'utente
        instance._init_params = (args, kwargs)
        return instance

    def __repr__(self):
        class_name = self.__class__.__name__
        # Non esce ModuleList,ma il wrapper
        args, kwargs = self._init_params
        # dizionario contenente tutti i parametri degli init
        # con relativi valori di default (se ci sono)
        init_signature = inspect.signature(self.__init__)
        
        # Ricostruzione della rappresentazione
        param_strs = []
        for i, (arg_name, default_value) in enumerate(init_signature.parameters.items()):
            # NOTE: !r nelle f-string e' un modo per chiedere la repr della
            # variabile nelle graffe

            # se e' un argomento posizionale
            if i < len(args):
                param_strs.append(f"{arg_name}={args[i]!r}")

            # se e' un argomento passato con keyword
            elif arg_name in kwargs:
                param_strs.append(f"{arg_name}={kwargs[arg_name]!r}")
                
            # allora e' un parametro che ha preso il valore di default
            else:
                param_strs.append(f"{arg_name}={default_value.default!r}")
        
        formatted_params = ", ".join(param_strs)
        return f"{class_name}({formatted_params})"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __iter__(self):
        for attr_name in self.__slots__:
            yield attr_name, getattr(self, attr_name)


class ModuleList(Module):
    __slots__ = ("modules",)

    def __init__(self, *args: list[Module] | tuple[Module]) -> None:
        self.modules = list(*args)

    def __iter__(self) -> Generator[Module, None, None]:
        yield from self.modules

    def __call__(self, x: Tensor) -> Tensor:
        # al momento non supporta i multi-input
        for module in self:
            x = module(x)
        return x

    def __getattribute__(self, function_name):
        # chiamata ogni volta che si accede ad un attributo.
        # Questo mi permette di intercettare le chiamate alle
        # funzioni di Module per poter ciclare, senza doverle
        # ridefinire una ad una in ModuleList

        def function_call_wrapper(*args, **kwargs):
            # chiamo la funzione interpellata da ogni Module
            for module in self.modules:
                yield from getattr(module, function_name)(*args, **kwargs)
        
        # puo' essere un attributo o una funzione
        x = super(ModuleList, self).__getattribute__(function_name)

        # non wrappo i metodi __magic__ e gli attributi
        if function_name.startswith('__') and function_name.endswith('__') or not callable(x):
            return x
        else:
            return function_call_wrapper
    
    def __getitem__(self, index: int|slice):
        return self.modules[index]














class Linear(Module):
    __slots__ = ("has_bias", "weights", "bias")

    def __init__(self, n_input: int=None, n_output: int=None, bias=True, _weights=None, _bias=None, _init_funct=nn.xavier_norm):
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
    def forward(self, x: Tensor) -> Tensor:
        x = x.matmul(self.weights)
        
        return x + self.bias if self.has_bias else x
    
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
    
    def __value_check(self, n_input=None, n_output=None, weights=None, bias=None):
        if not weights and not bias:
            if n_input is None or n_output is None:
                raise ValueError("You have to pass a value for n_input and n_output.")
            
            if n_input < 1 or n_output < 1:
                raise ValueError(
                    "n_input and n_output have to be strictly positive, received"
                    f"n_input={n_input} and n_output={n_output} instead."
                    )

    # metodo chiamato con print()
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        shape = self.weights.shape()
        return f"{self.__class__.__name__}(n_input={shape[0]}, n_output={shape[1]}, bias={self.has_bias})"
        


# ==================
from torchic.tools.helpers import zeros, ones, arange, log, sin, cos
from torchic import float32
from torchic.backward import EmbeddingBackward

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

class TokenEmbeddings(Module):
    '''
    Rete neurale che si occupa di creare gli embeddings.
    La rete cerca di modellare il significato delle parole per poi sfornare dei vettori
    d-dimensionali (embeddings) che racchiudono tutte le relative informazioni.

    d_model: int che rappresenta la lunghezza dei vettori di input (= embedding_dim)
    vocab_size: quante parole sono presenti all'interno del vocabolario

    output: embedding vector di dimensione 1 x d_model (vettore "sdraiato")
    '''
    # l'unica differenz dagli embedding normali e' che qua moltiplichi per sqrt(d_model)
    __slots__ = ("d_model_sqrt", "embedding")
    
    def __init__(self, d_model: int, vocab_size: int):
        # sto usando la nomenclatura del video (e anche del paper) per non fare confusione
        # cal
        self.d_model_sqrt = Tensor(d_model).sqrt()
        self.embedding = Embedding(vocab_size, d_model, _init_funct=nn.xavier_uniform)

    def forward(self, x: Tensor) -> Tensor:
        # nel paper originale si moltiplica per la radice di d_model

        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * self.d_model_sqrt
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    

class PositionalEncoding(Module):
    '''
    d_model: valore passato dall'InputEmbedding, è la lunghezza dei vettori di embedding
    seq_length: massimo numero di embedding che processo in una volta, quindi il numero di
        parole che posso processare in un colpo solo senza splittare
    dropout: vecchio brosky

    l'output di questo blocco, che è di dim , va direttamente alla multi head attention.
    '''
    __slots__ = ("pe_matrix", "dropout")

    def __init__(self, d_model: int, seq_length: int, dropout: float):
        self.dropout = nn.Dropout(dropout)
        # Empty matrix of shape (seq_len, d_model)
        self.pe_matrix = zeros([seq_length, d_model])
        # la formula da applicare cambia a seconda dell'index del vettore del
        # positional encoding. Se la posizione è pari uso sin, se è dispari cos.
        # (NOTE: perchè si usa una funzione pari se il numero è pari e viceversa?)
        # Anziché fare un ciclo for e calcolare la componente nelle singole iterazioni,
        # mi conviene vettorizzare tutto. Per questo, creo un vettore aventi valori
        # che vanno da [0, seq_length)
        # creo un vettore di shape (seq_len, 1)
        #position = np.expand_dims(np.arange(0, seq_length, dtype=np.float32), axis=1) # (seq_len, 1)
        position = arange(0, seq_length, dtype=float32).unsqueeze(1)  # (seq_len, 1)
        #https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        # Ora preparo il denominatore degli argomenti delle funzioni trigonometriche.
        # È stato riscritto sfruttando il trucco e^ln(x) per una maggiore stabilità numerica
        #denominator = np.exp(np.arange(0, d_model, 2) * -(np.log(10000) / d_model))  # (d_model / 2)
        denominator = (arange(0, d_model, 2) * -(log(10000) / Tensor(d_model))).exp() # (d_model / 2)
        # calcolo il seno nelle posizioni pari
        # (l'index strano sta dicendo di ignorare il n° di righe e operare a partire
        # dalla colonna 0, operando ogni due colonne, quindi una si e una no)
        #import numpy as np
        #self.pe_matrix.data[:, 0::2] = np.sin(position.data * denominator.data)
        #self.pe_matrix.data[:, 1::2] = np.cos(position.data * denominator.data)
        self.pe_matrix[:, 0::2] = sin(position.data * denominator.data).data
        self.pe_matrix[:, 1::2] = cos(position.data * denominator.data).data

        # equivalente ad unsqueeze di PyTorch. passo da [[a,b], [c,d]] a [ [[a,b], [c,d]] ],
        # da dim (seq_len x d_model) a (1 x seq_len x d_model)
        # ma non ho ancora ben capito perché bisogna farlo...
        # A quanto pare si fa perché quell'1 si riferisce al numero
        # di frasi (o meglio, sequenze) prese in esame.
        # Es: ciao sono Manu / Che schifo il pininfarina
        # Parametri: d_model = 512; seq_len (numero di token fisso per sequenza) = 5
        # Per semplicità, faccio finta che una parola sia un token.
        # => [ciao] = embedding vector (1 x 512) (= codifica del singolo token);
        # => [[ciao], [sono], [Manu], [EOS], [PADDING]] = sequenza (5 x 512)
        # => [[[ciao], [sono], [Manu], [EOS], [PADDING]], [[Che], [schifo], [il], [pininfarina], [EOS]]] = 2 sequenze diverse (2 x 5 x 512)
        # Quella singola dimensione si aggiunge per avere la possibilità di accumulare frasi
        #self.pe_matrix = Tensor(np.expand_dims(self.pe_matrix, axis=0)).data  # (1, seq_len, d_model)
        self.pe_matrix = self.pe_matrix.unsqueeze(0)  # (1, seq_len, d_model)

        # questo layer non è necessario salvarlo perché non ha valori trainati. Basta
        # segnare che c'è e ricrearlo da zero (PyTorch lo fa tramite "register_buffer")

    def forward(self, x: Tensor) -> Tensor:
        # aggiungo il pos_encoding vector all'input corrispondente
        # pe_matrix: 1 (dummy dim) x seq_len (n° righe) x d_model (n° colonne)
        # x: 1 x embedding_size
        # Non capisco l'indexing... Aspetto a sbatterci ancora la testa, magari lo corregge più tardi
        # Tensor + Tensor
        #x = x + Tensor(self.pe_matrix[:, :x.data.shape[1], :])
        x = x + self.pe_matrix[:, :x.data.shape[1], :]
        return self.dropout(x)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
class LayerNormalization(Module):
    '''
    https://www.youtube.com/watch?v=G45TuC6zRf4
    Fa esattamente ciò che promette il nome: normalizzare i valori
    per far sì che non si smaiali il training.
    (Presumibilmente farà rientrare i valori in un range [-1, 1]).

    Prende un embedding (X) dalla matrice, sottrae la media (vanilla, quindi somma
    delle componenti diviso numero delle componenti) dei suoi valori (M) e dividie per
    la std deviation (G). Questo viene moltiplicato per un coefficiente uguale
    per tutto il layer e sommato per un "bias", anch'esso uguale per tutto questo layer.
    y = k*[ (Xi-Mi)/(Gi^2+eps) ] + b
    Nota che M può essere riciclata nel calcolo di G.
    eps si usa nella formula della std dev per evitare che sia zero (visto che è un
    denominatore sarebbe un problemone)
    '''
    __slots__ = ("gamma", "eps", "bias")
    
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        #self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.gamma = ones(d_model, requires_grad=True)
        #self.bias = Tensor(np.zeros(d_model), requires_grad=True)
        self.bias = zeros(d_model, requires_grad=True)
        self.eps = Tensor(eps)

    def forward(self, x: Tensor) -> Tensor:
        # x è (batch_dim x 1 x embedding_size). Quel "axis=-1" indica che voglio
        # la somma sull'ultima dim, quindi in embedding_size, e keepdims=True
        # è per far sì che non perdo la batch_dim, che è 1
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True) # (batch, seq_len, 1)
        # '-1' means last dimension

        # con std che e' "div by zero" safe grazie a + eps
        #import numpy as np
        #return self.gamma * ((x - mean) / Tensor(np.sqrt(var + self.eps))) + self.bias
        # NOTE: questa sqrt da' problemi! Fa spaccare la bw tra parentesi
        # in realta' sqrt va, ma gli arriva un grad di shape diversa da self...
        return self.gamma * ((x - mean) / (var + self.eps).sqrt()) + self.bias
    
    def __call__(self, x: Tensor) -> Tensor:
        x = self.forward(x)
        return x
    

class FeedForwardBlock(Module):
    '''
    Non è altro che un layer con ReLU + dropout + layer senza activation function:
    Dropout ( ReLU(x @ W1 + b1) ) @ W2 + b2
    Alla fine della computazione, il layer restituisce un vettore delle
    stesse dimensioni di quello di input, quindi della dimensione degli embedding.
    Può essere vista come una mini rete neurale vanilla
    '''
    __slots__ = ("linear_layer1", "linear_layer2", "dropout", "activation")

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        # d_model x d_ff, nuovo parametro di questo layer
        # d_ff = 2048 nel paper
        self.linear_layer1 = Linear(d_model, d_ff, _init_funct=nn.xavier_uniform)
        self.dropout = Dropout(dropout)
        self.linear_layer2 = Linear(d_ff, d_model, _init_funct=nn.xavier_uniform)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_layer2( self.dropout( self.activation( self.linear_layer1(x) ) ) )

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    

class MultiHeadAttention(Module):
    # puoi parallelizzarla! Assegna un process a ciascuno dei tre forward
    # OPPURE puoi combinarli in un unico layer! Costa meno overhead e sarebbe
    # gia' tutto gestito!
    # https://github.com/ajhalthor/Transformer-Neural-Network/blob/main/transformer.py
    '''
    Il vettore che riceve in input (embedding vector di dimensione n x seq_len x dim) viene
    striplicato e prende il nome di query, key e value vector. Ognuno di
    questi vettori passa per un suo linear layer (indipendente) e il risultato viene diviso
    in d_model / h vettori. Nel paper, le attention head sono 8, quindi ciascun risultato
    viene diviso in 64 vettori. Da qui si applica la formula dell'attention, si concatenano
    i risultati e il vettore trovato passa per un ultimo linear layer.

    head_num: number of self attention heads
    '''
    __slots__ = (
        "head_num", "head_dim",
        "query_layer", "key_layer", "value_layer",
        "output_layer", "dropout", "softmax"
    )

    def __init__(self, d_model: int, head_num: int, dropout: int=None):
        if d_model % head_num != 0:
            ValueError(f"Cannot divide evenly d_model of {d_model} times h {head_num} head attentions") 
        
        # dimensione della frammentazione dei singoli embeddings. Ogni attention head
        # prende una sottoparte di esso e hanno tutte le stesse dimensioni (batch x seq_len x d_k)
        self.head_num = head_num                      # h
        self.head_dim = Tensor(d_model // head_num)         # d_k
        # NOTE: potrei usare un unica matrice per
        # questi 3 layer perche' la loro init e'
        # uniforme, MA richiederebbe la riscrittura
        # della classe per il forward
        self.query_layer = Linear(d_model, d_model, _init_funct=nn.xavier_uniform)  # Wq
        self.key_layer = Linear(d_model, d_model, _init_funct=nn.xavier_uniform)    # Wk
        self.value_layer = Linear(d_model, d_model, _init_funct=nn.xavier_uniform)  # Wv
        self.output_layer = Linear(d_model, d_model, _init_funct=nn.xavier_uniform) # Wo
        self.dropout = Dropout(dropout) if dropout is not None else None

        self.softmax = nn.Softmax(-1)

    # Scaled Dot-Product Attention
    # si puo' integrare meglio nel codice
    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> tuple[Tensor, Tensor]:
        # NOTE: questa self.head_dim si puo' calcolare nell'init e castarla subito a tensor
        
        # (batch, head_num, seq_len, head_dim) --> (batch, head_num, seq_len, seq_len)
        attention_scores = q.matmul(k.transpose(-2, -1)) / self.head_dim.sqrt()
        if mask is not None:
            # rimpiazzo con un valore piccolo; ci penserà poi la softmax a farlo
            # tendere a 0. Questo, ad esempio, sarà usato per i [PADDING] embedding
            # perché non devono interagire con il resto della frase.
            # NOTE: posso evitare di calcolare questa attention se è solo [PADDING]?
            # NOTE: non ha detto nulla riguardo la possibilità di mascherare
            # post softmax direttamente con 0. Perché? Ha solo seguito il paper
            # o c'è un motivo?
            attention_scores = attention_scores.masked_fill(mask, -1e9)

        # softmax solo sull'ultima dimensione
        # (batch x head_num x seq_len x seq_len)
        attention_scores = self.softmax(attention_scores)
        #print(attention_scores.shape())

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # se vuoi visualizzare gli scores, devi farlo prima di questa matmul
        # (batch, head_num, seq_len, seq_len) -> (batch, head_num, seq_len, head_dim)
        return attention_scores.matmul(v)
    
    def forward(self, q, k, v, mask=None):
        '''
        mask: utilizzato per fare in modo che alcune parole non interagiscano tra di loro.
            Si imposta l'entrata della matrice ad un numero molto piccolo in modo che la
            softmax la setti a 0.
        '''
        # (batch x seq_len x d_model) -> (batch x seq_len x d_model)
        q = self.query_layer(q) # Q'
        k = self.key_layer(k)   # K'
        v = self.value_layer(v) # V'
        
        # Splitto la terza dim del tensor in due, facendola
        # passare da (batch x seq_len x d_model) a
        # (batch x seq_len x head_num x head_dim), infine traspongo
        # la seconda e la terza dimensione, ritrovandomi con un
        # (batch x head_num x seq_len x head_dim). Quest'ultimo passaggio
        # è importante perché io dividerò i compiti a un certo
        # numero di head (head_num) e voglio che ciascuna di esse
        # lavori su un pezzo di frase, quindi su una matrice (seq_len x head_dim)
        # In soldoni, ogni head riceverà un vettore del tipo
        # (batch x 1 x seq_len x head_dim). Per chiarire, se sommassi i
        # vettori nella quarta dimensione, avrei (batch x 1 x seq_len x d_model)

        # (batch x seq_len x d_model) -> (batch x seq_len x head_num x head_dim) -> .T(1, 2) -> (batch x head_num x seq_len x head_dim)
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        x = self.attention(q, k, v, mask)
        # rimetto seq_len al suo posto, ovvero passo da
        # (batch x head_num x seq_len x head_dim) a (batch x seq_len x head_num x head_dim) e a (batch x seq_len x d_model)
        x = self.concat(x)

        # (batch x seq_len x d_model) -> (batch x seq_len x d_model)
        return self.output_layer(x)
    
    
    def split(self, tensor: Tensor) -> Tensor:
        '''
        split tensor by number of head

        (le dim hanno una nomenclatura diversa, ma aiutano a dare un altro punto di vista)
        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, head, seq_len, head_dim]

        source: https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py#L41
        '''
        # per assurdo potrebbe essere piu' veloce cosi' se e' vero che
        # le reference by . sono nemiche delle performance e i metodi
        # performano meglio se usano variabili dichiarate al loro interno.
        # E' una doppia ottimizzazione e il codice diventa molto piu' chiaro!
        batch_size, seq_len, d_model = tensor.data.shape
        d_tensor = d_model // self.head_num
        
        return tensor.view(batch_size, seq_len, self.head_num, d_tensor).transpose(1, 2)
    
    def concat(self, tensor: Tensor) -> Tensor:
        '''
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head_num, seq_len, head_dim]
        :return: [batch_size, seq_len, d_model]
        '''
        batch_size, head_num, seq_len, head_dim = tensor.data.shape#tensor.size()
        d_model = head_num * head_dim

        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    def __call__(self, q, k, v, mask=None) -> Tensor:
        return self.forward(q, k, v, mask)
    
# ---- quasilayers ----
    
# torch la chiama TransformerEmbeddings
class TransformerEmbeddings(Module):
    __slots__ = ("token_embeddings", "pos_encoding")

    def __init__(self, d_model: int, vocab_size: int, max_len: int, dropout: float):
        self.token_embeddings = TokenEmbeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.pos_encoding(self.token_embeddings(x))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


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


# Equivalente dell'encoder block di Jamil
# LAYER perche' sono di fila, NON in parallelo!
class EncoderLayer(Module):
    __slots__ = (
        "self_attention", "ffb",
        "dropout1", "dropout2",
        "norm1", "norm2"
    )


    def __init__(self, d_model: int, head_num: int, dim_feed_forward: int, drop_prob: float) -> None:
        self.self_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.dropout1 = Dropout(drop_prob)
        self.norm1 = LayerNormalization(d_model)

        self.ffb = FeedForwardBlock(d_model=d_model, d_ff=dim_feed_forward, dropout=drop_prob)
        self.dropout2 = Dropout(drop_prob)
        self.norm2 = LayerNormalization(d_model)
    
    def forward(self, x: Tensor, src_mask: Tensor=None) -> Tensor:
        residual_x = x
        x = self.dropout1(self.self_attention(q=x, k=x,v=x, mask=src_mask))
        x = self.norm1(residual_x + x)

        residual_x = x
        x = self.dropout2(self.ffb(x))
        x = self.norm2(residual_x + x)

        return x
    
    def __call__(self, x: Tensor, src_mask: Tensor=None) -> Tensor:
        return self.forward(x, src_mask)


# ---- pezzo fatto e finito ----

class Encoder(Module):
    __slots__ = ("emb", "layers")

    def __init__(self, d_model: int, enc_vocab_size: int, max_len: int, drop_prob: float, n_layers: int, head_num: int, dim_feed_forward: int):
        self.emb = TransformerEmbeddings(d_model, enc_vocab_size, max_len, drop_prob)
        self.layers = ModuleList(EncoderLayer(d_model, head_num, dim_feed_forward, drop_prob) for _ in range(n_layers))

    def forward(self, enc_input: Tensor, enc_mask: Tensor = None) -> Tensor:
        enc_input = self.emb(enc_input)

        for layer in self.layers:
            enc_input = layer(enc_input, enc_mask)
        return enc_input
    
    def __call__(self, enc_input: Tensor, enc_mask: Tensor = None) -> Tensor:
        return self.forward(enc_input, enc_mask)

# ==================
"""

















# Classe derivata ExampleModule
class ExampleModule(Module):
    __slots__ = ('a', 'b', 'c')

    def __init__(self, a, b, c=10):
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        return x * self.a + self.b - self.c



def get_real_size(obj):
    # overhead base e size __slots__
    size = sys.getsizeof(obj)
    
    for slot in obj.__slots__:
        attr = getattr(obj, slot)

        if isinstance(attr, ModuleList):
            for module in attr:
                size += get_real_size(module)
        elif isinstance(attr, (Tensor, Module)):
            size += get_real_size(attr)
        #elif isinstance(attr, (np.ndarray, cp.ndarray)):
            #size += attr.nbytes
        else:
            size += sys.getsizeof(attr)
    return size







sss = ModuleList([ExampleModule(2,3), ExampleModule(2,3)])
print(repr(sss))

print("______________")

b = ExampleModule(2,3)
#print(int(get_real_size(b)) /1024 /1024)         
#print(repr(b))

linear = Linear(2,3)
#print(int(get_real_size(linear)) /1024 /1024)    
#print(repr(linear))


mha = MultiHeadAttention(512, 8, 0.2)
# 4206768
# 4206768
print(int(get_real_size(mha)))       
print(repr(mha))


enc = Encoder(512, 22000, 310, 0.3, 1, 8, 2048)
# 57676480
# 57676480
print(int(get_real_size(enc)))     
print(repr(enc))
'''
multi_layer_enc = Encoder(512, 22000, 310, 0.3, 6, 8, 2048)
# 72.203
print(int(get_real_size(multi_layer_enc)) /1024 /1024)  
'''     
