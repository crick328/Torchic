'''
=> RULE OF THUMB: tutti gli array usati ni forward devono essere tensori.
   Perché? Perché anche se non richiedono gradient e non sono utili per il
   crt, devono comunque essere sullo stesso device!

Ci sono un po' di sanity check da fare.
Innanzitutto, pulisci il codice da tutti questi commenti estensivi.
Copia e organizza ciò che serve sul doc di Replica.
Secondo, bisogna controllare che le backward funzionino a dovere.

Dividi queste classi in file a seconda del loro tipo

CONTROLLA LE INIT RICHESTE DAI SINGOLI LAYER

'''

# parole > numeri tramite words embeddings
# positional_encoding => prendo i valori della rete di word embeddings e gli do in input a delle
# funzioni seno e coseno che diventano sempre più larghe per ogni parola (tipo cos(output[0], sin(output[1]), cos(2*output[3]), ecc.)).
# Sommo gli output delle funzioni trigonometriche agli output della word embeddings per ottenere le parole fatte e finite.
# Ottenuti questi nuovi valori, faccio un'altra matmul con i weights, ma stavolta non sommo il bias e non uso la act_fun.
# Questi si chiamano Query Values. Si fa lo stesso processo per calcolare le Keys, valori che consentono di calcolare
# la similarità del significato della stessa parola in contesti diversi grazie al dot product


from torchic.cuda_tensor import Tensor
#from torchic import nn
from torchic.tools.helpers import zeros, ones, arange, log, sin, cos #, chunk
from torchic.nn.module import Module, ModuleList
from torchic.nn.models.sequential import Sequential
from torchic.nn.activations import ReLU, Softmax
from torchic.nn.layers. linears import Linear, Embedding, Dropout
from torchic.nn.initializers import xavier_uniform, normal
from torchic.dtypes import float32
import warnings
import re


# Come scegliere i parametri:
# - embedding_dim e d_ff hanno il maggior impatto sulle capacita' del modello.
#   Avere un d_ff = 4*embedding_dim e' buono, mentre il minimo e' una scala 1:2.
# - il numero di teste dovrebbe crescere di pari pass con embedding_dim, altrimenti
#   si rischia di avere teste troppo grosse o troppo settoriali. Per farsi un'idea,
#   il rapporto embedding_dim/num_head non deve essere minore di 32.
# - il numero di layer pesa molto di piu' per generare testo che non nelle
#   traduzioni (come per embedding_dim). 12 pare essere il minimo sindacale.
# - la seq_len, come prima, diventa piu' importante in task di generazione. Il modello
#   puo' avere tutti i parametri che vuoi, ma, se non riesce a vedere i token in un
#   colpo solo, non puo' costruire relazioni (GPT-3 ha 2048).
#
# Capisic bene che sfruttare la mixed precision diventa vitale per far funzionare
# questo mostro... Credo che, per qualcosa di un minimo prestante, devi stare
# sul 1B (1 miliardo). Parti gradualmente ovviamente


# ---- Components ----

# NOTE: COMMENTA LE DIM OVUNQUE
class PositionalEncoding(Module):
    '''
    Implements positional encoding as described in "Attention is All You Need" (Vaswani et al., 2017).

    Positional encoding adds information about the position of tokens in a sequence to their embedding vectors. 
    Since the Transformer model does not have a built-in notion of sequence order (unlike RNNs), this mechanism 
    introduces positional information through a fixed set of sinusoidal functions.

    Args:
        embedding_dim (int): the dimensionality of the embedding vectors, passed from the
            InputEmbedding layer.
        seq_len (int): maximum number of tokens that can be processed in a single
            sequence without splitting.
        dropout (float): Dropout rate to prevent overfitting.

    Theory:
    The positional encoding is computed using sine and cosine functions of different frequencies. 
    For each position `pos` in the sequence and each dimension `i` in the embedding, the encoding is defined as:
    - \( PE(pos, 2i) = sin(pos / 10000^{2i/d\_model}) \)
    - \( PE(pos, 2i+1) = cos(pos / 10000^{2i/d\_model}) \)
    where `i` is the index of the dimension, `pos` is the position, and `embedding_dim` is the embedding size.

    **Even indices use sine**, while **odd indices use cosine**. This alternation ensures that each dimension has unique 
    positional information while maintaining smoothness for small positional changes.
    NOTE: perchè si usa una funzione pari se il numero è pari e viceversa?
    
    -The encoding is added directly to the token embeddings before being passed to the Multi-Head Attention block, 
    providing the model with positional context.

    Implementation Notes:
    - The use of \( exp(\log(x)) \) improves numerical stability when calculating the denominators of the sine and cosine functions.
    - The positional encoding matrix is expanded to include a dummy batch dimension (size 1) to facilitate broadcasting 
      when added to the input embeddings. This allows handling multiple sequences in parallel during training and inference.
    '''

    __slots__ = ("pe_matrix", "dropout")
    
    def __init__(self, embedding_dim: int, seq_len: int, dropout: float):
        self.dropout = Dropout(dropout)
        self.pe_matrix = zeros([seq_len, embedding_dim])

        # (seq_len, ) -> (seq_len, 1)
        position = arange(0, seq_len, dtype=float32).unsqueeze(1)
        # Riscritto sfruttando il trucco e^ln(x) per stabilità numerica
        denominator = (arange(0, embedding_dim, 2) * -(log(10000) / Tensor(embedding_dim))).exp() # (embedding_dim / 2)
        # prendo tutte le righe (":"), per le colonne parto da idx=0 con passo 2
        self.pe_matrix[:, 0::2] = sin(position.data * denominator.data).data
        #    //     //       //     idx=1 con passo 2
        self.pe_matrix[:, 1::2] = cos(position.data * denominator.data).data

        # Aggiungo una dimensione dummy in modo da poter fare broadcasting sulla batch dim
        # (seq_len, embedding_dim) -> (1, seq_len, embedding dim)
        self.pe_matrix = self.pe_matrix.unsqueeze(0)

    def __call__(self, x: Tensor) -> Tensor:
        # NOTE: uso l'indexing sulla seq_len in modo da sommare solo fino alla
        # lunghezza effettiva della sequenza, permettendo cosi' di ricevere
        # come input anche sequenze che non raggiungono seq_len.
        # Le altre dimensioni hanno ":", che fa sommare per tutti i valori

        # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)
        x = x + self.pe_matrix[:, :x.data.shape[1], :]
        return self.dropout(x)

    
class LayerNormalization(Module):
    # https://www.youtube.com/watch?v=G45TuC6zRf4
    # (Presumibilmente farà rientrare i valori in un range [-1, 1]).
    '''
    Fa esattamente ciò che promette il nome: normalizzare i valori
    per far sì che non si smaiali il training.

    Prende un embedding (X) dalla matrice, sottrae la media (M) e divide per
    la "std deviation" (sqrt(var + eps)). Questo viene moltiplicato per un
    coefficiente uguale per tutto il layer e sommato per un "bias",
    anch'esso uguale per tutto questo layer.
    y = k*[ (Xi-Mi)/sqrt(var+eps) ] + b
    
    eps si usa per evitare che sia zero
    '''
    __slots__ = ("gamma", "eps", "bias")
    
    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        self.gamma = ones(embedding_dim, requires_grad=True)
        self.bias = zeros(embedding_dim, requires_grad=True)
        self.eps = Tensor(eps)

    def __call__(self, x: Tensor) -> Tensor:
        # keepdims=True e' per non perdere la dimensione appena ridotta
        # (batch_dim, seq_len, embedding_size) -> (batch, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)

        #std = x.std(dim=-1, keepdim=True, unbiased=True)
        #return self.gamma * (x - mean) / (std + self.eps) + self.bias
        # questa versione e' conforme a PyTorch (eps e' sotto la radice della std)
        # e la varianza viene calcolata come UNBIASED, che ha piu' senso visto
        # che non sto stimando la varianza di una popolazione partendo
        # da un suo campione lol. Serve calcolare la varianza vera
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.bias
        
    

# Forse la xavier come init puzza
class FeedForwardBlock(Module):
    '''
    Non è altro che un layer con ReLU + dropout + layer senza activation function:
    Dropout ( ReLU(x @ W1 + b1) ) @ W2 + b2
    Alla fine della computazione, il layer restituisce un vettore delle
    stesse dimensioni di quello di input, quindi della dimensione degli embedding.
    Può essere vista come una mini rete neurale vanilla
    '''
    __slots__ = ("linear1", "linear2", "dropout", "activation")

    def __init__(self, embedding_dim: int, d_ff: int, dropout: float):
        # embedding_dim x d_ff, nuovo parametro di questo layer
        # d_ff = 2048 nel paper
        self.linear1 = Linear(embedding_dim, d_ff, _init_funct=xavier_uniform)#, gain=calculate_gain('relu'))
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, embedding_dim, _init_funct=xavier_uniform)#, gain=calculate_gain('relu'))
        self.activation = ReLU()

    def __call__(self, x: Tensor) -> Tensor:
        # (batch, seq_len, embedding_dim) -> (batch, seq_len, d_ff) -> (batch, seq_len, embedding_dim)
        return self.linear2( self.dropout( self.activation( self.linear1(x) ) ) )
    

class MultiHeadAttention(Module):
    '''
    Il vettore che riceve in input (embedding vector di dimensione n x seq_len x dim) viene
    striplicato e prende il nome di query, key e value vector. Ognuno di
    questi vettori passa per un suo linear layer (indipendente) e il risultato viene diviso
    in embedding_dim / h vettori. Nel paper, le attention head sono 8, quindi ciascun risultato
    viene diviso in 64 vettori. Da qui si applica la formula dell'attention, si concatenano
    i risultati e il vettore trovato passa per un ultimo linear layer.

    head_num: number of self attention heads
    '''
    __slots__ = (
        "head_num", "k_dim", "v_dim",
        "embedding_dim", "head_dim", "head_dim_sqrt",
        "query_layer", "key_layer", "value_layer",
        "output_layer", "dropout", "softmax",
        "qkv_layer", "_qkv_fused", "clip",
        "save_attn_weights", "_attn_weights"
    )

    def __init__(self, embedding_dim: int, head_num: int, dropout: int=None, k_dim: int=None,
                 v_dim: int=None, bias=True, clip: bool=True, allow_fusion: bool=False,
                 save_attn_weights: bool=False):
        if embedding_dim % head_num != 0:
            raise ValueError(f"Cannot divide evenly embedding_dim of {embedding_dim} times h {head_num} head attentions") 
        
        # dimensione della frammentazione dei singoli embeddings 
        self.head_num = head_num                       # h
        self.head_dim = embedding_dim // head_num   # d_k
        # per evitare computazioni inutili ad ogni forward
        self.head_dim_sqrt = Tensor(embedding_dim // head_num).sqrt()   # d_k, ma sotto radice
        self.embedding_dim = embedding_dim
        self.clip = clip
        self.save_attn_weights = save_attn_weights

        self.k_dim = k_dim if k_dim is not None else embedding_dim
        self.v_dim = v_dim if v_dim is not None else embedding_dim
        self._qkv_fused = allow_fusion and self.k_dim == self.v_dim and self.v_dim == embedding_dim

        if self._qkv_fused:
            # fondo i layer
            self.qkv_layer = Linear(embedding_dim, 3*embedding_dim, bias, _init_funct=xavier_uniform)
            # altrimenti l'__iter__ di Module cerca di prendere params che non esistono
            # (pytorch ha un problema simile perche' fa la stessa cosa)
            self.query_layer = None
            self.key_layer = None
            self.value_layer = None
        else:
            # dichiaro i layers separatamente
            self.qkv_layer = None
            self.query_layer = Linear(embedding_dim, embedding_dim, bias, _init_funct=xavier_uniform)  # Wq
            self.key_layer = Linear(embedding_dim, self.k_dim, bias, _init_funct=xavier_uniform)    # Wk
            self.value_layer = Linear(embedding_dim, self.v_dim, bias, _init_funct=xavier_uniform)  # Wv

        self.output_layer = Linear(embedding_dim, embedding_dim, bias, _init_funct=xavier_uniform) # Wo
        self.dropout = Dropout(dropout) if dropout is not None else None
        # popolato solo se l'utente chiede di salvare i weights prodotti dall'attention
        self._attn_weights = None

        self.softmax = Softmax(-1)

    # Scaled Dot-Product Attention
    # si puo' integrare meglio nel codice
    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor=None) -> tuple[Tensor, Tensor]:
        '''
        Computes the scaled dot-product attention. 

        In the first step, a dot product is performed between `q` (queries) and the transpose of `k` (keys) 
        (`q @ k.transpose(-2, -1)`). This operation evaluates the relevance or similarity between each query 
        and key. The result is then scaled by dividing it by the square root of `embedding_dim`, which improves 
        numerical stability and balances the attention scores.

        Next, a `Softmax` function is applied along the last dimension to normalize the attention scores, 
        converting them into probabilities. Optionally, a mask can be applied before the softmax step to 
        exclude certain positions from contributing to the attention. For instance, this is useful to mask 
        `[PAD]` tokens in sequences or, in the decoder, to prevent the model from attending to future tokens 
        during training, ensuring predictions depend only on past tokens.

        In the final step, the attention probabilities are used to compute a weighted sum of the value vectors 
        (`v`). This weighting process allows the model to focus on relevant parts of the input sequence.

        While the softmax is primarily used for normalization and numerical stability, it also introduces 
        interpretability, making it possible to understand which inputs the model is focusing on.
        '''

        # (batch, head_num, seq_len, head_dim) --> (batch, head_num, seq_len_q, seq_len_k)
        attn_weights = q.matmul(k.transpose(-2, -1)) / self.head_dim_sqrt
        # il clipping va applicato PRIMA della mask o la clippi e 
        # spacchi la softmax!
        if self.clip:
            # setto a min_value/max_value i valori fuori
            # dall'intervallo [min_value, max_value] per
            # stabilita' numerica
            attn_weights = attn_weights.clamp_(-10, 10)

        if mask is not None:
            # NOTE: perche' si maschera con -inf prima della softmax
            # e non si pone a 0 dopo? Perche' nel secondo modo i token
            # mascherati parteciperebbero al calcolo di probabilita' della
            # softmax e perderebbe tutto di senso!
            # NOTE: PyTorch fa semplicemente mask*float("-inf") e altro
            # NOTE: torch semplicemente accetta entrambe le maschere. Per una
            # usa masked_fill_(), per l'altra += float_mask
            attn_weights = attn_weights.masked_fill_(mask==False, float('-inf'))
            # ho usato questo per il training del 20/12
            #attn_weights = attn_weights.masked_fill(mask==False, -1e9)

        # softmax solo sull'ultima dimensione
        # (batch x head_num x seq_len x seq_len)
        attn_weights = self.softmax(attn_weights)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        if self.save_attn_weights:
            self._attn_weights = attn_weights

        # se vuoi visualizzare gli scores, devi farlo prima di questa matmul
        # (batch, head_num, seq_len, seq_len) -> (batch, head_num, seq_len, head_dim)
        return attn_weights.matmul(v)
    
    # NOTE: bisogna implementare la visualizzazione degli attn_scores (magari
    # prendi il codice da Ajay e non da Jamil)
    # https://github.com/pytorch/pytorch/blob/dc4db95540da06623c747bf0f2bf9f4af3d2925a/torch/nn/functional.py#L4026
    def __call__(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor=None) -> Tensor:
        if self._qkv_fused:
            if q is k and k is v:
                # self-attention
                qkv = self.qkv_layer(q)
                # (batch x seq_len x 3*embedding_dim) -> (batch x head_num x seq_len x 3*head_dim)
                batch, seq_len = qkv.data.shape[0], qkv.data.shape[1]

                # separo il qkv e faccio anche il passaggio da embedding_dim a sottovettori
                # (batch, seq_len, 3*embedding_dim) -> (batch, seq_len, 3, head_num, head_dim)
                qkv = qkv.view(batch, seq_len, 3, self.head_num, self.head_dim)
                # (batch, seq_len, 3, head_num, head_dim) -> (3, batch, head_num, seq_len, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)

                # chiedo la contiguita' per velocita' delle operazioni.
                # La chiedo separatamente per avere piu' chance di avere blocchi
                # di memoria disponibili (anziche' il mega blocco contiguo che
                # sarebbe richiesto da qkv) perche' .contiguous crea una copia in memoria
                q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()

            elif q is not k and k is v:
                # encoder-decoder attention

                # prendo le giuste porzioni di matrice
                W_q = self.qkv_layer.weights[:, :self.embedding_dim]
                W_kv = self.qkv_layer.weights[:, self.embedding_dim:]

                if self.qkv_layer.has_bias:
                    b_q = self.qkv_layer.bias[:self.embedding_dim]
                    b_kv = self.qkv_layer.bias[self.embedding_dim:]

                    q = q.matmul(W_q) + b_q
                    kv = v.matmul(W_kv) + b_kv
                else:
                    q = q.matmul(W_q)
                    kv = v.matmul(W_kv)

                # prendo le prime due dimensioni
                batch, seq_len_kv = kv.data.shape[0], kv.data.shape[1]
                # (batch, seq_len, 2*embedding_dim) -> (batch, seq_len, 2, head_num, head_dim)
                kv = kv.view(batch, seq_len_kv, 2, self.head_num, self.head_dim)
                # (batch, seq_len, 2, head_num, head_dim) -> (2, batch, head_num, seq_len, head_dim)
                kv = kv.permute(2, 0, 3, 1, 4)

                k, v = kv[0].contiguous(), kv[1].contiguous()
                q = self.split(q)
            else:
                warnings.warn(
                    "La MHA e' configurata per avere un'unica matrice fusa Wqkv, ma "
                    "in input sono stati ricevuti tre tensor diversi. Verra' "
                    "usato lo slicing per la corretta computazione, ma probabilmente "
                    "sara' piu' lento di un approccio con layer distinti per q, k e v."
                )
                # Slicing della matrice fusa per gestire input distinti
                W_q = self.qkv_layer.weights[:, :self.embedding_dim]
                W_k = self.qkv_layer.weights[:, self.embedding_dim:2*self.embedding_dim]
                W_v = self.qkv_layer.weights[:, 2*self.embedding_dim:]
                
                if self.qkv_layer.has_bias:
                    b_q = self.qkv_layer.bias[:self.embedding_dim]
                    b_k = self.qkv_layer.bias[self.embedding_dim:2*self.embedding_dim]
                    b_v = self.qkv_layer.bias[2*self.embedding_dim:]
                    
                    q = q.matmul(W_q) + b_q
                    k = k.matmul(W_k) + b_k
                    v = v.matmul(W_v) + b_v
                else:
                    q = q.matmul(W_q)
                    k = k.matmul(W_k)
                    v = v.matmul(W_v)
        else:
            # (batch x seq_len x embedding_dim) -> (batch x seq_len x embedding_dim)
            q = self.query_layer(q) # Q'
            # (batch x seq_len x embedding_dim) -> (batch x seq_len x k_dim)
            k = self.key_layer(k)   # K'
            v = self.value_layer(v) # V'
        
            # Ogni head riceverà un vettore del tipo (batch x 1 x seq_len x head_dim) 
            # in modo che si concentrino su una parte diversa dell'embedding
            # (batch x seq_len x embedding_dim) -> (batch x seq_len x head_num x head_dim) -> .T(1, 2) -> (batch x head_num x seq_len x head_dim)
            q = self.split(q)
            k = self.split(k)
            v = self.split(v)
        
        x = self.attention(q, k, v, mask)
        # rimetto seq_len al suo posto, ovvero passo da
        # (batch x head_num x seq_len x head_dim) a (batch x seq_len x head_num x head_dim) e a (batch x seq_len x embedding_dim)
        x = self.concat(x)

        # (batch x seq_len x embedding_dim) -> (batch x seq_len x embedding_dim)
        return self.output_layer(x)
    
    
    def split(self, tensor: Tensor) -> Tensor:
        """
        split tensor by number of head

        (le dim hanno una nomenclatura diversa, ma aiutano a dare un altro punto di vista)
        :param tensor: [batch_size, seq_len, embedding_dim]
        :return: [batch_size, head, seq_len, head_dim]

        source: https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py#L41
        """
        # per assurdo potrebbe essere piu' veloce cosi' se e' vero che
        # le reference by . sono nemiche delle performance e i metodi
        # performano meglio se usano variabili dichiarate al loro interno.
        # E' una doppia ottimizzazione e il codice diventa molto piu' chiaro!
        batch_size, seq_len, _embedding_dim = tensor.data.shape
        # d_tensor e' head_dim
        
        return tensor.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2).contiguous()
    
    def concat(self, tensor: Tensor) -> Tensor:
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head_num, seq_len, head_dim]
        :return: [batch_size, seq_len, embedding_dim]
        """
        #batch_size, head_num, seq_len, head_dim = tensor.data.shape
        batch_size, seq_len = tensor.data.shape[0], tensor.data.shape[2]
        #embedding_dim = head_num * head_dim
        #print(tensor.shape(), batch_size, seq_len)

        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
    
    
# ---- Layers ----

# questa classe la toglierei, anche perche' non so se voglio tenere
# lo scaling e in futuro voglio cambiare PositionalEncoding.
# Terrei TransformerV2 come TransformerBilingual
class TransformerEmbeddings(Module):
    '''
    Modulo che si occupa della gestione dei vettori di significato dei tokens.
    Dato un vettore di ids di dimensione (batch, seq_len), per ciascun is
    (di cui ne hai seq_len) viene assegnato un vettore (1, embedding_dim)

    Args:
        embedding_dim (int): (= d_model nel paper)
        vocab_size (int): quante parole sono presenti all'interno del vocabolario

    '''
    __slots__ = ("token_embeddings", "embedding_dim_sqrt", "pos_encoding")

    def __init__(self, embedding_dim: int, vocab_size: int, max_len: int, dropout: float):
        self.token_embeddings = Embedding(vocab_size, embedding_dim, _init_funct=normal)
        self.embedding_dim_sqrt = Tensor(embedding_dim).sqrt()
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len, dropout)

    def __call__(self, x: Tensor) -> Tensor:
        # come nel paper, riscalo gli embeddings per la sqrt (si fa ancora?)
        # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        return self.pos_encoding(self.token_embeddings(x)*self.embedding_dim_sqrt)


# Equivalente dell'encoder block di Jamil
# LAYER perche' sono di fila, NON in parallelo!
class EncoderLayer(Module):
    __slots__ = (
        "self_attention", "ffb",
        "dropout1", "dropout2",
        "norm1", "norm2", "norm_first"
    )


    def __init__(self, embedding_dim: int, head_num: int, dim_feed_forward: int, drop_prob: float,
                 norm_first: bool=True, allow_fusion: bool=False) -> None:
        self.norm_first = norm_first
        
        self.self_attention = MultiHeadAttention(embedding_dim, head_num, allow_fusion=allow_fusion)
        self.dropout1 = Dropout(drop_prob)
        self.norm1 = LayerNormalization(embedding_dim)

        self.ffb = FeedForwardBlock(embedding_dim=embedding_dim, d_ff=dim_feed_forward, dropout=drop_prob)
        self.dropout2 = Dropout(drop_prob)
        self.norm2 = LayerNormalization(embedding_dim)
    
    def __call__(self, x: Tensor, src_mask: Tensor=None) -> Tensor:
        if self.norm_first:
            residual_x = x
            # applico la normalizzazione prima di passare x al layer
            x = self.norm1(x)
            x = residual_x + self.dropout1(self.self_attention(q=x, k=x,v=x, mask=src_mask))
            
            residual_x = x
            # applico la normalizzazione prima di passare x al layer
            x = self.norm2(x)
            x = residual_x + self.dropout2(self.ffb(x))
        else:
            # applico la normalizazione come ultimo step (come nel paper originale)
            residual_x = x
            x = self.dropout1(self.self_attention(q=x, k=x,v=x, mask=src_mask))
            x = self.norm1(residual_x + x)

            residual_x = x
            x = self.dropout2(self.ffb(x))
            x = self.norm2(residual_x + x)
        return x


class DecoderLayer(Module):
    __slots__ = (
        "self_attention", "cross_attention", "ffb",
        "dropout1", "dropout2", "dropout3",
        "norm1", "norm2", "norm3", "norm_first"
    )

    def __init__(self, embedding_dim: int, head_num: int, dim_feed_forward:int, drop_prob: float,
                 norm_first: bool=True, allow_fusion: bool=False) -> None:
        self.norm_first = norm_first
        
        self.self_attention = MultiHeadAttention(embedding_dim=embedding_dim, head_num=head_num, allow_fusion=allow_fusion)
        self.dropout1 = Dropout(drop_prob)
        self.norm1 = LayerNormalization(embedding_dim)

        self.cross_attention = MultiHeadAttention(embedding_dim=embedding_dim, head_num=head_num, allow_fusion=allow_fusion)
        self.dropout2 = Dropout(drop_prob)
        self.norm2 = LayerNormalization(embedding_dim)

        self.ffb = FeedForwardBlock(embedding_dim=embedding_dim, d_ff=dim_feed_forward, dropout=drop_prob)
        self.dropout3 = Dropout(drop_prob)
        self.norm3 = LayerNormalization(embedding_dim)

    def __call__(self, x: Tensor, enc_out: Tensor, src_mask: Tensor=None, trg_mask: Tensor=None) -> Tensor:
        '''
        Args:
            x: Tensor di input (decoder embedding + positional encoding).
            enc_out: Output del layer encoder (memory).
            src_mask: Mask opzionale per nascondere posizioni nell'encoder output.
            trg_mask: Mask causale per nascondere posizioni future nel decoder input.

        Returns:
            Tensor processato dal decoder layer.
        '''
        # questo x dovrebbe essere trg post embedding...
        if self.norm_first:
            residual_x = x
            x = self.norm1(x)
            x = residual_x + self.dropout1(self.self_attention(q=x, k=x, v=x, mask=trg_mask))
        
            residual_x = x
            # enc_out e' memory
            x = residual_x + self.dropout2(self.cross_attention(q=self.norm2(x), k=enc_out, v=enc_out, mask=src_mask))
        
            residual_x = x
            x = residual_x + self.dropout3(self.ffb(self.norm3(x)))
        else:
            residual_x = x
            x = self.dropout1(self.self_attention(q=x, k=x,v=x, mask=trg_mask))
            x = self.norm1(residual_x + x)

            residual_x = x
            x = self.dropout2(self.cross_attention(q=x, k=enc_out, v=enc_out, mask=src_mask))
            x = self.norm2(residual_x + x)

            residual_x = x
            x = self.dropout3(self.ffb(x))
            x = self.norm3(residual_x + x)
        return x


# ---- pezzo fatto e finito ----

class Encoder(Module):
    __slots__ = ("emb", "layers", "norm")

    def __init__(self, embedding_dim: int, enc_vocab_size: int, max_len: int, drop_prob: float,
                 n_layers: int, head_num: int, dim_feed_forward: int, allow_fusion: bool=False):
        self.emb = TransformerEmbeddings(embedding_dim, enc_vocab_size, max_len, drop_prob)
        self.layers = ModuleList(EncoderLayer(embedding_dim, head_num, dim_feed_forward, drop_prob, allow_fusion=allow_fusion) for _ in range(n_layers))
        self.norm = LayerNormalization(embedding_dim)

    def __call__(self, enc_input: Tensor, enc_mask: Tensor = None) -> Tensor:
        enc_input = self.emb(enc_input)

        for layer in self.layers.modules:
            enc_input = layer(enc_input, enc_mask)
        # NOTE: qua probabilmente non deve esserci Norm se norm_fisrt e' False 
        return self.norm(enc_input)
    
    
class Decoder(Module):
    __slots__ = ("emb", "layers", "norm")

    def __init__(self, embedding_dim: int, dec_vocab_size: int, max_len: int, drop_prob: float,
                 n_layers: int, head_num: int, dim_feed_forward: int, allow_fusion: bool=False):
        self.emb = TransformerEmbeddings(embedding_dim, dec_vocab_size, max_len, drop_prob)
        self.layers = ModuleList(DecoderLayer(embedding_dim, head_num, dim_feed_forward, drop_prob, allow_fusion=allow_fusion) for _ in range(n_layers))
        self.norm = LayerNormalization(embedding_dim)
        #self.linear = nn.Linear(embedding_dim, dec_vocab_size)
    
    def __call__(self, dec_input: Tensor, enc_out: Tensor, src_mask: Tensor=None, trg_mask: Tensor=None) -> Tensor:
        dec_input = self.emb(dec_input)

        for layer in self.layers.modules:
            dec_input = layer(dec_input, enc_out, src_mask, trg_mask)
        return self.norm(dec_input)
    
# Volendo, Transformer puo' ereditare da Module... Bisogna solo aggiornare
# il load per distinguere tra Sequential e Module (e altre cose).
# Permetterebbe l'uso degli slots, ma perderei il print e le altre
# funzioni custom di Sequential
# Pro:
# - model.parameters() sarebbe nativo
# - la repr() diventa piu' chiara (posso riproporla stile print, divisa a livelli?)
# - ho gli slots
# - concettualmente piu' accurato
# Contro:
# - devo modificare il load per leggere direttamente gli args del modello
#   (potrebbe essere un pro perche' posso staccarmi dalla vecchia logica)
# - devo rivedere la creazione dello state_dict() (cercherei di incorporarlo in Module)
# - devo rivedere la creazione del print (posso renderlo un metodo esterno, un helper o
#   un visualizer, e __str__() chiama a quello, ma deve passare su cosa iterare)
# - non ho piu' number_of_params()
# In caso, fai un TransformerV2 per non perdere la retrocompatibilita' con il vecchio.
# PRIMA ragiona su come fixxare i contro.
# PS: la conversione puoi farla anche in un altro momento, tanto ora devi lavorare
# sul checkpoint che ha il formato con Sequential
class Transformer(Sequential):
    # usa xavier uniform
    
    def __init__(self, *args,
                 embedding_dim: int=None,
                 encoder_vocab_size: int=None,
                 dec_vocab_size: int=None,
                 max_len: int=None,
                 drop_prob: float=0.1,
                 n_layers: int=6,
                 head_num: int=8,
                 dim_feed_forward: int=2048,
                 allow_fusion=True) -> None:
        # se l'utente (o la funzione load()) mi passa i layers:
        if args:
            assert all(isinstance(arg, Module) for arg in args)
            super().__init__(*args)

            if False:
                # trimmo l'ultimo char per togliere gli identificatori che qua non servono
                for name in list(self.__dict__.keys()):
                    self.__dict__[name[:-1]] = self.__dict__.pop(name)
        else:
            self.encoder0 = Encoder(embedding_dim, encoder_vocab_size, max_len, drop_prob, n_layers, head_num, dim_feed_forward)
            self.decoder1 = Decoder(embedding_dim, dec_vocab_size, max_len, drop_prob, n_layers, head_num, dim_feed_forward)
            # projection layer, converte le dimensioni
            self.linear2 = Linear(embedding_dim, dec_vocab_size)
        
    
    '''
    def __init__(self, *args, _trim_indices: bool=False):
        super().__init__(*args)
    '''

    #def __iter__(self):
        #"""Itera attraverso gli attributi definiti in __slots__."""
        #for attr_name in self.__slots__:
            #yield self.__getattribute__(attr_name)

    def __call__(self, enc_input: Tensor, decoder_input: Tensor, enc_mask, dec_mask) -> Tensor:
        enc_out = self.encoder0(enc_input, enc_mask) # (batch, seq_len, embedding_dim)
        output = self.decoder1(decoder_input, enc_out, enc_mask, dec_mask) # (batch, seq_len, embedding_dim)
        output = self.linear2(output)  # (batch, seq_len, trg_voab_size)
        # chiamo la softmax sulla dim di dec_vocab_size perche' voglio
        # la prob del next token
        # NOTE: NON va qui perche' la softmax e' incorporata nella CrossEntropyLoss
        #return self.softmax(output)
        return output
    
    def get_saved_attn_weights(self) -> list[Tensor]:
        """
        Extracts and organizes the _attn_weights from a Transformer model's layers into a hierarchical dictionary.

        Returns:
            dict: A hierarchical dictionary containing _attn_weights organized by encoder/decoder, layer index, 
                and attention type (self or cross).
        """
        def get_layer_structure(name: str):
            import re
            """
            Extracts information about encoding/decoding, layer index, and attention type from a parameter name.

            Args:
                name (str): The parameter name to analyze.

            Returns:
                tuple: A tuple containing (encoder/decoder, layer index, attention type) or None if the pattern doesn't match.
            """
            pattern = r"(encoder|decoder)\d\.layers\.(\d+)\.(self_attention|cross_attention)"
            match = re.search(pattern, name)
            if match:
                return match.group(1), int(match.group(2)), match.group(3)
            return None
            
        # Dictionary to store the attention weights
        hierarchical_weights = {}

        # Iterate through the state_dict to find _attn_weights
        for name, param in self.parameters(trainable_only=False, with_names=True, tensor_only=False):
            # Extract base name of the parameter
            p_name = name.rsplit(".", 1)[-1]

            # Check if the name indicates attention weights
            if p_name == "_attn_weights":
                # Extract structure info (encoder/decoder, layer index, attention type)
                info = get_layer_structure(name)
                if info:
                    enc_dec, layer_idx, attn_type = info

                    # Organize into hierarchical dictionary
                    if enc_dec not in hierarchical_weights:
                        hierarchical_weights[enc_dec] = {}
                    if layer_idx not in hierarchical_weights[enc_dec]:
                        hierarchical_weights[enc_dec][layer_idx] = {}
                    hierarchical_weights[enc_dec][layer_idx][attn_type] = param

        return hierarchical_weights

    def set_save_attn_weights(self, flag: bool) -> None:
        def insepct_layer(layer, flag):
            for _name, param in layer:
                if isinstance(param, MultiHeadAttention):
                    param.save_attn_weights = flag

                #elif isinstance(param, ModuleList):
                    #for module in param:
                        #insepct_layer(module, flag) 

                elif isinstance(param, Module):
                    insepct_layer(param, flag)

        for layer in self:
            insepct_layer(layer, flag)
    

def create_transformer(embedding_dim: int,
                 encoder_vocab_size: int,
                 dec_vocab_size: int,
                 max_len: int,
                 drop_prob: float=0.1,
                 n_layers: int=6,
                 head_num: int=8,
                 dim_feed_forward: int=2048,
                 allow_fusion=True) -> Transformer:
    
    # Inizializzo i layer
    encoder = Encoder(embedding_dim, encoder_vocab_size, max_len, drop_prob, n_layers, head_num, dim_feed_forward)
    decoder = Decoder(embedding_dim, dec_vocab_size, max_len, drop_prob, n_layers, head_num, dim_feed_forward)
    linear = Linear(embedding_dim, dec_vocab_size)
    return Transformer(encoder, decoder, linear)























class TransformerV2(Module):
    __slots__ = ("encoder", "decoder", "linear")
    
    def __init__(self,
                 embedding_dim: int=None,
                 encoder_vocab_size: int=None,
                 dec_vocab_size: int=None,
                 max_len: int=None,
                 drop_prob: float=0.1,
                 n_layers: int=6,
                 head_num: int=8,
                 dim_feed_forward: int=2048,
                 allow_fusion=True) -> None:

        self.encoder = Encoder(embedding_dim, encoder_vocab_size, max_len, drop_prob, n_layers, head_num, dim_feed_forward, allow_fusion=allow_fusion)
        self.decoder = Decoder(embedding_dim, dec_vocab_size, max_len, drop_prob, n_layers, head_num, dim_feed_forward, allow_fusion=allow_fusion)
        # projection layer, converte le dimensioni
        self.linear = Linear(embedding_dim, dec_vocab_size)

    def __call__(self, enc_input: Tensor, decoder_input: Tensor, enc_mask, dec_mask) -> Tensor:
        enc_out = self.encoder(enc_input, enc_mask) # (batch, seq_len, embedding_dim)
        output = self.decoder(decoder_input, enc_out, enc_mask, dec_mask) # (batch, seq_len, embedding_dim)
        output = self.linear(output)  # (batch, seq_len, trg_voab_size)
        # chiamo la softmax sulla dim di dec_vocab_size perche' voglio
        # la prob del next token
        # NOTE: NON va qui perche' la softmax e' incorporata nella CrossEntropyLoss
        #return self.softmax(output)
        return output
    
    def get_saved_attn_weights(self) -> list[Tensor]:
        """
        Extracts and organizes the _attn_weights from a Transformer model's layers into a hierarchical dictionary.

        Returns:
            dict: A hierarchical dictionary containing _attn_weights organized by encoder/decoder, layer index, 
                and attention type (self or cross).
        """
        # Dictionary to store the attention weights
        hierarchical_weights = {}
        pattern = r"(encoder|decoder)\.layers\.(\d+)\.(self_attention|cross_attention)\._attn_weights"

        # Iterate through the state_dict to find _attn_weights
        for name, param in self.parameters(trainable_only=False, with_names=True, tensor_only=False):
            match = re.search(pattern, name)

            if match:
                enc_dec, layer_idx, attn_type = match.group(1), int(match.group(2)), match.group(3)

                # Organize into hierarchical dictionary
                if enc_dec not in hierarchical_weights:
                    hierarchical_weights[enc_dec] = {}

                if layer_idx not in hierarchical_weights[enc_dec]:
                    hierarchical_weights[enc_dec][layer_idx] = {}
                hierarchical_weights[enc_dec][layer_idx][attn_type] = param

        return hierarchical_weights

    def set_save_attn_weights(self, flag: bool) -> None:
        def insepct_layer(layer, flag):
            for name, param in layer:
                if isinstance(param, MultiHeadAttention):
                    param.save_attn_weights = flag

                elif isinstance(param, Module):
                    insepct_layer(param, flag)

        for name, layer in self:
            if isinstance(layer, Module):
                insepct_layer(layer, flag)
    


# NOTE: torch ha una sua nomenclatura per ciascuno di essi (che l'indiano
# simpatico ha seguito)! Copiala a fine lavoro
# Prima segui Jamil per costruirti un quadro completo e coerente del funzionamento
# del modello, DOPO vedi se si puo' smussare e se vale la pena scrivere delle
# attention diverse per encoder e decoder

# https://github.com/ajhalthor/Transformer-Neural-Network/blob/main/transformer.py
# https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py#L41
# https://github.com/hyunwoongko/transformer/blob/master/models/blocks/encoder_layer.py