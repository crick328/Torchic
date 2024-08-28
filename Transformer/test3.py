import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadAttention:
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

    num_heads: number of self attention heads
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: int=None):
        if d_model % num_heads != 0:
            ValueError(f"Cannot divide evenly d_model of {d_model} times h {num_heads} head attentions") 
        
        # dimensione della frammentazione dei singoli embeddings. Ogni attention head
        # prende una sottoparte di esso e hanno tutte le stesse dimensioni (batch x seq_len x d_k)
        self.num_heads = num_heads                      # h
        self.head_dim = d_model // num_heads            # d_k
        # NOTE: potrei usare un unica matrice per
        # questi 3 layer perche' la loro init e'
        # uniforme, MA richiederebbe la riscrittura
        # della classe per il forward
        self.query_layer = nn.Linear(d_model, d_model)  # Wq
        self.key_layer = nn.Linear(d_model, d_model)    # Wk
        self.value_layer = nn.Linear(d_model, d_model)  # Wv
        self.output_layer = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self.softmax = nn.Softmax(-1)

    # Scaled Dot-Product Attention
    # si puo' integrare meglio nel codice
    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> tuple[Tensor, Tensor]:
        device = torch

        # formula applicata para para
        # (batch x head_num x seq_len x head_dim) -> (batch x head_num x seq_len x seq_len)
        # NOTE: questa self.head_dim si puo' calcolare nell'init e castarla subito a tensor
        attention_scores = (q.matmul(k.transpose(-2, -1))) / device.sqrt(Tensor([self.head_dim]))
        if mask is not None:
            # rimpiazzo con un valore piccolo; ci penserà poi la softmax a farlo
            # tendere a 0. Questo, ad esempio, sarà usato per i [PADDING] embedding
            # perché non devono interagire con il resto della frase.
            # NOTE: posso evitare di calcolare questa attention se è solo [PADDING]?
            # NOTE: non ha detto nulla riguardo la possibilità di mascherare
            # post softmax direttamente con 0. Perché? Ha solo seguito il paper
            # o c'è un motivo?
            attention_scores = device.where(mask == 0, -1e9, mask) # rimpiazza con ~ -inf dove la condizione e' verificata

        # softmax solo sull'ultima dimensione
        # (batch x head_num x seq_len x seq_len)
        attention_scores = self.softmax(Tensor(attention_scores))

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # se vuoi visualizzare gli scores, devi farlo prima di questa matmul
        return (attention_scores.matmul(v))
    
    def forward(self, q, k, v, mask=None):
        '''
        mask: utilizzato per fare in modo che alcune parole non interagiscano tra di loro.
            Si imposta l'entrata della matrice ad un numero molto piccolo in modo che la
            softmax la setti a 0.
        '''
        # la riassegnazione non e' il problema...
        # non vorrei che si creano casini perche' tutti provano a sovrascrivere
        # la backward di x (che sarebbe q, k e v). No, lo impedisco

        # (batch x seq_len x d_model) -> (batch x seq_len x d_model)
        q = self.query_layer(q) # Q', stesse dim di x (batch x seq_len x d_model)
        k = self.key_layer(k)   # K'
        #print(f"prev: {v.requires_grad}, {v.grad_fn}")
        v = self.value_layer(v) # V'
        # settimo calcolo di value vector
        #print(f"self.value_layer(v): {v.requires_grad}, {v.grad_fn}, {v.is_leaf}")
        
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
        print(x.shape, x.grad_fn)
        x = self.output_layer(x)
        print(x.shape, x.grad_fn)
        return x#self.output_layer(x)
    
    
    def split(self, tensor: Tensor) -> Tensor:
        """
        split tensor by number of head

        (le dim hanno una nomenclatura diversa, ma aiutano a dare un altro punto di vista)
        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, head, seq_len, head_dim]

        source: https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py#L41
        """
        # per assurdo potrebbe essere piu' veloce cosi' se e' vero che
        # le reference by . sono nemiche delle performance e i metodi
        # performano meglio se usano variabili dichiarate al loro interno.
        # E' una doppia ottimizzazione e il codice diventa molto piu' chiaro!
        batch_size, length, d_model = tensor.data.shape
        d_tensor = d_model // self.num_heads
        
        return tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)
    
    def concat(self, tensor: Tensor) -> Tensor:
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head_num, seq_len, head_dim]
        :return: [batch_size, seq_len, d_model]
        """
        batch_size, head_num, length, head_dim = tensor.data.shape#tensor.size()
        d_model = head_num * head_dim

        return tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
    
    def __call__(self, q, k, v, mask=None) -> Tensor:
        return self.forward(q, k, v, mask)
    

# Esempio di utilizzo del modulo MultiHeadAttention
batch_size = 8
seq_length = 350
embed_dim = 512
num_heads = 8

x = torch.randn(batch_size, seq_length, embed_dim)
mha = MultiHeadAttention(embed_dim, num_heads)
output = mha(x, x, x)

# Backpropagation
loss = output.mean()
loss.backward()