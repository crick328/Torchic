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


from torchic import Tensor, nn
import numpy as np


# ---- componenti subatomiche ----

class TokenEmbeddings:
    '''
    Rete neurale che si occupa di creare gli embeddings.
    La rete cerca di modellare il significato delle parole per poi sfornare dei vettori
    d-dimensionali (embeddings) che racchiudono tutte le relative informazioni.

    d_model: int che rappresenta la lunghezza dei vettori di input (= embedding_dim)
    vocab_size: quante parole sono presenti all'interno del vocabolario

    output: embedding vector di dimensione 1 x d_model (vettore "sdraiato")
    '''
    def __init__(self, d_model: int, vocab_size: int):
        # sto usando la nomenclatura del video (e anche del paper) per non fare confusione
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x è un vettore OHE
        # nel paper si moltiplica per la radice di d_model
        # NOTE: non hai una MulBackward
        return self.embedding(x) * np.sqrt(self.d_model)
    

class PositionalEncoding:
    '''
    d_model: valore passato dall'InputEmbedding, è la lunghezza dei vettori di embedding
    seq_length: massimo numero di embedding che processo in una volta, quindi il numero di
        parole che posso processare in un colpo solo senza splittare
    dropout: vecchio brosky

    l'output di questo blocco, che è di dim , va direttamente alla multi head attention.
    '''
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        self.dropout = nn.Dropout(dropout)
        self.pe_matrix = np.zeros(seq_length, d_model)
        # la formula da applicare cambia a seconda dell'index del vettore del
        # positional encoding. Se la posizione è pari uso sin, se è dispari cos.
        # (NOTE: perchè si usa una funzione pari se il numero è pari e viceversa?)
        # Anziché fare un ciclo for e calcolare la componente nelle singole iterazioni,
        # mi conviene vettorizzare tutto. Per questo, creo un vettore aventi valori
        # che vanno da [0, seq_length)
        position = np.arange(0, seq_length)
        #https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        # Ora preparo il denominatore degli argomenti delle funzioni trigonometriche.
        # È stato riscritto sfruttando il trucco e^ln(x) per una maggiore stabilità numerica
        denominator = np.exp(np.arange(0, d_model, 2) * -(np.log(10000) / d_model))
        # calcolo il seno nelle posizioni pari
        # (l'index strano sta dicendo di ignorare il n° di righe e operare a partire
        # dalla colonna 0, operando ogni due colonne, quindi una si e una no)
        self.pe_matrix[:, 0::2] = np.sin(position * denominator)
        self.pe_matrix[:, 1::2] = np.cos(position * denominator)

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
        # => [[ciao], [sono], [Manu], [EOS], [PADDING]]; [[Che], [schifo], [il], [pininfarina], [EOS]] = 2 sequenze diverse (2 x 5 x 512)
        # Quella singola dimensione si aggiunge per avere la possibilità di accumulare frasi
        self.pe_matrix = Tensor(np.expand_dims(self.pe_matrix, axis=0))
        #self.pe_matrix = Tensor(self.pe_matrix, requires_grad=True).unsqueeze(0)

        # questo layer non è necessario salvarlo perché non ha valori trainati. Basta
        # segnare che c'è e ricrearlo da zero (PyTorch lo fa tramite "register_buffer")

    def forward(self, x: Tensor) -> Tensor:
        # aggiungo il pos_encoding vector all'input corrispondente
        # pe_matrix: 1 (dummy dim) x seq_len (n° righe) x d_model (n° colonne)
        # x: 1 x embedding_size
        # Non capisco l'indexing... Aspetto a sbatterci ancora la testa, magari lo corregge più tardi
        # Tensor + Tensor
        x = x + self.pe_matrix[:, :x.shape[1], :]
        return self.dropout(x)
    
class LayerNormalization:
    '''
    https://www.youtube.com/watch?v=G45TuC6zRf4
    Fa esattamente ciò che promette il nome: normalizzare i valori
    per far sì che non si smaiali il training.
    (Presumibilmente farà rientrare i valori in un range [-1, 1]).

    Prende un embeding (X) dalla matrice, sottrae la media (vanilla, quindi somma
    delle componenti diviso numero delle componenti) dei suoi valori (M) e dividie per
    la std deviation (G). Questo viene moltiplicato per un coefficiente uguale
    per tutto il layer e sommato per un "bias", anch'esso uguale per tutto questo layer.
    y = k*[ (Xi-Mi)/(Gi^2+eps) ] + b
    Nota che M può essere riciclata nel calcolo di G.
    eps si usa nella formula della std dev per evitare che sia zero (visto che è un
    denominatore sarebbe un problemone)
    '''
    def __init__(self, d_model: int, eps: float = 1e-12) -> None:
        self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.bias = Tensor(np.zeros(d_model), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        # x è (batch_dim x 1 x embedding_size). Quel "axis=-1" indica che voglio
        # la somma sull'ultima dim, quindi in embedding_size, e keepdims=True
        # è per far sì che non perdo la batch_dim, che è 1
        # NOTE: MEAN_BACKWARD RICHIEDE FIX PER WORKARE SU DIM
        mean = np.mean(x, axis=-1, keepdims=True)
        # sqrt( (x[0]-mean)^2 + (x[1]-mean)^2 + ... (x[i]-mean)^2)
        # NOTE: RICHIEDE BACKWARD
        std = np.std(x, axis=-1, keepdims=True)
        # (x - mean) / sqrt(std^2 + eps)
        return self.gamma * ((x - mean) / np.sqrt(np.pow(std, 2) + self.eps)) + self.bias
    

class FeedForwardBlock:
    '''
    Non è altro che un layer con ReLU + dropout + layer senza activation function:
    Dropout ( ReLU(x @ W1 + b1) ) @ W2 + b2
    Alla fine della computazione, il layer restituisce un vettore delle
    stesse dimensioni di quello di input, quindi della dimensione degli embedding.
    Può essere vista come una mini rete neurale vanilla
    '''

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        # d_model x d_ff, nuovo parametro di questo layer
        # d_ff = 2048 nel paper
        self.linear_layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # (batch x seq_len x d_model) -> (batch x seq_len x d_model)
        return self.linear_layer2( self.dropout( self.linear_layer1(x) ) )
    

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
        self.head_dim = d_model / num_heads             # d_k
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

    # e se la rendo una nested function di forward? In questo modo posso fare x._device
    # ed evitare di passare tutti questi parametroni enormi. Tanto questa funzione non
    # la uso al di fuori del forward...
    # Scaled Dot-Product Attention
    def attention(self, s_q: Tensor, s_k: Tensor, s_v: Tensor, mask=None) -> tuple[Tensor, Tensor]:
        # NOTE: s_ sta per splitted
        device = s_q._device

        # formula applicata para para
        # (batch x head_num x seq_len x head_dim) -> (batch x head_num x seq_len x seq_len)
        attention_scores = (s_q @ s_k.transpose(-2, -1)) / device.sqrt(self.head_dim)
        if mask is not None:
            # rimpiazzo con un valore piccolo; ci penserà poi la softmax a farlo
            # tendere a 0. Questo, ad esempio, sarà usato per i [PADDING] embedding
            # perché non devono interagire con il resto della frase.
            # NOTE: posso evitare di calcolare questa attention se è solo [PADDING]?
            # NOTE: non ha detto nulla riguardo la possibilità di mascherare
            # post softmax direttamente con 0. Perché? Ha solo seguito il paper
            # o c'è un motivo?
            attention_scores = device.where(mask == 0, -1e9) # rimpiazza con ~ -inf

        # softmax solo sull'ultima dimensione
        # (batch x head_num x seq_len x seq_len)
        attention_scores = self.softmax(attention_scores)

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        # se vuoi visuallizzare gli scores, devi farlo prima di questa matmul
        return (attention_scores @ s_v)

    # NOTE: questa è la parte che può causare più bug di tutte.
    # Ho dovuto aggiungere qualche metodo alla classe tensor e
    # non ho potuto usare le stesse funzioni di Jamir
    # In teoria, reshape e transpose non dovrebbero aver bisogno di backward...
    def forward(self, q, k, v, mask=None):
        '''
        mask: utilizzato per fare in modo che alcune parole non interagiscano tra di loro.
            Si imposta l'entrata della matrice ad un numero molto piccolo in modo che la
            softmax la setti a 0.
        '''
        q = self.query_layer(q) # Q', stesse dim di x (batch x seq_len x d_model)
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

        # (batch x seq_len x d_model) -> (batch x seq_len x head_num x head_dim) -> .T -> (batch x head_num x seq_len x head_dim)
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

        return tensor.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)
    
    def concat(self, tensor: Tensor) -> Tensor:
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head_num, seq_len, head_dim]
        :return: [batch_size, seq_len, d_model]
        """
        batch_size, head_num, length, head_dim = tensor.size()
        d_model = head_num * head_dim

        return tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
    
# ---- quasilayers ----
    
# torch la chiama TransformerEmbeddings
class Embeddings:
    def __init__(self, d_model: int, vocab_size: int, max_len: int, dropout: float):
        self.token_embeddings = TokenEmbeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # secondo la logica di Jamil verrebbe cosi', MA...
        return self.pos_encoding(self.token_embeddings(x))
        # ...puoi anche fare return self.dropout(self.token_emb(x) + self.pos_encoding(x))
        # e alleggerire il forward di pos_enc


# Equivalente dell'encoder block di Jamil
# LAYER perche' sono di fila, NON in parallelo!
class EncoderLayer:

    def __init__(self, d_model: int, num_heads: int, dim_feed_forward: int, drop_prob: float) -> None:
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(drop_prob)
        self.norm1 = LayerNormalization()

        self.ffb = FeedForwardBlock(d_model=d_model, d_ff=dim_feed_forward, dropout=drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.norm2 = LayerNormalization()
    
    def forward(self, x: Tensor, src_mask: Tensor=None) -> Tensor:
        residual_x = x
        x = self.dropout1(self.self_attention(q=x, k=x,v=x, mask=src_mask))
        x = self.norm1(residual_x + x)

        residual_x = x
        x = self.dropout2(self.ffb(x))
        x = self.norm2(residual_x + x)

        return x


class DecoderLayer:
    def __init__(self, d_model: int, num_heads: int, dim_feed_forward:int, drop_prob: float) -> None:
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(drop_prob)
        self.norm1 = LayerNormalization()

        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout2 = nn.Dropout(drop_prob)
        self.norm2 = LayerNormalization()

        self.ffb = FeedForwardBlock(d_model=d_model, d_ff=dim_feed_forward, dropout=drop_prob)
        self.dropout3 = nn.Dropout(drop_prob)
        self.norm3 = LayerNormalization()

    def forward(self, x: Tensor, enc_out: Tensor, src_mask: Tensor=None, trg_mask: Tensor=None) -> Tensor:
        '''
        src_mask: eventuale mask dell'encoder
        trg_mask: mask del decoder
        '''
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

class Encoder:
    def __init__(self, d_model: int, enc_vocab_size: int, max_len: int, drop_prob: float, n_layers: int, num_heads: int, dim_feed_forward: int):
        self.emb = Embeddings(d_model, enc_vocab_size, max_len, drop_prob)
        self.layers = [EncoderLayer(d_model, num_heads, dim_feed_forward, drop_prob) for _ in range(n_layers)]

    def forward(self, x: Tensor, src_mask: Tensor = None) -> Tensor:
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x
    
class Decoder:
    def __init__(self, d_model: int, dec_vocab_size: int, max_len: int, drop_prob: float, n_layers: int, num_heads: int, dim_feed_forward: int):
        self.emb = Embeddings(d_model, dec_vocab_size, max_len, drop_prob)
        self.layers = [DecoderLayer(d_model, num_heads, dim_feed_forward, drop_prob) for _ in range(n_layers)]
        #self.linear = nn.Linear(d_model, dec_vocab_size)
    
    def forward(self, y: Tensor, enc_out: Tensor, src_mask: Tensor=None, trg_mask: Tensor=None) -> Tensor:
        y = self.emb(y)

        for layer in self.layers:
            y = layer(y, enc_out, src_mask, trg_mask)
        return y
    

class Transformer():
    def __init__(self,
                 d_model: int,
                 encoder_vocab_size: int,
                 dec_vocab_size: int,
                 max_len: int,
                 drop_prob: float,
                 n_layers: int,
                 num_heads: int,
                 dim_feed_forward: int) -> None:
        self.encoder = Encoder(d_model, encoder_vocab_size, max_len, drop_prob, n_layers, num_heads, dim_feed_forward)
        self.decoder = Decoder(d_model, dec_vocab_size, max_len, drop_prob, n_layers, num_heads, dim_feed_forward)
        self.linear = nn.Linear(d_model, dec_vocab_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, src: Tensor, trg: Tensor, src_mask, trg_mask) -> Tensor:
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, src, enc_src, trg_mask)
        output = self.linear(output)
        # chiamo la softmax sulla dim di dec_vocab_size perche' voglio
        # la prob del next token
        return self.softmax(output)

# NOTE: torch ha una sua nomenclatura per ciascuno di essi (che l'indiano
# simpatico ha seguito)! Copiala a fine lavoro
# Prima segui Jamil per costruirti un quadro completo e coerente del funzionamento
# del modello, DOPO vedi se si puo' smussare e se vale la pena scrivere delle
# attention diverse per encoder e decoder

# NOTE: encoder, decoder e le altre componenti puoi farle ereditare da sequential?

# https://github.com/ajhalthor/Transformer-Neural-Network/blob/main/transformer.py
# https://github.com/hyunwoongko/transformer/blob/master/models/layers/multi_head_attention.py#L41
# https://github.com/hyunwoongko/transformer/blob/master/models/blocks/encoder_layer.py

# c'e' un po' di riscrittura da fare. Molto codice lavora con np di default e molti
# forward avvengono senza Tensor, cosa che non permette di fare la backward.
# Setta anche il requires_grad=True dove serve.
# Ricorda: basta solo che tutte le operazioni che avvengono in un forward abbiano
# un backward. Il resto puoi farlo senza passare per i tensor
# Controlla se nei vari forward appaiono nuove funzioni che devono essere
# implementate nei Tensor