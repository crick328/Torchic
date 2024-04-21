'''
Ci sono un po' di sanity check da fare.
Innanzitutto, pulisci il codice da tutti questi commenti estensivi.
Copia e organizza ciò che serve sul doc di Replica.
Secondo, bisogna controllare che le backward funzionino a dovere.
'''

# parole > numeri tramite words embeddings
# positional_encoding => prendo i valori della rete di word embeddings e gli do in input a delle
# funzioni seno e coseno che diventano sempre più larghe per ogni parola (tipo cos(output[0], sin(output[1]), cos(2*output[3]), ecc.)).
# Sommo gli output delle funzioni trigonometriche agli output della word embeddings per ottenere le parole fatte e finite.
# Ottenuti questi nuovi valori, faccio un'altra matmul con i weights, ma stavolta non sommo il bias e non uso la act_fun.
# Questi si chiamano Query Values. Si fa lo stesso processo per calcolare le Keys, valori che consentono di calcolare
# la similarità del significato della stessa parola in contesti diversi grazie al dot product


from replica import CudaTensor, nn
import numpy as np


class InputEmbeddings:
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

    def forward(self, x):
        # x è un vettore OHE
        # nel paper si moltiplica per la radice di d_model
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
        self.pe_matrix = np.expand_dims(self.pe_matrix, axis=0)

        # questo layer non è necessario salvarlo perché non ha valori trainati. Basta
        # segnare che c'è e ricrearlo da zero (PyTorch lo fa tramite "register_buffer")

    def forward(self, x):
        # aggiungo il pos_encoding vector all'input corrispondente
        # pe_matrix: 1 (dummy dim) x seq_len (n° righe) x d_model (n° colonne)
        # x: 1 x embedding_size
        # Non capisco l'indexing... Aspetto a sbatterci ancora la testa, magari lo corregge più tardi
        x = CudaTensor(x + self.pe_matrix[:, :x.shape[1], :], requires_grad = False)
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
    def __init__(self, eps: float = 10**-6) -> None:
        self.eps = eps
        # scalari comuni per qualsiasi input
        self.gamma = CudaTensor(1)
        self.bias = CudaTensor(0)

    def forward(self, x):
        # x è (batch_dim x 1 x embedding_size). Quel "axis=-1" indica che voglio
        # la somma sull'ultima dim, quindi in embedding_size, e keepdims=True
        # è per far sì che non perdo la batch_dim, che è 1
        mean = np.mean(x, axis=-1, keepdims=True)
        # sqrt( (x[0]-mean)^2 + (x[1]-mean)^2 + ... (x[i]-mean)^2)
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
        self.linear_layer1 = nn.LLayer(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.LLayer(d_ff, d_model)

    def forward(self, x) -> CudaTensor:
        # (batch x seq_len x d_model) -> (batch x seq_len x d_model)
        return self.linear_layer2( self.dropout( self.linear_layer1(x) ) )
    

class MultiHeadAttention:
    # puoi parallelizzarla! Assegna un process a ciascuno dei tre forward
    '''
    Il vettore che riceve in input (embedding vector di dimensione n x seq_len x dim) viene
    striplicato e prende il nome di query, key e value vector. Ognuno di
    questi vettori passa per un suo linear layer (indipendente) e il risultato viene diviso
    in d_model / h vettori. Nel paper, le attention head sono 8, quindi ciascun risultato
    viene diviso in 64 vettori. Da qui si applica la formula dell'attention, si concatenano
    i risultati e il vettore trovato passa per un ultimo linear layer.

    h: number of self attention heads
    '''
    def __init__(self, d_model: int, h: int, dropout: int):
        if d_model % h != 0:
            ValueError(f"Cannot divide evenly d_model of {d_model} times h {h} head attentions") 
        # dimensione della frammentazione dei singoli embeddings. Ogni attention head
        # prende una sottoparte di esso e hanno tutte le stesse dimensioni (batch x seq_len x d_k)
        self.head_number = h
        self.head_dim = d_model / h                     # d_k
        self.query_layer = nn.LLayer(d_model, d_model)  # Wq
        self.key_layer = nn.LLayer(d_model, d_model)    # Wk
        self.value_layer = nn.LLayer(d_model, d_model)  # Wv
        self.output_layer = nn.LLayer(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax()

    def attention(self, splitted_q: CudaTensor.data, splitted_k: CudaTensor.data, splitted_v: CudaTensor.data, mask=None):
        # scritto per chiarezza, ma puoi gestirlo come vuoi, l'importante è che
        # il device dei tensor.data e della radice coincidano
        device = splitted_q._device

        # formula applicata para para
        attention_scores = (splitted_q @ splitted_k.transpose(-2, -1)) / device.sqrt(self.head_dim)
        if mask:
            # rimpiazzo con un valore piccolo; ci penserà poi la softmax a farlo
            # tendere a 0. Questo, ad esempio, sarà usato per i [PADDING] embedding
            # perché non devono interagire con il resto della frase.
            # NOTE: posso evitare di calcolare questa attention se è solo [PADDING]?
            # NOTE: non ha detto nulla riguardo la possibilità di mascherare
            # post softmax direttamente con 0. Perché? Ha solo seguito il paper
            # o c'è un motivo?
            attention_scores = device.where(mask == 0, -1e9) # rimpiazza con ~ -inf

        # devo fare la softmax solo sull'ultima dim, ma la mia non lo supporta...

    # NOTE: questa è la parte che può causare più bug di tutte.
    # Ho dovuto aggiungere qualche metodo alla classe tensor e
    # non ho potuto usare le stesse funzioni di Jamir
    def forward(self, x, q, k, v, mask=None):
        '''
        mask: utilizzato per fare in modo che alcune parole non interagiscano tra di loro.
            Si imposta l'entrata della matrice ad un numero molto piccolo in modo che la
            softmax la setti a 0.
        '''
        # nel video prende q, k, v. Perché non prendere x visto che è lo stesso?
        query = self.query_layer(x) # Q', stesse dim di x (batch x seq_len x d_model)
        key = self.key_layer(x)     # K'
        value = self.value_layer(x) # V'
        
        # l'equivalente di torch.tensor.view è reshape, ma
        # forse fa una copia dell'array e forse no.
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
        # NOTE: occhio che potrebbe essere necessario ricastarlo a CudaTensor
        # NOTE: potrei semplicemente chiamare x.data.shape in tutti e tre perché
        # hanno le stesse shape, ma così è più chiaro
        query = query.data.reshape(query.data.shape[0], query.data.shape[1], self.head_number, self.head_dim).transpose(1, 2)
        key = key.data.reshape(key.data.shape[0], key.data.shape[1], self.head_number, self.head_dim).transpose(1, 2)
        value = value.data.reshape(value.data.shape[0], value.data.shape[1], self.head_number, self.head_dim).transpose(1, 2)

        x, attention_scores = self.attention(query, key, value, mask)