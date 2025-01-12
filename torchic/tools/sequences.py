from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import cat, max


def pad_sequence(seq: Tensor, target_len: int, pad_token: Tensor) -> Tensor:
    '''
    Returns the given sequence with enough padding to reach the target_len
    '''
    seq_len = len(seq)
    if seq_len < target_len:
        return cat((seq, pad_token.expand(target_len - seq_len, )))
    return seq


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, use_dec_mask: bool=True):
    # si puo' prendere dal src o tgt, tanto e' uguale per entrambi
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    logits_history = []

    # ottengo un output processato solo dall'encoder
    encoder_output = model.encoder(source, source_mask)
    # vettore di size (batch_size, seq_len) che verra' creato iterazione per
    # iterazione dal decoder. La batch_size e' 1 per definizione, mentre
    # la seq_len e' la dimensione in cui aggiungo tokens
    decoder_input = Tensor([[sos_idx]])
    decoder_input.data = Tensor._to_dtype(decoder_input.data, source.dtype)

    # do-while
    while True:
        # voglio che veda un nuovo token per volta, quindi maschero i
        # token "futuri" (senza mask, )
        if use_dec_mask:
            decoder_mask = Tensor(causal_mask(decoder_input, 1))
            decoder_mask.data = Tensor._to_dtype(decoder_mask.data, source_mask.dtype)
        else:
            decoder_mask = None

        out = model.decoder(decoder_input, encoder_output, source_mask, decoder_mask)

        # leggo il prossimo token
        prob = model.linear(out[:, -1])
        logits_history.append(prob)

        # per definizione di greedy search, prendo il prossimo token piu' probabile
        # secondo il modello
        _value, next_token = max(prob, dim=1)
        decoder_input = cat([decoder_input, Tensor([next_token])], dim=1)

        # se raggiungo la seq_len massima
        if decoder_input.shape()[1] == max_len:
            break
        # se il modello decide di terminare la frase
        if next_token == eos_idx:
            break

    raw_output = cat(logits_history, dim=0)
    # rimuovo la dimension della batch_size
    #print(f"DECODER IN: {decoder_input.data}, {decoder_input.shape()}")
    return decoder_input.squeeze(0), raw_output



def create_pad_mask(padded_sequence: Tensor, pad_token: Tensor) -> Tensor:
    '''
    Given a padded sequence, return a boolean mask having False where
    a PAD token is present, True elsewhere.

    The idea is to mark with True the values that can participate to
    the attention scores.

    Example::
        >>> pad_token =  Tensor([3])
        >>> tensor = Tensor([1, 26, 345, 2, pad_token, pad_token])
        >>> mask = create_pad_mask(tensor, pad_token)
        >>> mask
        Tensor([True, True, True, True, False, False])
    '''
    pad_mask = padded_sequence != pad_token
    # aggiungo le dim di batch e num_head (per il broadcast)
    return pad_mask.unsqueeze(0).unsqueeze(0)


# impedisce di costruire la self attention con parole che devono ancora
# esserci nella frase. Questo si fa con una matrice avente tutti zeri
# sopra la diagonale
# Cambiala in modo che non prenda un tensor in input perche'
# e' fuorviante
def causal_mask(tensor: Tensor, dim=0, dtype=None) -> Tensor:
    '''
    Questa funzione genera una matrice triangolare superiore di False
    **da sopra la diagonale**.

    Example::
        >>> mask = casual_mask((1,4,4))
        >>> mask
        Tensor([[[ True, False, False, False],
                 [ True,  True, False, False],
                 [ True,  True,  True, False],
                 [ True,  True,  True,  True]]])
    
    Note: per convenzione, in generale nel deep learning si usa 0 per mascherare un
    elemento e 1 (True) per consentirne il calcolo
    '''
    size = tensor.size(dim)
    mask = Tensor(tensor._device.triu( tensor._device.ones([1, size, size]), k=1), dtype=dtype)
    # restituisco una matrice di valori bool
    return mask == 0