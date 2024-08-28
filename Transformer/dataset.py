import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.1")
import torchic
from torchic import Tensor
from torchic.tools.helper import cat
from torchic.datasets.base_classes import Dataset
import numpy as np


# "Dataset stores the samples and their corresponding labels, and DataLoader
# wraps an iterable around the Dataset to enable easy access to the samples."
# la classe Dataset e' solo una classe astratta senza attributi e metodi,
# quindi se non ti serve non la fare.
# DataLoader


# ne hai gia' una in Torchic.dataset, ma non va bene
# Avere una classe Dataset (che e' solo un wrapper)
# aiuta perche standardizza. In particolare, ho bisogno
# di leggere bene la len e iterare con getitem





class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len) -> None:
        super().__init__(dataset)
        #self.dataset = dataset

        self.tok_src = tokenizer_src
        self.tok_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len
        
        # src e trg tokenizer condividono gli stessi token speciali
        # devo specificare il dtype
        self.sos_token = Tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torchic.int64)
        self.eos_token = Tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torchic.int64)
        self.pad_token = Tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torchic.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index): #-> dict[Tensor]:
        src_target_pair = self.dataset[index]

        src_text = src_target_pair["translation"][self.src_lang]
        trg_text = src_target_pair["translation"][self.trg_lang]

        # splitta il testo in token e restituisce una lista
        # contenente gli id (numeri, no ohe) corrispondenti
        enc_input_tokens = self.tok_src.encode(src_text).ids
        dec_input_tokens = self.tok_trg.encode(trg_text).ids

        # calcolo il numero di [PAD] tokens che devo aggiungere
        # per arrivare a seq_len (togliendo dal conto [SOS] e [EOS])
        n_pad_tokens_enc = self.seq_len - len(enc_input_tokens) -2
        # qua invece e' solo -1 perche' [EOS] non e' presente nell'
        # input del decoder (perche'?)
        n_pad_tokens_dec = self.seq_len - len(dec_input_tokens) -1

        # verifico che la seq_len scelta sia sufficiente per rappresentare
        # persino la frase piu' lunga in tutto il ds, senno' do'
        # errore (perche? credo che non ha modo di fare collegamenti tra
        # frasi diverse)
        if n_pad_tokens_enc < 0:
            raise RuntimeError(
                f"The seq_len chosen ({self.seq_len}) isn't enough to fit the "
                f"sentence {src_text} to give to the encoder."
                )
        if n_pad_tokens_dec < 0:
            raise RuntimeError(
                f"The seq_len chosen ({self.seq_len}) isn't enough to fit the "
                f"sentence {trg_text} to give to the decoder."
                )
        
        # preparo la frase, quindi aggiungo [SOS], [EOS] e [PAD] fino a raggiungere
        # la seq_len
        encoder_input = cat(
                [
                    self.sos_token.data,
                    enc_input_tokens,
                    self.eos_token.data,
                    #[self.pad_token.data] * n_pad_tokens_enc
                    self.pad_token._device.tile(self.pad_token.data, n_pad_tokens_enc)
                ],
            dtype=torchic.int64)
        
        decoder_input = cat(
                [
                    self.sos_token.data,
                    dec_input_tokens,
                    #self.eos_token.data,
                    #[self.pad_token.data] * n_pad_tokens_dec
                    self.pad_token._device.tile(self.pad_token.data, n_pad_tokens_dec)
                ],
            dtype=torchic.int64)
        
        # frase che voglio dal decoder (non ho capito perche' e'
        # uguale all'input del decoder ma con [EOS] anziche' [SOS])
        target = cat(
                [
                    dec_input_tokens,
                    self.eos_token.data,
                    #[self.pad_token.data] * n_pad_tokens_dec
                    self.pad_token._device.tile(self.pad_token.data, n_pad_tokens_dec)
                ],
            dtype=torchic.int64)
        

        assert encoder_input.size() == self.seq_len
        assert decoder_input.size() == self.seq_len
        assert target.size() == self.seq_len

        # encoder mask: serve a mascherare i token di [PAD] in modo
        # tale che non facciano parte della self attention. Faranno comunque
        # parte dei calcoli, ma non vengono relazionati agli altri tokens.

        # bool mask con False se token = pad, else True e converto
        # da bool a 1 e 0 con il cast ad int64 
        encoder_mask = Tensor(encoder_input.data != self.pad_token, dtype=torchic.int64)
        # aggiungo le dim di batch e d_model (non sicuro di quest'ultimo)
        encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)


        # decoder mask: anch'essa maschera i [PAD], ma in piu' nasconde
        # anche i token successivi alla parola coinvolta in modo tale
        # che il transformer non impari a sfruttare le parole successive
        # per ponderare quella attuale, ma che si appoggi solo su quelle
        # gia' viste perche' in inference sara' cosi'

        # bool mask con False se token = pad, else True e
        # converto da bool a 1 e 0
        decoder_mask = Tensor(decoder_input.data != self.pad_token, dtype=torchic.int64)
        # aggiungo le dim di batch e d_model (non sicuro di quest'ultimo)
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)
        decoder_mask.data = decoder_mask.data & causal_mask(decoder_input.size(0))
        

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": encoder_mask, # (1, 1, seq_len)
            "decoder_mask": decoder_mask, # (1, seq_len) & (1, seq_len, seq_len)
            "label": target, # (seq_len)
            "src_text": src_text,
            "trg_text": trg_text
        }
    


# impedisce di costruire la self attention con parole che devono ancora
# esserci nella frase. Questo si fa con una matrice avente tutti zeri
# sopra la diagonale
def causal_mask(size: int):
    mask = Tensor(np.triu( np.ones([1, size, size]) ), dtype=torchic.int64)
    # restituisco una matrice di valori bool
    return mask.data == 0
    
    # https://youtu.be/ISNdQcPhsts?feature=shared&t=5734 
    # https://youtu.be/ISNdQcPhsts?feature=shared&t=5948
    # https://youtu.be/ISNdQcPhsts?feature=shared&t=6372
    # https://youtu.be/ISNdQcPhsts?feature=shared&t=6446
    # https://youtu.be/ISNdQcPhsts?feature=shared&t=6478
    # https://youtu.be/ISNdQcPhsts?feature=shared&t=7623