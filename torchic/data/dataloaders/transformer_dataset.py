from torchic.dtypes import uint8, uint16, int64
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import cat
from torchic.data.dataloaders.base_classes import Dataset
import numpy as np
from torchic.tools.sequences import causal_mask, create_pad_mask


# "Dataset stores the samples and their corresponding labels, and DataLoader
# wraps an iterable around the Dataset to enable easy access to the samples."
# la classe Dataset e' solo una classe astratta senza attributi e metodi,
# quindi se non ti serve non la fare.
# DataLoader


# ne hai gia' una in Torchic.dataset, ma non va bene
# Avere una classe Dataset (che e' solo un wrapper)
# aiuta perche standardizza. In particolare, ho bisogno
# di leggere bene la len e iterare con getitem


DTYPE = uint16 #int64


class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_trg, src_lang: str, trg_lang: str,
                 seq_len: int, dynamic_masking: bool=True) -> None:
        super().__init__(dataset)

        self.tok_src = tokenizer_src
        self.tok_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len
        self.dynamic_masking = dynamic_masking
        
        # src e trg tokenizer condividono gli stessi token speciali
        # devo specificare il dtype
        # perche' sono dichiarati come vettori e non come scalari??
        # Per ottimizzazione, infatti ti conviene rifarlo (ma esplicitalo
        # usando .view())
        self.sos_token = Tensor([tokenizer_src.token_to_id("[SOS]")], dtype=DTYPE)
        self.eos_token = Tensor([tokenizer_src.token_to_id("[EOS]")], dtype=DTYPE)
        self.pad_token = Tensor([tokenizer_src.token_to_id("[PAD]")], dtype=DTYPE)

    def __len__(self):
        return len(self.dataset) 
    
    # precomputare tutta sta roba aiuterebbe a gestire bene la memoria?
    def __getitem__(self, index) -> dict[str: Tensor]:
        src_target_pair = self.dataset[index]

        src_text = src_target_pair[self.src_lang]
        trg_text = src_target_pair[self.trg_lang]

        # splitta il testo in token e restituisce una lista
        # contenente gli id (numeri, no ohe) corrispondenti
        enc_input_tokens = Tensor(self.tok_src.encode(src_text).ids, dtype=DTYPE)
        dec_input_tokens = Tensor(self.tok_trg.encode(trg_text).ids, dtype=DTYPE)

        if not self.dynamic_masking:
            # calcolo il numero di [PAD] tokens che devo aggiungere
            # per arrivare a seq_len (togliendo dal conto [SOS] e [EOS])
            n_pad_tokens_enc = self.seq_len - len(enc_input_tokens) -2
            # qua invece e' solo -1 perche' [EOS] non e' presente nell'
            # input del decoder (perche'?)
            n_pad_tokens_dec = self.seq_len - len(dec_input_tokens) -1

            # verifico che la seq_len scelta sia sufficiente per rappresentare
            # la frase piu' lunga in tutto il ds
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
            
            # aggiungo [SOS], [EOS] e [PAD] fino a raggiungere la seq_len
            encoder_input = cat(
                    [
                        self.sos_token,
                        enc_input_tokens,
                        self.eos_token,
                        self.pad_token.expand(n_pad_tokens_enc, )
                        #Tensor([self.pad_token.data] * n_pad_tokens_enc, dtype=DTYPE)
                        # da modificare
                        #Tensor(self.pad_token._device.tile(self.pad_token.data, n_pad_tokens_enc), dtype=DTYPE)
                    ],
                dtype=DTYPE)
            
            decoder_input = cat(
                    [
                        self.sos_token,
                        dec_input_tokens,
                        self.pad_token.expand(n_pad_tokens_dec, )
                        #Tensor([self.pad_token.data] * n_pad_tokens_dec, dtype=DTYPE)
                        #Tensor(self.pad_token._device.tile(self.pad_token.data, n_pad_tokens_dec), dtype=DTYPE)
                    ],
                dtype=DTYPE)
            
            # frase che voglio dal decoder (si omette [SOS] perche' quello sara'
            # il primo token di partenza cui cui far partire l'autoregressione)
            target = cat(
                    [
                        dec_input_tokens,
                        self.eos_token,
                        self.pad_token.expand(n_pad_tokens_dec, )
                        # e se facessi un broadcast?
                        #Tensor([self.pad_token.data] * n_pad_tokens_dec, dtype=DTYPE)
                        #Tensor(self.pad_token._device.tile(self.pad_token.data, n_pad_tokens_dec), dtype=DTYPE)
                    ],
                dtype=DTYPE)

            # creo le maschere
            encoder_mask = create_pad_mask(encoder_input, self.pad_token)

            # decoder mask: maschera i [PAD] e nasconde i token successivi alla
            # parola coinvolta in modo tale che il transformer non impari a sfruttare le parole successive
            # per ponderare quella attuale
            decoder_mask = create_pad_mask(decoder_input, self.pad_token)
            # maschero dove i token sono di padding e dove i token sono "futuri"
            # (1, seq_len, seq_len) & (1, 1, seq_len) -> (1, seq_len, seq_len)
            decoder_mask = causal_mask(decoder_input, dtype=DTYPE) & decoder_mask
        else:
            # i PAD saranno aggiunti in fase di batching
            encoder_input = cat([self.sos_token, enc_input_tokens, self.eos_token], dtype=DTYPE)
            decoder_input = cat([self.sos_token, dec_input_tokens], dtype=DTYPE)
            target = cat([dec_input_tokens, self.eos_token], dtype=DTYPE)
            # da creare in fase di batching
            encoder_mask = None
            decoder_mask = None

        # le maschere sono tensori di bool
        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": encoder_mask, # (1, 1, seq_len)
            "decoder_mask": decoder_mask, # (1, seq_len) & (1, seq_len, seq_len)
            "target": target, # (seq_len)
            "src_text": src_text,
            "trg_text": trg_text
        }
    


    
# https://youtu.be/ISNdQcPhsts?feature=shared&t=5734 
# https://youtu.be/ISNdQcPhsts?feature=shared&t=5948
# https://youtu.be/ISNdQcPhsts?feature=shared&t=6372
# https://youtu.be/ISNdQcPhsts?feature=shared&t=6446
# https://youtu.be/ISNdQcPhsts?feature=shared&t=6478
# https://youtu.be/ISNdQcPhsts?feature=shared&t=7623