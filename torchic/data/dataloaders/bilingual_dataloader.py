from torchic.dtypes import uint8, uint16, int64
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import cat
from torchic.data.dataloaders.base_classes import Dataset, DataLoader
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


# alla fine questa classe e' necessaria perche' non ho pretokenizzato
class BilingualDataset(Dataset):
    '''
    Class designed for translation tasks.

    dynamic padding: instead of always reaching seq_len, it adds PAD tokens
        until it reaches the longest sentence in the current batch.
    '''
    def __init__(self, dataset, tokenizer_src, tokenizer_trg, src_lang: str, trg_lang: str,
                 seq_len: int, dynamic_padding: bool=True) -> None:
        super().__init__(dataset)

        self.tok_src = tokenizer_src
        self.tok_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len
        self.dynamic_padding = dynamic_padding
        
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

        # calcolo il numero di [PAD] tokens che devo aggiungere
        # per arrivare a seq_len (togliendo dal conto [SOS] e [EOS])
        n_pad_tokens_enc = self.seq_len - len(enc_input_tokens) -2
        if n_pad_tokens_enc < 0:
            raise RuntimeError(f"Encoder sequence exceeds seq_len.")

        # qua invece e' solo -1 perche' [EOS] non e' presente nell'
        # input del decoder (perche'?)
        n_pad_tokens_dec = self.seq_len - len(dec_input_tokens) -1
        if n_pad_tokens_dec < 0:
            raise RuntimeError("Decoder sequence exceeds seq_len.")

        if not self.dynamic_padding:
            # aggiungo [SOS], [EOS] e [PAD] fino a raggiungere la seq_len
            encoder_input = cat(
                    [
                        self.sos_token,
                        enc_input_tokens,
                        self.eos_token,
                        self.pad_token.expand(n_pad_tokens_enc, )
                    ],
                dtype=DTYPE)
            
            decoder_input = cat(
                    [
                        self.sos_token,
                        dec_input_tokens,
                        self.pad_token.expand(n_pad_tokens_dec, )
                    ],
                dtype=DTYPE)
            
            # frase che voglio dal decoder (si omette [SOS] perche' quello sara'
            # il primo token di partenza cui cui far partire l'autoregressione)
            target = cat(
                    [
                        dec_input_tokens,
                        self.eos_token,
                        self.pad_token.expand(n_pad_tokens_dec, )
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
            # => aggiungo solo [SOS] e [EOS]
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
    


from torchic.tools.sequences import causal_mask, pad_sequence
from torchic.tools.helpers import cat, ones, full
from torchic.tools.helpers import stack
import random

class BilingualDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool=True, my_stack=True) -> None:
        super().__init__(dataset, batch_size, shuffle)
        self.my_stack = my_stack

    @staticmethod
    def stack_on_batch_dim(dict_list: list[dict[str, Tensor]], dynamic_padding, my_stack, pad_token: Tensor):
        """
        Combina un elenco di dizionari su una nuova dimensione batch e genera maschere opzionali.
        
        Args:
            dict_list: Elenco di dizionari contenenti tensori con chiavi comuni.
            dynamic_padding: Booleano che abilita o disabilita il padding dinamico.
            pad_token: Token utilizzato per il padding.
        
        Returns:
            Un dizionario combinato con eventuali maschere.
        """
        # estraggo i dati
        encoder_input = [dct['encoder_input'] for dct in dict_list]
        decoder_input = [dct['decoder_input'] for dct in dict_list]
        target = [dct['target'] for dct in dict_list]

        if dynamic_padding:
            # calcolo le lunghezze massime della batch
            max_enc_len = max(len(seq) for seq in encoder_input)
            max_dec_trg_len = max(max(len(seq) for seq in decoder_input), max(len(seq) for seq in target))

            if my_stack:
                encoder_input = stack(
                    pad_sequence(seq, max_enc_len, pad_token) for seq in encoder_input
                )
                decoder_input = stack(
                    pad_sequence(seq, max_dec_trg_len, pad_token) for seq in decoder_input
                )
                target = stack(
                    pad_sequence(seq, max_dec_trg_len, pad_token) for seq in target
                )
            else:
                # creo delle matrici di [PAD] in modo da applicare
                # in batch il padding alle sequenze
                batch_size = len(encoder_input)         # (la posso prendere da self)
                device = encoder_input[0].device
                dtype = encoder_input[0].dtype

                #encoder_input_pad = pad_token * ones((batch_size, max_enc_len), dtype=dtype, device=device)
                #decoder_input_pad = pad_token * ones((batch_size, max_dec_trg_len), dtype=dtype, device=device)
                #target_pad = pad_token * ones((batch_size, max_dec_trg_len), dtype=dtype, device=device)

                # mi creo delle matrici piene di [PAD]
                encoder_input_pad = full((batch_size, max_enc_len), pad_token, dtype=dtype, device=device)
                decoder_input_pad = full((batch_size, max_dec_trg_len), pad_token, dtype=dtype, device=device)
                target_pad = full((batch_size, max_dec_trg_len), pad_token, dtype=dtype, device=device)

                # copio in-place le sequenze nelle matrici di [PAD]
                for i, (enc_seq, dec_seq, trg_seq) in enumerate(zip(encoder_input, decoder_input, target)):
                    encoder_input_pad[i, :len(enc_seq)] = enc_seq
                    decoder_input_pad[i, :len(dec_seq)] = dec_seq
                    target_pad[i, :len(trg_seq)] = trg_seq

                # riassegno i valori
                encoder_input = encoder_input_pad
                decoder_input = decoder_input_pad
                target = target_pad

            # creo le maschere
            # (batch, max_len) -> (batch, 1, 1, max_len)
            encoder_mask = (encoder_input != pad_token).unsqueeze(1).unsqueeze(1)
            # (batch, 1, max_len) & (batch, max_len, max_len) -> (batch, 1, max_len, max_len)
            decoder_mask = (causal_mask(decoder_input, dim=1) & (decoder_input != pad_token).unsqueeze(1)).unsqueeze(1)
        else:
            # devo solo fare stacking
            encoder_input = stack(encoder_input)
            decoder_input = stack(decoder_input)
            encoder_mask = stack([dct['encoder_mask'] for dct in dict_list])
            decoder_mask = stack([dct['decoder_mask'] for dct in dict_list])
            target = stack(target)

        # Gestisco le frasi in chiaro
        src_text = [dct['src_text'] for dct in dict_list]
        trg_text = [dct['trg_text'] for dct in dict_list]

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'target': target,
            'src_text': src_text,
            'trg_text': trg_text
        }

    
    # funzione chiamata ogni volta che viene chiesto un iteratore alla
    # classe, come ad esempio in un for
    def __iter__(self):
        if self.shuffle:
            # NOTE: potrei fare self.dataset.shuffle()
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            # questo purtroppo non posso usarlo perche' BilingualDataset
            # non supporta piu' di un idx per volta tramite __getitem__(),
            # o meglio, non sa come tokenizzare il testo
            #batch = self.dataset[batch_indices]
            print("AAAA")
            print(batch)

            yield self.stack_on_batch_dim(batch, self.dataset.dynamic_padding, self.my_stack, self.dataset.pad_token)


    
# https://youtu.be/ISNdQcPhsts?feature=shared&t=5734 
# https://youtu.be/ISNdQcPhsts?feature=shared&t=5948
# https://youtu.be/ISNdQcPhsts?feature=shared&t=6372
# https://youtu.be/ISNdQcPhsts?feature=shared&t=6446
# https://youtu.be/ISNdQcPhsts?feature=shared&t=6478
# https://youtu.be/ISNdQcPhsts?feature=shared&t=7623