import random
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import stack
from datasets.arrow_dataset import Dataset as HFDataset
import pandas as pd
from math import ceil
from torchic.tools.helpers import cat, ones, full
from torchic.tools.sequences import causal_mask, pad_sequence
from torchic.dtypes import int64


class Dataset:
    '''
    Universal wrapper class for both `datasets` Hugging Face package
    and `kaggle`.
    When possible, is suggested the use of `datasets` because
    it provides a very efficient and versatile class to
    work with data. `kaggle` is a fallback when the data isn't available on HF.
    In this case, the dataset has to be converted in a pandas
    `DataFrame` before creating the object Dataset.

    Example with HF::

        >>> from datasets import load_dataset
        >>> mnist = Dataset(load_dataset("mnist"))

    Example with Kaggle::

        >>> import os
        >>> import pandas as pd
        >>> os.environ['KAGGLE_CONFIG_DIR'] = r'torchic\datasets'
        >>> from kaggle.api.kaggle_api_extended import KaggleApi

        >>> api = KaggleApi()
        >>> api.authenticate()
        >>> api.dataset_download_files(
                dataset='oddrationale/mnist-in-csv',
                path="./data",
                unzip=True
            )
        >>> train_ds = Dataset(pd.read_csv('data/mnist_train.csv'))
        >>> test_ds = Dataset(pd.read_csv('data/mnist_test.csv'))
    '''

    def __init__(self, dataset) -> None:
        if isinstance(dataset, HFDataset):#(HFDataset, HFArrowDataset)):
            self.dataset_type = 'huggingface'
        elif isinstance(dataset, pd.DataFrame):
            self.dataset_type = 'kaggle'
        else:
            raise TypeError(
                "Unsupported dataset type. Use Hugging Face Dataset or pandas DataFrame."
            )
        
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset_type == 'huggingface':
            # Hugging Face dataset restituisce un dict già strutturato
            return self.dataset[index]
        elif self.dataset_type == 'kaggle':
            # Trasformare una riga di pandas DataFrame in un dict compatibile
            return self.dataset.iloc[index]
        else:
            raise NotImplementedError("Dataset type not supported")
        

    def subset(self, fraction: float = 0.1, seed: int = 42):
        """
        Returns a fraction of the original dataset.
        
        Args:
            fraction (float): La frazione del dataset da selezionare (default: 0.1).
            seed (int): Seed per la riproducibilità (default: 42).
            
        Returns:
            Dataset: Un nuovo oggetto Dataset contenente il subset.
        """
        if not 0 < fraction <= 1:
            raise ValueError("La frazione deve essere compresa tra 0 e 1.")

        total_size = len(self)
        subset_size = int(total_size * fraction)

        random.seed(seed)  # Per riproducibilità
        # prendo n (subset_size) indici unici tra quelli iniziali
        indices = random.sample(range(total_size), subset_size)

        if self.dataset_type == 'huggingface':
            # Filtriamo il dataset Hugging Face usando gli indici
            subset_data = self.dataset.select(indices)
        elif self.dataset_type == 'kaggle':
            # Filtriamo il DataFrame pandas usando gli indici
            subset_data = self.dataset.iloc[indices].reset_index(drop=True)
        else:
            raise NotImplementedError("Dataset type not supported")

        if self.__class__ == Dataset:
            return Dataset(subset_data)
        else:
            self.dataset = Dataset(subset_data)
            return self
        

class DataLoader:
    """
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle: rimescola i dati ad ogni epoch
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool=True, my_stack=True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.my_stack = my_stack

    @staticmethod
    def my_stack_on_batch_dim(dict_list: list[dict[str, Tensor]], dynamic_masking, pad_token: Tensor):
        # estraggo i dati
        encoder_input = [dct['encoder_input'] for dct in dict_list]
        decoder_input = [dct['decoder_input'] for dct in dict_list]
        target = [dct['target'] for dct in dict_list]

        if dynamic_masking:
            # calcolo le lunghezze massime
            max_enc_len = max(len(seq) for seq in encoder_input)
            max_dec_len = max(len(seq) for seq in decoder_input)
            max_trg_len = max(len(seq) for seq in target)
            #max_dec_trg_len = max(max_dec_len, max_trg_len)

            # metto i token di padding
            # (seq_len,) -> (max_len,) -> (batch, max_len)
            encoder_input = stack(
                pad_sequence(seq, max_enc_len, pad_token) for seq in encoder_input
            )
            decoder_input = stack(
                pad_sequence(seq, max_dec_len, pad_token) for seq in decoder_input
            )
            target = stack(
                pad_sequence(seq, max_trg_len, pad_token) for seq in target
            )

            # creo le maschere
            # (batch, max_len) -> (batch, 1, 1, max_len)
            encoder_mask = (encoder_input != pad_token).unsqueeze(1).unsqueeze(1)
            # (batch, 1, max_len) & (batch, max_len, max_len) -> (batch, 1, max_len, max_len)
            decoder_mask = ((decoder_input != pad_token).unsqueeze(1) & causal_mask(decoder_input, dim=1)).unsqueeze(1)
        else:
            # aggiungo la batch_dim e accumulo su di essa
            encoder_input = stack(encoder_input)
            decoder_input = stack(decoder_input)
            target = stack(target)
            encoder_mask = stack(dct['encoder_mask'] for dct in dict_list)
            decoder_mask = stack(dct['decoder_mask'] for dct in dict_list)

        # gestisco le frasi in chiaro (generators, lazy)
        src_text = [d['src_text'] for d in dict_list]
        trg_text = [d['trg_text'] for d in dict_list]

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'target': target,
            'src_text': src_text,
            'trg_text': trg_text
        }


    @staticmethod
    def stack_on_batch_dim(dict_list: list[dict[str, Tensor]], dynamic_masking, pad_token: Tensor):
        """
        Combina un elenco di dizionari su una nuova dimensione batch e genera maschere opzionali.
        
        Args:
            dict_list: Elenco di dizionari contenenti tensori con chiavi comuni.
            dynamic_masking: Booleano che abilita o disabilita il padding dinamico.
            pad_token: Token utilizzato per il padding.
        
        Returns:
            Un dizionario combinato con eventuali maschere.
        """
        # estraggo i dati
        encoder_input = [dct['encoder_input'] for dct in dict_list]
        decoder_input = [dct['decoder_input'] for dct in dict_list]
        target = [dct['target'] for dct in dict_list]

        if dynamic_masking:
            # calcolo le lunghezze massime della batch
            max_enc_len = max(len(seq) for seq in encoder_input)
            max_dec_trg_len = max(max(len(seq) for seq in decoder_input), max(len(seq) for seq in target))

            # creo delle matrici di [PAD] in modo da applicare
            # in batch il padding alle sequenze
            batch_size = len(dict_list)         # (la posso prendere da self)
            device = encoder_input[0].device
            dtype = encoder_input[0].dtype

            #encoder_input_pad = pad_token * ones((batch_size, max_enc_len), dtype=dtype, device=device)
            #decoder_input_pad = pad_token * ones((batch_size, max_dec_trg_len), dtype=dtype, device=device)
            #target_pad = pad_token * ones((batch_size, max_dec_trg_len), dtype=dtype, device=device)

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
            encoder_mask = (encoder_input != pad_token).unsqueeze(1).unsqueeze(1)
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

            if not self.my_stack:
                yield self.stack_on_batch_dim(batch, self.dataset.dynamic_masking, self.dataset.pad_token)
            else:
                yield self.my_stack_on_batch_dim(batch, self.dataset.dynamic_masking, self.dataset.pad_token)

    def __len__(self) -> int:
        return ceil(len(self.dataset) / self.batch_size)