import random
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import stack
from datasets.arrow_dataset import Dataset as HFDataset
import pandas as pd
from math import ceil
#from torchic.tools.helpers import cat, ones, full
#from torchic.tools.sequences import causal_mask, pad_sequence
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

            # it can also load custom data:
        >>> dataset_raw = load_dataset(
                "csv", 
                data_files="torchic/datas/SG_en-it/SG_transaltion_dataset.csv", 
                delimiter="|",                      # specifica il separatore
                column_names=["name", "en", "it"],  # (ri)nomino le colonne
                usecols=["en", "it"],               # ignoro la colonna "name"
                split="train"                       # carico tutto come unico split
            )
        >>> train_test_split = dataset_raw.train_test_split(test_size=0.025)
        >>> train_ds = Dataset(train_test_split["train"])

    Example with Kaggle::

        >>> import os
        >>> import pandas as pd
        >>> os.environ["KAGGLE_CONFIG_DIR"] = r"torchic/datasets"
        >>> from kaggle.api.kaggle_api_extended import KaggleApi

        >>> api = KaggleApi()
        >>> api.authenticate()
        >>> api.dataset_download_files(
                dataset='oddrationale//mnist-in-csv',
                path="./data",
                unzip=True
            )
        >>> train_ds = Dataset(pd.read_csv("data/mnist_train.csv"))
        >>> test_ds = Dataset(pd.read_csv("data/mnist_test.csv"))
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
        

    def subset(self, fraction: float = 0.1, seed: int=42):
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

        # imposto un seed, ma lo faccio su un generatore locale
        # per non avere impatto altrove
        rng = random.Random(seed)
        # prendo n (subset_size) indici unici tra quelli iniziali
        indices = rng.sample(range(total_size), subset_size)

        return Subset(self.dataset, indices, self.dataset_type)


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices sampled from the whole dataset

    This class allows for a lazy approach: no data is duplicated and
    the subset are computed only when accessed
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: list[int] | int):
        # reindirizzo al __getitem__ del Dataset
        #if isinstance(idx, list):
            #return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        # La lunghezza è semplicemente il numero di indici selezionati
        return len(self.indices)

        
class DataLoader:
    """
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle: rimescola i dati ad ogni epoch
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool=True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))

    def __len__(self) -> int:
        return ceil(len(self.dataset) / self.batch_size)

    
    # funzione chiamata ogni volta che viene chiesto un iteratore alla
    # classe, come ad esempio in un for
    def __iter__(self):
        if self.shuffle:
            # NOTE: potrei fare self.dataset.shuffle(), ma
            # NON farlo perche' e' un disastro prestazionale
            # e di memoria (giustamente). Il dataset va
            # lasciato intonso
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            #yield stack(batch)
            yield self.universal_stacker(batch)

    def universal_stacker(self, batch):
        """
        Combina una lista di elementi (batch) in un'unica struttura dati batchata.
        Supporta: dict, tuple, np.ndarray, torch.Tensor, int, float, str.
        """
        elem = batch[0]
        elem_type = type(elem)

        
        # 1. Se il singolo elemento è un Dizionario (es. HuggingFace Datasets, CSV letti come dict)
        if isinstance(elem, dict):
            return {key: self.universal_stacker([d[key] for d in batch]) for key in elem}

        # 2. Se è una Tupla o Lista (es. (X, y))
        elif isinstance(elem, (tuple, list)):
            transposed = zip(*batch)
            return elem_type(self.universal_stacker(samples) for samples in transposed)

        # 3. Se è un NumPy Array
        elif isinstance(elem, np.ndarray):
            return np.stack(batch, axis=0)

        # 4. Se è un PyTorch Tensor
        elif isinstance(elem, Tensor):
            return stack(batch, dim=0)

        # 5. Se è un Intero, Float o Scalare NumPy -> Converti in NumPy Array 1D (o Tensor)
        elif isinstance(elem, (int, float, np.integer, np.floating)):
            return Tensor(batch)

        # 6. Se è una Stringa o altro tipo non impilabile -> Mantieni la lista
        elif isinstance(elem, str):
            return batch

        else:
            # non so che cazzo sia... Provo a castare in Tensor e via
            return Tensor(batch)


        raise TypeError(f"Tipo di dato non supportato per il collate automatico: {elem_type}")




import random
from torch.utils.data import Subset


def random_split(dataset, lengths, seed=42):
    """Suddivide un Dataset in più Subset senza copiare i dati in memoria.

    Parameters:
    dataset (Dataset): Il dataset PyTorch originale.
    lengths (list of int): Le lunghezze di ciascuno split (es. [80, 20]).
    seed (int, optional): Seed per la riproducibilità.

    Returns:
    list of Subset: Lista di oggetti torch.utils.data.Subset.

    Example::

        >>> train_len = int(0.8 * len(my_dataset))
        >>> val_len = len(my_dataset) - train_len

        # Generi i due Subset
        >>> train, val = random_split(
            my_dataset, [train_len, val_len], seed=42
        )
    """
    ds_len = len(dataset)
    if sum(lengths) != ds_len:
        raise ValueError(
            f"La somma delle lunghezze ({sum(lengths)}) deve essere uguale alla lunghezza del dataset ({ds_len})."
        )

    # 1. Creiamo e mescoliamo solo gli indici (0, 1, 2, ..., N-1)
    indices = list(range(ds_len))

    if seed is not None:
        # generatore locale per il seed
        rng = random.Random(seed)
        rng.shuffle(indices)
    else:
        # randomizza se l'utente lo esplicita passando None
        random.shuffle(indices)

    # 2. Creiamo i Subset usando lo slicing sugli INDICI (non sui dati!)
    splits = []
    current_idx = 0

    for length in lengths:
        # Estraiamo solo il blocco di indici per questo split
        split_indices = indices[current_idx : current_idx + length]

        # Creiamo un Subset leggero che punta al dataset originale
        splits.append(Subset(dataset, split_indices))

        current_idx += length

    return splits


import numpy as np

class BinaryDataset(Dataset):

    def __init__(self, binary_file_path, shape, dtype=np.float32, is_raw_binary=False):
        self.file_path = binary_file_path
        self.shape = shape
        self.dtype = dtype
        self.is_raw_binary = is_raw_binary

        # NON apriamo il file qui per evitare bug con num_workers > 0!
        self.data = None

    def _init_mmap(self):
        """Apre il file mmap solo quando realmente necessario (Lazy

        Initialization).
        """
        if self.is_raw_binary:
            # Per file binari grezzi (.bin, .raw)
            self.data = np.memmap(
                self.file_path, dtype=self.dtype, mode="r", shape=self.shape
            )
        else:
            # Per file salvati con np.save() (.npy)
            # Il file NON viene caricato in RAM, viene solo mappato!
            # non conviene usare cupy mi sa
            self.data = np.load(self.file_path, mmap_mode="r")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        # idx = una riga completa
        # Regola FONDAMENTALE: tutte le righe devono avere una dimensione
        # fissa in byte! -> se i dati sono di lunghezza variabile (transformer),
        # devi applicare il padding prima di scrivere i dati in memoria!
        if self.data is None:
            self._init_mmap()

        # Esempio: estraiamo la riga idx da 256 interi (128 en, 128 it)
        row = np.array(self.data[idx], copy=True)

        # Separiamo input (EN) e target (IT)
        src_tokens = row[:128]  # primi 128 int
        tgt_tokens = row[128:]  # ultimi 128 int

        return Tensor(src_tokens), Tensor(tgt_tokens)

'''
ALTERNATIVA:
Fai due file binari: il primo, quello dei dati, lo farcisci di merda.
Butti tutte le frasi una dopo l'altra, senza padding.
Il secondo sara' un file di supporto index.bin. Questo praticamente
sara' come il file dell'approccio precedente: struttura rigidissima. Ogni
riga e' composta esattamente da tot byte e basta. In questi, ci gli indici che
ti servono per interpretare data.bin.
Ad esempio, sai che indices.bin e' composto da righe di 4 numeri interi. Il primo
rappresenta lo SOS della frase en, il secondo il SOS della it, il terzo qualcos'altro.
Prendi gli indici, li trasformi in numeri e li usi per accedere all'altro.

Parlo di "righe", ma il file sara' una fila di 0 e 1. Sei tu che sai la lunghezza
di una riga e quante righe hai scritto. In questo modo, ogni riga e' un indice e
tutte le classi precedenti lavorano con gli indici. Puoi randomizzare, dividere e
poi interpelli l'indice che ti serve

  
data.bin:  [12, 45, 9, 8, 33, 91, 88, 3, 14, 5...] (Niente padding!)
index.bin: [(start: 0, len: 3), (start: 3, len: 3), (start: 6, len: 4)...]

Come funzionerebbe:
class VariableBinaryDataset(Dataset):
    def __init__(self, data_file, index_file):
        self.data_file = data_file
        self.index_file = index_file

        # Il file degli indici ha dimensione FISSA (es. 2 int64 per riga: start e len)
        # Quindi possamo fare il mmap sull'indice!
        self.index_mmap = np.memmap(
            index_file, dtype=np.int64, mode="r"
        ).reshape(-1, 2)
        self.data_mmap = None

    def __len__(self):
        return len(self.index_mmap)

    def __getitem__(self, idx):
        if self.data_mmap is None:
            # Apriamo il file di dati vero e proprio
            self.data_mmap = np.memmap(self.data_file, dtype=np.int32, mode="r")

        # 1. Troviamo dove inizia e quanto è lunga la frase idx
        start_pos, length = self.index_mmap[idx]

        # 2. Leggiamo ESATTAMENTE quei token dal file di dati
        tokens = np.array(
            self.data_mmap[start_pos : start_pos + length], copy=True
        )

        return Tensor(tokens)
'''

'''
Dataset su Disco/Cloud (es. 1 TB)
 ├── shard_00000.tar / .bin
 ├── shard_00001.tar / .bin
 └── shard_00002.tar / .bin
       │
       ▼ (Lettura in stream sequenziale)
 [ DataLoader ] ──► Decomprime al volo e passa i batch alla GPU
'''