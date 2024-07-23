import random
import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.0")
from torchic import Tensor
from torchic.tools.helper import cat


class _Dataset:
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite`__getitem__` and `__len__`
    """
    def __init__(self, dataset) -> None:
        # dataset puo' essere una lista di liste, di dizionari, np array, ecc.
        # Per questo e' necessario definire il comportamento di __getitem__
        self.dataset = dataset

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
    
    def __len__(self):
        # non sempre funziona, dipende dal tipo di self.dataset. In caso
        # e' da overridare
        return len(self.data)



class Dataset:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        #return self.dataset[idx]
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")


    #def select(self, indices: list[int]):
        #selected_data = [self.dataset[i] for i in indices]
        #return self.__class__(selected_data)


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

    '''
    # funzione chiamata ogni volta che viene chiesto un iteratore alla
    # classe, come ad esempio in un for
    def __iter__(self):
        # devi dare qualcosa che yielda a botte di batch_size
        i = 0
        while i < len(self.dataset):
            yield self.dataset[i:i + self.batch_size]
            i += self.batch_size
        
        # dovrei rimescolare qua il ds?
    '''
    @staticmethod
    def merge_dicts(dict_list):

        # Estrarre e concatenare i tensori
        merged_encoder_input = cat([d['encoder_input'] for d in dict_list])
        merged_decoder_input = cat([d['decoder_input'] for d in dict_list])
        merged_encoder_mask = cat([d['encoder_mask'] for d in dict_list])
        merged_decoder_mask = cat([d['decoder_mask'] for d in dict_list])
        merged_label = cat([d['label'] for d in dict_list])
        
        # Gestire le stringhe
        merged_src_text = [d['src_text'] for d in dict_list]
        merged_trg_text = [d['trg_text'] for d in dict_list]

        # Creare il dizionario unito
        return {
            'encoder_input': merged_encoder_input,
            'decoder_input': merged_decoder_input,
            'encoder_mask': merged_encoder_mask,
            'decoder_mask': merged_decoder_mask,
            'label': merged_label,
            'src_text': merged_src_text,
            'trg_text': merged_trg_text
        }
    

    # funzione chiamata ogni volta che viene chiesto un iteratore alla
    # classe, come ad esempio in un for
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            #yield self.dataset.select(batch_indices)
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.merge_dicts(batch)