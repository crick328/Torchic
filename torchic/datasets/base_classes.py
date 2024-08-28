import random
from torchic.cuda_tensor import Tensor
from torchic.tools.helpers import cat, stack



class Dataset:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        #return self.dataset[idx]
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")


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


    @staticmethod
    def stack_on_batch_dim(dict_list: list[dict[str, Tensor]]):
        
        # NOTE: il dtype dovrebbe essere ereditato dinamicamente, anche se
        # e' forzato a 32 bit. CI vorrebbe una flag che permette di annullare
        # il casting. Il device invece non viene passato

        # accumulo i dati sulla dimensione delle batch
        # (seq_len) -> (batch, seq_len)
        #merged_encoder_input = Tensor([ d['encoder_input'].data for d in dict_list ])
        # (seq_len) -> (batch, seq_len)
        #merged_decoder_input = Tensor([ d['decoder_input'].data for d in dict_list ])
        # (1, 1, seq_len) -> (batch, 1, 1, seq_len)
        #merged_encoder_mask = Tensor([ d['encoder_mask'].data for d in dict_list ])
        # (1, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
        #merged_decoder_mask = Tensor([ d['decoder_mask'].data for d in dict_list ])
        # (seq_len) -> (batch, seq_len)
        #merged_target = Tensor([ d['target'].data for d in dict_list ])

        # estraggo i dati
        encoder_inputs = [d['encoder_input'].data for d in dict_list]
        decoder_inputs = [d['decoder_input'].data for d in dict_list]
        encoder_masks = [d['encoder_mask'].data for d in dict_list]
        decoder_masks = [d['decoder_mask'].data for d in dict_list]
        targets = [d['target'].data for d in dict_list]

        # concateno sulla prima dimensione
        merged_encoder_input = stack(encoder_inputs)
        merged_decoder_input = stack(decoder_inputs)
        merged_encoder_mask = stack(encoder_masks)
        merged_decoder_mask = stack(decoder_masks)
        merged_target = stack(targets)
        
        # gestisco le frasi in chiaro
        merged_src_text = [d['src_text'] for d in dict_list]
        merged_trg_text = [d['trg_text'] for d in dict_list]

        return {
            'encoder_input': merged_encoder_input,
            'decoder_input': merged_decoder_input,
            'encoder_mask': merged_encoder_mask,
            'decoder_mask': merged_decoder_mask,
            'target': merged_target,
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
            batch = [self.dataset[idx] for idx in batch_indices]

            yield self.stack_on_batch_dim(batch)