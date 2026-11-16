from torchic.data.dataloaders.base_classes import DataLoader
from torchic.tools.helpers import cat, ones, full
from torchic import Tensor, uint16
import random



class CausalLMDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int, pad_token_id: int, shuffle: bool = True) -> None:
        super().__init__(dataset, batch_size, shuffle)
        # per compatibilita' con full() lo rendo Tensor, ma non so se gradisco
        # Aaaah, perche' full() chiama direttamente stile cp.full, per
        # quello vuole il backend
        self.pad_token_id = Tensor(pad_token_id)

    def stack_on_batch_dim(self, batch_dict):
        """
        Supporto la dynamic max seq len
        """
        # per ogni batch, prendo gli ids e li converto in Tensor
        sequences = [Tensor(seq, dtype=uint16) for seq in batch_dict['input_ids']]
        
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)

        # creo delle matrici piene di PAD, dove ci copiero' gli ids.
        # In pratica e' un modo efficiente per aggiungere il padding
        # necessario ad ogni frase per raggiungere la max_len della batch
        input_ids_pad = full((batch_size, max_len), self.pad_token_id, dtype=uint16)
        for i, seq in enumerate(sequences):
            input_ids_pad[i, :len(seq)] = seq

        # creo la maschera in modo tale che i pad non partecipino all'attention
        attention_mask = (input_ids_pad != self.pad_token_id)

        return {
            'input_ids': input_ids_pad,
            'attention_mask': attention_mask,
            # per debug... Non si sa mai
            'game': batch_dict.get('game', None),
            'scene': batch_dict.get('scene', None)
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            
            # HFArrow supporta nativamente l'accesso a set di indici,
            # ma restituisce un dizionario di colonne (unico dict avente
            # matrici ad ogni entry) anziché una lista di righe
            batch_dict = self.dataset[batch_indices]
            
            yield self.stack_on_batch_dim(batch_dict)