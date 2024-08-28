import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.2")
from torchic import Tensor, set_default_device
from torchic.tools.visualization.graphs import draw_graph
#import torchic.nn as nn
from dataset import BilingualDataset, causal_mask
from model_config import get_config

from torchic.datasets.base_classes import DataLoader
from torchic.nn.optim import AdamV3
from torchic.nn.loss import CrossEntropyLoss
from torchic.nn.models.save_load import save_model, load_architecture
from transformer import Transformer

from datasets import load_dataset # pip install datasets
from tokenizers import Tokenizer # pip install tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import random

from tqdm import tqdm


set_default_device("cpu")

# serve implementare la cross entropy loss (e anche la categorical se vuoi)
# NOTE: cross entropy e categorical cross entropy sono la stessa cosa!
# serve anche random_split, Dataset e Dataloader (tutti e tre parte di torch.utils.data)
# serve un random che abbia gli stessi seed di Torch


def get_all_sentences(dataset, lang):
    # dataset: lista di dict con coppia frase {it: eng}
    for item in dataset:
        yield item["translation"][lang]


def get_or_build_tokenizer(model_config, dataset, lang):
    tokenizer_path = model_config["tokenizer_file"].format(lang)

    if not os.path.exists(tokenizer_path):
        #os.makedirs(tokenizer_path)
        # se il tokenizer incontra una parola che non fa parte del suo
        # vocabolario, la rimpiazza con [UNK]
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # il tokenizer dividera' le parole in base agli spazi
        # prima di operare
        tokenizer.pre_tokenizer = Whitespace()
        # min_frequency=2 significa che il tokenizer inserira' nel vocabolario
        # solo le parole che appaiono almeno due volte
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        # training del tokenizer (what? Training? Pensavo fosse tutta prog. vanilla)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    return tokenizer


# funzione di torch
def __random_split(dataset: list, train_split: float):
    random.shuffle(dataset)
    train_data = dataset[:int((len(dataset)) * train_split)]
    test_data = dataset[int(len(dataset) * train_split):]

    assert len(dataset) == len(test_data) + len(train_data)
    return train_data, test_data


def _random_split(dataset, lengths, seed=None):
    """
    Suddivide il dataset in due parti in base alle lunghezze specificate.
    
    Args:
        dataset (Dataset): Il dataset da suddividere.
        lengths (list): Una lista di lunghezze. La somma delle lunghezze deve essere uguale alla lunghezza del dataset.
        seed (int, optional): Il seed per il generatore di numeri casuali. Default è None.
    
    Returns:
        tuple: Due dataset suddivisi.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("La somma delle lunghezze deve essere uguale alla lunghezza del dataset")
    
    if seed is not None:
        random.seed(seed)
    
    # Otteniamo gli indici del dataset
    indices = list(range(len(dataset)))
    
    # Mescoliamo gli indici
    random.shuffle(indices)
    
    # Suddividiamo gli indici in base alle lunghezze specificate
    split1_indices = indices[:lengths[0]]
    split2_indices = indices[lengths[0]:lengths[0] + lengths[1]]
    
    # Creiamo i due subset
    split1 = dataset.select(split1_indices)
    split2 = dataset.select(split2_indices)

    assert len(dataset) == len(split1) + len(split2)
    return split1, split2


def random_split(data, lengths, seed=None):
    """
    Suddivide una lista di dizionari in sottogruppi di lunghezze specificate.
    
    Parameters:
    data (list of dict): La lista di dizionari da suddividere.
    lengths (list of int): Una lista contenente le lunghezze dei sottogruppi.
    seed (int, optional): Un seme per il generatore di numeri casuali (per riproducibilità).
    
    Returns:
    list of list of dict: Una lista di liste di dizionari suddivisi.
    """
    if sum(lengths) != len(data):
        raise ValueError("La somma delle lunghezze deve essere uguale alla lunghezza dei dati.")
    
    if seed is not None:
        random.seed(seed)
    
    indices = list(range(len(data)))
    random.shuffle(indices)
    splits = []
    current_idx = 0
    
    for length in lengths:
        split_indices = indices[current_idx:current_idx + length]
        splits.append([data[i] for i in split_indices])
        current_idx += length
    
    return splits


def get_dataset(model_config):
    # questo ds non e' diviso in train e validation, percio' bisogna fare lo split a mano
    dataset_raw = load_dataset("opus_books", f"{model_config['lang_src']}-{model_config['lang_trg']}", split="train")

    tokenizer_src = get_or_build_tokenizer(model_config, dataset_raw, model_config["lang_src"])
    tokenizer_trg = get_or_build_tokenizer(model_config, dataset_raw, model_config["lang_trg"])

    train_ds_size = int(0.9*len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,
                                tokenizer_src,
                                tokenizer_trg,
                                model_config["lang_src"],
                                model_config["lang_trg"],
                                model_config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw,
                                tokenizer_src,
                                tokenizer_trg,
                                model_config["lang_src"],
                                model_config["lang_trg"],
                                model_config["seq_len"])

    # controllo la lunghezza massima delle frasi
    max_len_src = 0
    max_len_trg = 0

    for sentence_pair in dataset_raw:
        src_ids = tokenizer_src.encode(sentence_pair["translation"][model_config["lang_src"]]).ids
        trg_ids = tokenizer_trg.encode(sentence_pair["translation"][model_config["lang_trg"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))

    print(f"Max length of src lang: {max_len_src}")
    print(f"Max length of trg lang: {max_len_trg}")

    train_dataloader = DataLoader(train_ds, batch_size=model_config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg


def get_model(model_config, vocab_src_len, vocab_trg_len):
    model = Transformer(d_model=model_config["d_model"],
                        encoder_vocab_size=vocab_src_len,
                        dec_vocab_size=vocab_trg_len,
                        max_len=model_config["seq_len"])
    print(vocab_src_len, vocab_trg_len)
    return model


def train_model(model_config):
    initial_epoch = 0

    if not os.path.exists(model_config["model_folder"]):
        os.makedirs(model_config["model_folder"])

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_dataset(model_config)
    model = get_model(model_config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size())

    #save_model(model, "transformer")
    #print(model.__dict__)
    #model = load_architecture("transformer")

    optimizer = AdamV3(model.parameters(), lr=model_config["lr"], eps=1e-9)
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)

    # qui mettere logica per caricare modello
    if model_config["preload"]:
        # qua devi caricare il modello e lo stato dell'optim
        pass

    #print(model)
    #draw_graph(model)
    #return 0

    
    for epoch in range(initial_epoch, model_config["num_epochs"]):
        #model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch:02d}")

        for batch in batch_iterator:
            # frase
            encoder_input = batch["encoder_input"] # (batch, seq_len)
            # frase target
            decoder_input = batch["decoder_input"] # (batch, seq_len)
            # nasconde solo i token di padding
            encoder_mask = batch["encoder_mask"]   # (batch, 1, 1, seq_len)
            # nasconde anche i token successivi al token in esame
            decoder_mask = batch["decoder_mask"]   # (batch, 1, seq_len, seq_len)

            prediction = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            label = batch["label"] # (batch, seq_len)

            #draw_graph(prediction, format="png", complete_format=False)
            #return 0

            # (batch, seq_len, trg_voab_size) -> (batch * seq_len, trg_voab_size)
            loss = loss_fn(prediction.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.data : 6.3f}"})

            #draw_graph(model, loss)

            # niente mean backward???
            # no perche' e' gia' uno scalare!
            #print("INIZIO BACKWARD")
            loss.backward()
            #print("FINE BACKWARD!")

            optimizer.step()
            optimizer.zero_grad()

        # salva un checkpoint alla fine di ogni epoch
        #print("FINE EPOCH")

if __name__ == "__main__":
    config = get_config()
    train_model(config)