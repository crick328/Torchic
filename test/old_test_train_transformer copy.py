import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic import Tensor, set_default_device
from torchic.tools.visualization.graphs import draw_graph
from torchic.tools.code_analysis.mem import get_nbytes
#import torchic.nn as nn
from torchic.nn.models.transformer_config import get_config
from torchic.data.dataloaders.bilingual_dataloader import BilingualDataset, causal_mask
from torchic.data.dataloaders.base_classes import DataLoader
from torchic.nn.optim import AdamV3
from torchic.nn.loss import CrossEntropyLoss
from torchic.nn.models.save_load import save, load_architecture
from torchic.nn.models.transformer import Transformer

from datasets import load_dataset # pip install datasets
from tokenizers import Tokenizer # pip install tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import random

from tqdm import tqdm

# prima di far partire un training serio, farei due cose:
# 1) implementazione reload modello + checkpoint
# 2) validation test perche' la loss non e' cosi' esplicativa


ACCUMULATION_STEPS = 128
LOG_OPTIMIZER = False
LOG_GRADS = False
LOG_DATA = False
BREAK_AFTER = -1 # bathces, -1 se non vuoi breaks
PROFILER = False
MEM_TRACE = False
sys.setrecursionlimit(1500)
set_default_device("cpu")


# NOTE: controlla bene che TUTTE le operazioni avvengano sullo
# stesso device!


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
    dataset_raw = load_dataset("opus_books", f"{model_config['lang_src']}-{model_config['lang_trg']}", split="train", cache_dir=None)
    print(type(dataset_raw))

    tokenizer_src = get_or_build_tokenizer(model_config, dataset_raw, model_config["lang_src"])
    tokenizer_trg = get_or_build_tokenizer(model_config, dataset_raw, model_config["lang_trg"])

    train_ds_size = int(0.9*len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    print(type(train_ds_raw))

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

    #print(model.state_dict())
    #output = open("output.txt", "w")
    #for k, v in model.state_dict().items():
        #output.writelines(f'{k} {v}\n')
    #model = load_architecture("transformer")

    optimizer = AdamV3(model.parameters(), lr=model_config["lr"], eps=1e-9)
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)

    # qui mettere logica per caricare modello
    if model_config["preload"]:
        # qua devi caricare il modello e lo stato dell'optim
        pass

    
    for epoch in range(initial_epoch, model_config["num_epochs"]):
        #model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch:02d}")
        count = 0

        for batch in batch_iterator:
            # frase
            encoder_input = batch["encoder_input"] # (batch, seq_len)
            # frase target
            decoder_input = batch["decoder_input"] # (batch, seq_len)
            # nasconde solo i token di padding
            encoder_mask = batch["encoder_mask"]   # (batch, 1, 1, seq_len)
            # nasconde anche i token successivi al token in esame
            decoder_mask = batch["decoder_mask"]   # (batch, 1, seq_len, seq_len)

            if LOG_DATA:
                with open("datalog.txt", "a", encoding="utf-8") as file:
                    file.writelines([
                        f"encoder_input: {encoder_input}\n",
                        f"decoder_input: {decoder_input}\n",
                        f"encoder_mask: {encoder_mask}\n",
                        f"decoder_mask: {decoder_mask}\n",
                        f"target: {batch['target']}\n",
                        f"src_tect: {batch['src_text']}\n",
                        f"trg_text: {batch['trg_text']}\n",
                        "------------------------------\n"
                    ])

            prediction = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            target = batch["target"] # (batch, seq_len)

            #draw_graph(prediction, format="png", complete_format=False)
            #return 0

            #prediction._device.savetxt("predlog.txt", prediction.data, fmt='%.4e',)
            #with open("predlog.txt", "a", encoding="utf-8") as file:
                #file.write(f"{prediction.data}\n\n")

            # (batch, seq_len, trg_vocab_size) -> (batch * seq_len, trg_vocab_size)
            # in entrambi i tensor moltiplico le dimensioni di batch e seq_len
            loss = loss_fn(prediction.view(-1, tokenizer_trg.get_vocab_size()), target.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.data : 6.3f}"})

            #draw_graph(prediction, complete_format=False)

            # niente mean backward???
            # no perche' e' gia' uno scalare!
            #print("INIZIO BACKWARD")
            loss.backward()
            #print("FINE BACKWARD!")


            # ehm... perche' non riesce a vedere i grad?
            '''
            print(f"{get_nbytes(model)/1024/1024} Mb")
            size = 0
            for data_sample in train_dataloader.dataset.dataset:
                for item in data_sample.values():
                    size += sys.getsizeof(item)
            print(f"{size/1024/1024} Mb")
            return 0
            '''

            if LOG_GRADS:
                #with open("gradlog.txt", "a", encoding="utf-8") as file:
                    #for param in model.parameters():
                        #file.write(f"{param.grad}\n")
                for param in model.parameters():
                    param._device.savetxt("gradlog.txt", param.grad, fmt='%.4e',)

            if LOG_OPTIMIZER:
                with open("optimlog.txt", "a", encoding="utf-8") as file:
                    for param in optimizer.params:
                        file.write(f"{param.grad}\n")

            if ACCUMULATION_STEPS:
                if count % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            if count == 8 and PROFILER:
                profiler.disable()
                stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
                # Remove dir names
                stats.strip_dirs()
                stats.print_stats()
                profiler.dump_stats("profiler_data2.prof")

                subprocess.run(["snakeviz", "profiler_data2.prof"])

            if BREAK_AFTER != -1 and count >= BREAK_AFTER:
                exit()

            count += config["batch_size"]

        # salva un checkpoint alla fine di ogni epoch
        save(model, f"transformer_{epoch}")

if __name__ == "__main__":
    config = get_config()

    if PROFILER:
        import cProfile, pstats
        import subprocess

        profiler = cProfile.Profile()
        profiler.enable()

    if MEM_TRACE:
        #import psutil
        import time
        
    train_model(config)