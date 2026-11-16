import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torchic
from datasets import load_from_disk
from torchic.data.dataloaders.base_classes import Dataset
from LLM.data.SG_dataloader import CausalLMDataLoader

from tokenizers import Tokenizer # pip install tokenizers

SEED = 39
# questi andrebbero scritti su un file in modo da avere le stesse
# stringhe da file diversi
DATASET_PATH = f"LLM/data/dataset/SG_ready"
TOK_PATH = f"LLM/tok.json"



# ==== Gestione di dataset e dataloader ====

tokenizer = Tokenizer.from_file(TOK_PATH)

# carico in memoria il ds pre-tokenizzato. E' un DatasetDict (HFArrow format)
# dove ogni riga rappresenta una scena (dizionario) con
# dict_keys(['game', 'scene', 'input_ids'])
dataset = load_from_disk(DATASET_PATH)
print(f"Colonne del DatasetDict: {dataset['train'][1].keys()}")

max_seq_len = 0
max_index = -1

# itero a botte di 1000 scene alla volta per trovare la max len
for start in range(0, len(dataset), 1000):
    batch = dataset['train'][start:start + 1000]

    lengths = [
        len(seq)
        for seq in batch["input_ids"]
    ]
    batch_max = max(lengths)

    if batch_max > max_seq_len:
        max_seq_len = batch_max
        max_index = start + lengths.index(batch_max)

print("Sequenza massima:", max_seq_len)
print("Indice scena:", max_index)
print("Scena:", dataset['train'][max_index]["scene"])
# minchia, e dove me la metto una frase lunga 16k tokens?
exit()


# ha uno split unico dummy/default; ne faccio uno vero
dataset = dataset["train"].train_test_split(
    test_size=0.08,
    seed=SEED
)
train = Dataset(dataset['train'])
test = Dataset(dataset['test'])

#print("Scena di training (primi 200 caratteri):")
#print(tokenizer.decode(train[0]["input_ids"], skip_special_tokens=False)[:200])


train_dataloader = CausalLMDataLoader(train, batch_size=2, pad_token_id=0)

'''
# Tipo specifico della colonna
print(train.dataset.data.schema.field("input_ids").type)
# Tipo degli elementi contenuti nella lista
print(
    train.dataset.data.schema
    .field("input_ids")
    .type.value_type
)'''
from torchic import int32, int64
for batch in train_dataloader:
    print(batch)
    #print(tokenizer.decode(batch["input_ids"].data[0], skip_special_tokens=False)[-50:])
    print(batch["input_ids"].data[0][-10:])
    exit()

