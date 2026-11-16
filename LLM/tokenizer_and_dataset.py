import json
from os.path import join, exists

from tokenizers import Tokenizer # pip install tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers
from tokenizers import decoders

from datasets import load_dataset

# NOTE: se in futuro ci sara' un dataset di validation, il tokenizer
# NON deve usarlo per creare il vocabolario del LLM 


TOK_PATH = f"LLM/tok.json"
SPEAKER_TOK = "<|speaker|>"
EOT_TOK = "<|eot|>"
JSONL_PATH = f'LLM/data/dataset/SG_FUSED_LLM.jsonl'
VOCAB_SIZE = 2048


# usata per dare il testo di training in pasto al tok
def tok_training_corpus_iterator(jsonl_path: str):  
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for scene in f:
            scene_data = json.loads(scene)
            
            scene_text = ""
            for turn in scene_data["turns"]:
                # NON inserisco i token speciali nel corpus
                # di training del tokenizer
                scene_text += f"{turn['speaker']}\n{turn['text']}\n"
            yield scene_text


# usata per tokenizzare il ds tramite .map()
def tokenize_scene(example):
    '''
    Prendo tutte le battute di una scena (file), aggiungo i token
    speciali, tokenizzo e restituisco
    '''
    scene_text = ""
    for turn in example["turns"]:
        scene_text += f"<|speaker|>{turn['speaker']}\n{turn['text']}<|eot|>\n"
    
    encoded = tokenizer.encode(scene_text, add_special_tokens=False).ids
    # metto la frase tokenizzata in una nuova colonna
    # (cosi' droppo quella con il plain text)
    assert min(encoded) >= 0
    assert max(encoded) <= 65535
    return {"input_ids": encoded}


# ==== STEP 1: definizione ed addestramento del tokenizer ====

if not exists(TOK_PATH):
    # non metto nessun unknown token perche' tanto non si verificano
    tokenizer = Tokenizer(BPE(unk_token=None))
    # aggiunge uno spazio artificale ad inizio stringa. Di norma e'
    # una buona pratica per evitare rappresentazioni diverse per la stessa
    # parola ("cazzo" vs " cazzo")
    # Lo disattivo perche' me lo mette tra <|speaker|> e il nome
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # importante per visualizzare correttamente i tokens, come lo "spazio" Ġ
    tokenizer.decoder = decoders.ByteLevel()

    # visto che non fanno piu' parte del corpus, gli posso aggiungere direttamente
    # in fase di training, cosi' risparmio anche spazio sul disco (che altrimenti
    # verrebbe sprecato per sti tokens)
    special_tokens = ["<|pad|>", "<|speaker|>", "<|eot|>"]
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True
    )

    tokenizer.train_from_iterator(
        tok_training_corpus_iterator(JSONL_PATH),
        trainer
    )

    tokenizer.save(TOK_PATH)
    print(f"Tokenizer salvato con successo")
else:
    tokenizer = Tokenizer.from_file(TOK_PATH)
    print("Tokenizer caricato con successo")

from datasets import load_dataset, Features, Sequence, Value
# senza di questo si rompe il multiprocessing di dataset.map()
if __name__ == '__main__':

    # ==== STEP 2: tokenizzazione del ds e salvataggio ====
    dataset = load_dataset('json', data_files=JSONL_PATH)

    # tokenizzo tutto il dataset
    tokenized_dataset = dataset.map(
        tokenize_scene,
        remove_columns=["turns"], # droppo la colonna con le frasi in chiaro
        num_proc=4                # multiprocessing
    )

    tokenized_dataset = tokenized_dataset.cast_column(
        "input_ids",
        Sequence(Value("uint16"))
    )

    print(tokenized_dataset)
    print(tokenized_dataset['train'][0]) # Vedrai game, scene e input_ids pronti
    # Salva l'intero dataset Arrow su disco
    tokenized_dataset.save_to_disk("LLM/data/dataset/SG_ready")
    print("Dataset salvato su disco!")