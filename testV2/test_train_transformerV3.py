import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic import no_grad, set_default_device, set_default_precision
from torchic import set_num_backward_workers, set_time_threshold_ms
from torchic.tools.visualization.graphs import draw_graph
from torchic.tools.eval.text_scores import bleu_score
from torchic.data.dataloaders.bilingual_dataloader import BilingualDataset, BilingualDataLoader
from torchic.nn.optim import AdamV4
from torchic.nn.loss import CrossEntropyLoss
from torchic.nn.models.save_load import save, load_checkpoint
from torchic.nn.models.transformer import TransformerV2
from torchic.tools.sequences import greedy_decode, pad_sequence
from torchic.settings import set_seed
from torchic import Tensor, int16, float32

from datasets import load_dataset # pip install datasets
from tokenizers import Tokenizer # pip install tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
from tqdm import tqdm

# NOTE: probabilmente anche questi dataloaders hanno un bottleneck!
# Controlla il codice per vedere se puoi sfruttare lo stacking nativo
# e vedi che ti dice il profiler (ovviamente solo con una epoca)

set_seed(42)

# ==== TEST SETTINGS ====
DS_LEN = 1000       # prende solo N campioni dal dataset. "False" prende tutto
PROFILER = True
SG_DATASET = False  # se False prende Opus Books
DRAW_GRAPH = False  # interrompe il training e stampa il backward graph
MY_STACK = True     
ALLOW_FUSION = True # crea un Transformer con fusioni QKV-QK
DYNAMIC_PADDING = False  # non ricordo cosa fa, ma e' un bottleneck spaventoso

# ==== TRAINING SETTINGS ====
EARLY_STOP = False  # numero di batch (< epoch) a cui fermarsi
ACCUMULATION_STEPS = False
WARMUP = False
LABEL_SMOOTHING = 0.1
DROPOUT = 0.1

set_default_device("cuda")
set_default_precision(32)
set_num_backward_workers(4)
set_time_threshold_ms(5)


model_config = {
    "batch_size": 4,
    "num_epochs": 20,
    "lr": 10**-4,
    "seq_len": 320,
    "d_model": 512,
    "datasource": 'opus_books',
    "lang_src": "en",
    "lang_trg": "it",
    "model_folder": "weights",
    "model_filename": "tmodel_",
    "preload": True, #"latest",
    "tokenizer_file": "tokenizer_{0}.json",
    # per tensorboard
    "experiment_name": "runs/tmodel"
}


def get_or_build_tokenizer(model_config, dataset, lang):
    tokenizer_path = model_config["tokenizer_file"].format(lang)

    if not os.path.exists(tokenizer_path):
        # se il tokenizer incontra una parola che non fa parte del suo
        # vocabolario, la rimpiazza con [UNK]
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # il tokenizer dividera' le parole in base agli spazi prima di operare
        tokenizer.pre_tokenizer = Whitespace()
        # min_frequency=2 significa che il tokenizer inserira' nel vocabolario
        # solo le parole che appaiono almeno due volte
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        # training del tokenizer (what? Training? Pensavo fosse tutta prog. vanilla)
        tokenizer.train_from_iterator([item[lang] for item in dataset], trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def transformer_lr(step_num: int, d_model: int, warmup_steps: int) -> float:
    return d_model**-0.5 * min(step_num**-0.5, step_num * warmup_steps**-1.5)

def run_validation(model, validation_ds, tokenizer_trg, max_len, print_msg, num_examples=2):
    model.eval()
    count = 0
    total_loss = 0
    console_width = 80

    source_texts = []
    expected = []
    predicted = []

    # ma scusa, se il modello e' in eval, i gradienti
    # non dovrebbero essere disattivati di default?
    with no_grad():
        #batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch+1:02d}")
        #for batch in batch_iterator:

        # questo e' un mischione: l'eval deve essere fatto a batch, il BLEU
        # con greedy decode
        for batch in validation_ds:
            # frase
            encoder_input = batch["encoder_input"] # (batch, seq_len)
            # nasconde solo i token di padding
            encoder_mask = batch["encoder_mask"]   # (batch, 1, 1, seq_len)
            target = batch["target"]               # (batch, seq_len)

            # Ma perche'? Mi pare una grossa cazzata
            assert encoder_input.shape()[0] == 1, "Validation batch size must be 1"

            # predicted_tokens
            model_out, raw_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_trg, max_len)
            # 1. Assicurati che raw_out sia 2D: (319, 16511)
            logits = raw_out.view(-1, raw_out.shape()[-1]) if len(raw_out.shape()) > 2 else raw_out
            # 2. Trasforma il target da (1, 320) a 1D e allinealo alla lunghezza dei logits (319)
            target_1d = target.view(-1)[: logits.shape()[0]]  # Forma: (319,)
            loss = loss_fn(logits, target_1d)
            batch_iterator.set_postfix({"eval loss": f"{loss.data : 6.3f}"})
            total_loss += loss.data
            # ==== Visualizzazione delle frasi ====
            if count != num_examples:
                count += 1
                source_text = batch["src_text"][0]
                target_text = batch["trg_text"][0]
                # predicted text
                model_out_text = tokenizer_trg.decode(model_out.data.tolist())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)

                # per evitare di interferire con il print di training
                print_msg('-'*console_width)
                print_msg(f"Source: {source_text}")
                print_msg(f"Target: {target_text}")
                print_msg(f"Predicted: {model_out_text}")

                # bleu metrics, charErrorRate, WordErrorRate
                print_msg(f"BLEU score: {bleu_score(model_out_text, target_text)}")

    # per ora fatta ad cazzum
    print_msg(f"[-----> Mean eval loss: {total_loss.mean()}  ]")
    model.train()




# questo ds non e' diviso in train e validation, percio' bisogna fare lo split a mano
if SG_DATASET:
    dataset_raw = load_dataset(
        "csv", 
        data_files="torchic/data/datasets/SG_en-it/SG_transaltion_dataset.csv", 
        delimiter="|",                      # specificaro il separatore
        column_names=["name", "en", "it"],  # (ri)nomino le colonne
        usecols=["en", "it"],               # ignoro la colonna 'name'
        split="train"                       # carico tutto come unico split
    )
else:
    dataset_raw = load_dataset(model_config["opus_books"], f"{model_config['lang_src']}-{model_config['lang_trg']}", split="train", cache_dir=None)
    dataset_raw = dataset_raw.map(
        lambda x: {
            "en": x["translation"][model_config["lang_src"]],
            "it": x["translation"][model_config["lang_trg"]],
        },
        remove_columns=["translation"]
    )

# per prenderne solo una sottoparte
if DS_LEN:
    dataset_raw = dataset_raw.shuffle()
    dataset_raw = dataset_raw.take(DS_LEN)


tokenizer_src = get_or_build_tokenizer(model_config, dataset_raw, model_config["lang_src"])
tokenizer_trg = get_or_build_tokenizer(model_config, dataset_raw, model_config["lang_trg"])

train_test_split = dataset_raw.train_test_split(test_size=0.025)
train_ds = BilingualDataset(train_test_split["train"],
                            tokenizer_src,
                            tokenizer_trg,
                            model_config["lang_src"],
                            model_config["lang_trg"],
                            model_config["seq_len"],
                            dynamic_padding=DYNAMIC_PADDING)
val_ds = BilingualDataset(train_test_split["test"],
                            tokenizer_src,
                            tokenizer_trg,
                            model_config["lang_src"],
                            model_config["lang_trg"],
                            model_config["seq_len"],
                            dynamic_padding=DYNAMIC_PADDING)

print(f"Frasi di eval: {len(val_ds)}")

# controllo la lunghezza massima delle frasi
max_len_src = 0
max_len_trg = 0

# tokenizzo tutto il dataset
for sentence_pair in dataset_raw:
    src_ids = tokenizer_src.encode(sentence_pair[model_config["lang_src"]]).ids
    trg_ids = tokenizer_trg.encode(sentence_pair[model_config["lang_trg"]]).ids

    max_len_src = max(max_len_src, len(src_ids))
    max_len_trg = max(max_len_trg, len(trg_ids))

print(f"Max length of src lang: {max_len_src}")
print(f"Max length of trg lang: {max_len_trg}")

train_dataloader = BilingualDataLoader(train_ds, batch_size=model_config["batch_size"], shuffle=True, my_stack=MY_STACK)
val_dataloader = BilingualDataLoader(val_ds, batch_size=1, shuffle=True, my_stack=MY_STACK)


if not os.path.exists(model_config["model_folder"]):
    os.makedirs(model_config["model_folder"])

print(tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size())
total_batches = len(train_dataloader)
initial_epoch = 0


model = TransformerV2(
    d_model=model_config["d_model"],
    encoder_vocab_size=tokenizer_src.get_vocab_size(),
    dec_vocab_size=tokenizer_trg.get_vocab_size(),
    max_len=model_config["seq_len"],
    allow_fusion=ALLOW_FUSION
)
optimizer = AdamV4(model.parameters(), lr=model_config["lr"], beta2=0.98, eps=10**-9)
loss_fn = CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=LABEL_SMOOTHING)

# qui mettere logica per caricare modello
if model_config["preload"]:
    # qua devi caricare il modello e lo stato dell'optim
    #model, optimizer = load_checkpoint("checkpoint_e8", device="cuda")
    pass

print(model.number_of_params())

if PROFILER:
    import cProfile, pstats
    import subprocess

    profiler = cProfile.Profile()
    profiler.enable()

model.train()
for epoch in range(initial_epoch, model_config["num_epochs"]):
    batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch+1:02d}")
    step = 0

    for batch in batch_iterator:
        step += 1

        if WARMUP:
            dynamic_lr = transformer_lr(step, model_config["d_model"], WARMUP)
            optimizer.lr = dynamic_lr

        # frase
        encoder_input = batch["encoder_input"] # (batch, seq_len)
        # frase target
        decoder_input = batch["decoder_input"] # (batch, seq_len)
        # NOTE: le seconde dim delle maschere sono per il broadcast
        # a num_heads. Entrambe le mask sarano broadcastate
        # a (batch, num_heads, seq_len, seq_len)
        # nasconde solo i token di padding
        encoder_mask = batch["encoder_mask"]   # (batch, 1, 1, seq_len)
        # nasconde anche i token successivi al token in esame
        decoder_mask = batch["decoder_mask"]   # (batch, 1, seq_len, seq_len)
        target = batch["target"] # (batch, seq_len)

        #print(encoder_mask.shape(), decoder_mask.shape())
        #print(type(encoder_input))
        #print(encoder_input.dtype, decoder_input.dtype, encoder_mask.dtype, decoder_mask.dtype, target.dtype)
        #print(encoder_input.shape(), decoder_input.shape(), encoder_mask.shape(), decoder_mask.shape(), target.shape())
        #print((get_nbytes(encoder_input)+get_nbytes(decoder_input)+get_nbytes(encoder_mask)+get_nbytes(decoder_mask)+get_nbytes(target))/1024)
        #exit()

        prediction = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

        # (batch, seq_len, trg_vocab_size) -> (batch * seq_len, trg_vocab_size)
        # in entrambi i tensor moltiplico le dimensioni di batch e seq_len
        loss = loss_fn(prediction.view(-1, tokenizer_trg.get_vocab_size()), target.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.data : 6.3f}"})

        if DRAW_GRAPH:
            draw_graph(prediction, format="svg")#, complete_format=False)
            exit()

        loss.backward()

        if ACCUMULATION_STEPS:
            if step % ACCUMULATION_STEPS == 0 or step == total_batches-1:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
        
        if EARLY_STOP and EARLY_STOP == step:
            break
    
    # salva un checkpoint alla fine di ogni epoch
    run_validation(model, val_dataloader, tokenizer_trg, model_config["seq_len"], lambda msg: batch_iterator.write(msg), num_examples=1)
    save(model, optimizer, f"checkpoint_e{epoch}", epoch=str(epoch))
    break

if PROFILER:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
    # Remove dir names
    stats.strip_dirs()
    stats.print_stats()
    profiler.dump_stats("profiler_data.prof")

    subprocess.run(["snakeviz", "profiler_data.prof"])