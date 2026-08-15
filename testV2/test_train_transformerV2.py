#import sys
#sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Torchic_repo")
from torchic import set_default_device, set_default_precision
# CUDA settings
from torchic import set_num_backward_workers, set_time_threshold_ms
from torchic.tools.visualization.graphs import draw_graph
from torchic.tools.code_analysis.mem import get_nbytes
from torchic.nn.models.transformer_config import get_config
from torchic.data.dataloaders.transformer_dataset import BilingualDataset, DataLoaderForLLM
#from torchic.data.dataloaders.base_classes import DataLoader
from torchic.nn.optim import AdamV3, AdamV4
from torchic.nn.loss import CrossEntropyLoss
from torchic.nn.models.save_load import save, load_checkpoint
from torchic.nn.models.transformer import create_transformer, TransformerV2
from torchic.tools.sequences import greedy_decode
from torchic.settings import set_seed

from datasets import load_dataset # pip install datasets
from tokenizers import Tokenizer # pip install tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


import os
import random
import math

from tqdm import tqdm

# prima di far partire un training serio, farei due cose:
# 1) implementazione reload modello + checkpoint
# 2) validation test perche' la loss non e' cosi' esplicativa

# Questo codice va rifinito

set_seed(42)

EARLY_STOP = False
ACCUMULATION_STEPS = False
PROFILER = False
MEM_TRACE = False
WARMUP = False
LABEL_SMOOTHING = 0.1
DROPOUT = 0.1
DS_LEN = False
DYNAMIC_MASKING = False
MY_STACK = True
LEGACY_MODEL = False
ALLOW_FUSION = True
SG_DATASET = False

DRAW_GRAPH=False


set_default_device("cpu")
set_default_precision(32)
set_num_backward_workers(4)
set_time_threshold_ms(5)



# NOTE: controlla bene che TUTTE le operazioni avvengano sullo
# stesso device!
# NOTE: sei sicuro che non ci siano problemi a salvare i parametri
# del modello anche se si trova su cuda? La funzione di safetensors gestisce
# tutto bene? Controlla con un modello di test di classification!
# NOTE: gli embeddings di dim 512 dovrebbero essere overkill.
# GPT dice che si sceglie un valore <= sqrt(vocab_size) che si arrotonda
# alla piu' vicina potenza di 2.
# A questo segue la dimensione del feed forward. Solitamente si fa un
# embedding_size x 4.

# NOTE: bisogna plottare le loss con un grafico insieme al BLEU.
# Le classe Metrics che avevi fatto sono ancora buone?

# NOTE: OCCHIO che con il dynamic_batching non ha troppo senso mettere
# le info della pesantezza computazionale del grafo in cache! E' comunque
# meglio di andare a muzzo con una euristica random, pero' occhio


# corretta
def transformer_lr(step_num: int, d_model: int, warmup_steps: int) -> float:
    return d_model**-0.5 * min(step_num**-0.5, step_num * warmup_steps**-1.5)


# metterei tutti gli scores in un file a parte
def bleu_score(pred_sentence: str, trg_sentence: str, max_ngram_len: int=4) -> float:
    def extract_ngrams(words, ngram_len):
        '''
        Returns the sentence divided in n-grams of ngram_len.
        '''
        return [tuple(words[i:i+ngram_len]) for i in range(len(words) - ngram_len + 1)]
    
    def calc_clipped_precision(n: int):
        '''
        Returns the clipped precision of the given translation using
        n-grams of len n.
        '''
        # suddivido i tokens in n-grams (sequenze di n tokens consecutivi)
        # NOTE: l'n-gram successivo include l'ultimo token dell'n-gram
        # precedente (se n>1 ovviamente)
        pred_ngrams = extract_ngrams(pred_sentence, n)
        trg_ngrams = extract_ngrams(trg_sentence, n)

        # il bleu conta gli n-grams che matchano per calcolare lo score. Pero',
        # e' importante che il punteggio non venga inflazionato da n-grams generati
        # erroneamente piu' volte (trg: ["ciao", "Manu"], pred: ["ciao", "ciao", "ciao"].
        # Non devo dare il triplo dei punti perche' ci sono 3 n-grams che matchano)

        # prendo gli n-grams generati dal modello e non quelli della traduzione
        # in modo da avere gia' un count di 0 per quelli non predetti
        unique_ngrams = set(pred_ngrams)
        clipped_counts = {}

        for ngram in unique_ngrams:
            # per ogni n-gram unico, conto le sue occorrenze in entrambe le
            # frasi e memorizzo il count piu' basso (per evitare che prediction
            # sbagliate inflazionino il punteggio). Se la rete ha lavorato bene,
            # i due minimi coincidono, altrimenti il count nella frase predetta e'
            # (erroneamente) piu' elevato
            clipped_counts[ngram] = min(pred_ngrams.count(ngram), trg_ngrams.count(ngram))

        return sum(clipped_counts.values()) / max(1, len(pred_ngrams))
    

    # calcolo la clipped precision per ogni lunghezza di n-gram, da
    # 1 a max_ngram_len
    precisions = []
    for i in range(1, max_ngram_len+1):
        precisions.append(calc_clipped_precision(i))

    pred_len = len(pred_sentence)
    trg_len = len(trg_sentence)

    # non dovrebbe succedere, ma non si sa mai
    if pred_len == 0:
        return 0.0

    if pred_len > trg_len:
        bp = 1
    else:
        bp = math.exp(1 - trg_len/pred_len)

    # Geometric Average Precision
    gap = 1
    for pn in precisions:
        gap *= pn**(1/4)
    
    return bp * gap


def run_validation(model, validation_ds, tokenizer_trg, max_len, print_msg, num_examples=2):
    #model.eval()
    count = 0
    console_width = 80

    source_texts = []
    expected = []
    predicted = []

    # ma scusa, se il modello e' in eval, i gradienti
    # non dovrebbero essere disattivati di default?
    # with torch.no_grad()
    for batch in validation_ds:
        count += 1
        
        # frase
        encoder_input = batch["encoder_input"] # (batch, seq_len)
        # nasconde solo i token di padding
        encoder_mask = batch["encoder_mask"]   # (batch, 1, 1, seq_len)

        assert encoder_input.shape()[0] == 1, "Validation batch size must be 1"

        # predicted_tokens
        model_out, raw_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_trg, max_len)

        print(raw_out.shape())
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

        if count == num_examples:
            break
    

    # Perche' sono gia' None?
    #print(list(model.parameters(trainable_only=True))[-1].grad_fn)
    for param in model.parameters():
        param.grad_fn=None
    #print(list(model.parameters(trainable_only=True))[-1].grad_fn)


def get_all_sentences(dataset, lang):
    # dataset: lista di dict con coppia frase {it: eng}
    for item in dataset:
        yield item[lang]


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





def get_dataset(model_config):
    # questo ds non e' diviso in train e validation, percio' bisogna fare lo split a mano
    if SG_DATASET:
        dataset_raw = load_dataset(
            "csv", 
            data_files="torchic\datas\SG_en-it\SG_transaltion_dataset.csv", 
            delimiter="|",                      # specificaro il separatore
            column_names=["name", "en", "it"],  # (ri)nomino le colonne
            usecols=["en", "it"],               # ignoro la colonna 'name'
            split="train"                       # carico tutto come unico split
        )
    else:
        dataset_raw = load_dataset("opus_books", f"{model_config['lang_src']}-{model_config['lang_trg']}", split="train", cache_dir=None)
        dataset_raw = dataset_raw.map(
            lambda x: {
                "en": x["translation"][model_config["lang_src"]],
                "it": x["translation"][model_config["lang_trg"]],
            },
            remove_columns=["translation"]
        )

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
                                dynamic_masking=DYNAMIC_MASKING)
    val_ds = BilingualDataset(train_test_split["test"],
                                tokenizer_src,
                                tokenizer_trg,
                                model_config["lang_src"],
                                model_config["lang_trg"],
                                model_config["seq_len"],
                                dynamic_masking=DYNAMIC_MASKING)

    print(len(val_ds))
    
    #train_ds = train_ds.subset(1/1000)

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

    train_dataloader = DataLoaderForLLM(train_ds, batch_size=model_config["batch_size"], shuffle=True, my_stack=MY_STACK)
    val_dataloader = DataLoaderForLLM(val_ds, batch_size=1, shuffle=True, my_stack=MY_STACK)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg







def train_model(model_config):
    initial_epoch = 0

    if not os.path.exists(model_config["model_folder"]):
        os.makedirs(model_config["model_folder"])

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_dataset(model_config)
    vocab_src_len, vocab_trg_len = tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()
    total_batches = len(train_dataloader)
    print(total_batches)
    
    '''
    model = Transformer(d_model=model_config["d_model"],
                        encoder_vocab_size=tokenizer_src.get_vocab_size(),
                        dec_vocab_size=tokenizer_trg.get_vocab_size(),
                        max_len=model_config["seq_len"])
    '''
    
    if LEGACY_MODEL:
        model = create_transformer(d_model=model_config["d_model"],
                            encoder_vocab_size=tokenizer_src.get_vocab_size(),
                            dec_vocab_size=tokenizer_trg.get_vocab_size(),
                            max_len=model_config["seq_len"])
    else:
        model = TransformerV2(
            d_model=model_config["d_model"],
            encoder_vocab_size=tokenizer_src.get_vocab_size(),
            dec_vocab_size=tokenizer_trg.get_vocab_size(),
            max_len=model_config["seq_len"],
            allow_fusion=ALLOW_FUSION
        )
    '''
    model = create_transformer(d_model=model_config["d_model"],
                        encoder_vocab_size=tokenizer_src.get_vocab_size(),
                        dec_vocab_size=tokenizer_trg.get_vocab_size(),
                        max_len=model_config["seq_len"],
                        n_layers=1,
                        head_num=1,
                        dim_feed_forward=256,
                        drop_prob=DROPOUT)
    '''
    
    print(vocab_src_len, vocab_trg_len)
    

    #print(model.state_dict())
    #output = open("output.txt", "w")
    #for k, v in model.state_dict().items():
        #output.writelines(f'{k} {v}\n')
    #model = load_architecture("transformer")

    optimizer = AdamV4(model.parameters(), lr=model_config["lr"], beta2=0.98, eps=10**-9)
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=LABEL_SMOOTHING)

    # qui mettere logica per caricare modello
    if model_config["preload"]:
        # qua devi caricare il modello e lo stato dell'optim
        # C:\Users\admin\Desktop\NN Lab\Projects\Torchic\Model zoo\25_11_2024
        #model, optimizer = load_checkpoint("checkpoint_e8", device="cuda")
        pass
        #print(model.encoder0.layers[0].self_attention.query_layer.weights.device)
        # fai che splittare in path e filename
        #model, optimizer = load_checkpoint(r"C:\Users\admin\Desktop\NN Lab\Projects\Torchic\Model zoo\25_11_2024\checkpoint_e1")

    print(model.number_of_params())

    
    for epoch in range(initial_epoch, model_config["num_epochs"]):
        #model.train()
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
            #target = batch["target"] # (batch, seq_len)


            # (batch, seq_len, trg_vocab_size) -> (batch * seq_len, trg_vocab_size)
            # in entrambi i tensor moltiplico le dimensioni di batch e seq_len
            loss = loss_fn(prediction.view(-1, tokenizer_trg.get_vocab_size()), target.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.data : 6.3f}"})

            if DRAW_GRAPH:
                draw_graph(prediction, format="svg")#, complete_format=False)
                exit()

            loss.backward()

            #print(type(list(model.parameters(trainable_only=True))[-1].grad))

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
        #save(model, optimizer, f"checkpoint_e{epoch}", epoch=str(epoch))
        break

    if PROFILER:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
        # Remove dir names
        stats.strip_dirs()
        stats.print_stats()
        profiler.dump_stats("profiler_data.prof")

        subprocess.run(["snakeviz", "profiler_data.prof"])


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