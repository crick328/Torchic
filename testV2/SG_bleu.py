from preset import torchic
from torchic.nn.models.save_load import load_architecture
#from torchic.tools.sequences import causal_mask
from torchic import Tensor
from torchic.tools import helpers
from tokenizers import Tokenizer
from torchic.tools.sequences import greedy_decode
from os.path import join
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

torchic.set_default_device("cuda")


VISUALIZE_ENCODER_HEADS = True
NORMALIZE_ATTN = True
SEQ_LEN = 135
# se vuoi testare un modello specifico, metti il nome e verra'
# runnato solo quello
MODEL_NAME = "checkpoint_e48"#"checkpoint_e48"#"checkpoint_e44"
MODEL_FOLDER = "SG_BPE"#"SG_BPE"#"SG_WordLevel"
ZOO_PATH = r"D:\Model zoo"
#test_sentence = "My name is a secret"
#test_sentence = "I am... a waste of space"
#test_sentence = "merry Christmas!"

# tokenizers-0.21.0
tokenizer_en = Tokenizer.from_file(join(ZOO_PATH, MODEL_FOLDER, "tokenizer_en.json"))
tokenizer_it = Tokenizer.from_file(join(ZOO_PATH, MODEL_FOLDER, "tokenizer_it.json"))

sos_token = Tensor([tokenizer_en.token_to_id("[SOS]")])
eos_token = Tensor([tokenizer_en.token_to_id("[EOS]")])
pad_token = Tensor([tokenizer_en.token_to_id("[PAD]")])


model: torchic.nn.Transformer
model = load_architecture(join(ZOO_PATH, MODEL_FOLDER, MODEL_NAME))
model.set_save_attn_weights(True)
model.eval()
print(model.__repr__())


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


def inference(text: str):
    # converto la frase in tokens
    tok_sentence = Tensor(tokenizer_en.encode(text).ids)
    # aggiungo i tokens speciali
    encoder_input = helpers.cat(
        [
            sos_token,
            tok_sentence,
            eos_token,
            #pad_token.expand(SEQ_LEN - len(tok_sentence) -2,),
        ],
        dtype=sos_token.dtype
    )
    # creo la mask in modo che non crei self-attention con i pad tokens
    encoder_mask = Tensor(encoder_input.data != pad_token.data)
    encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)


    if encoder_input.all() == 0:
        unk_words = []
        indices = encoder_input._device.where(encoder_input.data == 0)
        for i in indices[0]:
            unk_words.append(text.split()[int(i-1)])

        print(
            f"Unknown words found: {unk_words}"
        )
        return None, None


    encoder_input = encoder_input.view(1, encoder_input.size())
    assert encoder_input.shape()[0] == 1, "Validation batch size must be 1"

    # predicted_tokens
    model_out, _raw_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_it, SEQ_LEN)

    # predicted text
    model_out_text = tokenizer_it.decode(model_out.data.tolist(), skip_special_tokens=True)

    return model_out_text, model_out




full_dataset = pd.read_csv(r'C:\Users\admin\Desktop\NN Lab\Projects\Torchic\Torchic 3.1\torchic\datas\SG_en-it\SG_transaltion_dataset.csv', sep='|', header=0, index_col=False).values
# (ds_len, cols) -> (ds_len, 1), (ds_len, 1) 
names, en_sentences, it_sentences = np.split(full_dataset, 3, axis=-1)

# (ds_len, 1) -> (ds_len,)
names = names.squeeze(-1).tolist()
en_sentences = en_sentences.squeeze(-1).tolist()
it_sentences = it_sentences.squeeze(-1).tolist()



for en_sentence, it_sentence in tqdm(zip(en_sentences, it_sentences), total=len(en_sentences)):
    model_out_text, _model_out = inference(en_sentence)

    if model_out_text is None:
        continue

    score = bleu_score(model_out_text, it_sentence)

    if score < 0.4:
        print(f"BLEU score: {score :.2f}")
        print(f"Source: {en_sentence}")
        print(f"Target: {it_sentence}")
        print(f"Predicted: {model_out_text}")

        #_ = input("Press to continue")
