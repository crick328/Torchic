from preset import torchic, compare_tensors
from torchic.data.dataloaders.transformer_dataset import BilingualDataset as MyBD
from torchic.data.dataloaders.base_classes import DataLoader as MyDL
from jamil.jamil_dataset import BilingualDataset as JamilBD
from torch.utils.data import DataLoader as JamilDL

from pathlib import Path
from tqdm import tqdm
from torch import int64
import numpy as np
import os

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torchic.data.dataloaders import transformer_dataset
from torchic import set_default_device
transformer_dataset.DTYPE = torchic.int64

NUM_SAMPLES = 2000
BATCH_SIZE = 20
DYNAMIC_MASKING = True
MY_STACK = True
DEBUG_COMPARISON = False
BATCH_TO_SKIP = 0
COMPARE_MY_DSS = False
set_default_device("cpu")

# NOTE: ho modificato il dtype delle maschere, la loro riassegnazione
# (anziche' lavorare sul .data) e fare mask==False. Eppure, ora
# i tempi di training sono saliti da 22 a 25 minuti. Che sia
# colpa di una di queste modifiche o di quel runtime?



def get_or_build_tokenizer(ds, lang):
    tokenizer_path = f"tokenizer_{lang}.json"

    if not os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]



# ==== Codice in comune ====
# It only has the train split, so we divide it overselves
ds_raw = load_dataset("opus_books", f"en-it", split='train')

# Build tokenizers
tokenizer_src = get_or_build_tokenizer(ds_raw, "en")
tokenizer_tgt = get_or_build_tokenizer(ds_raw, "it")


train_ds_raw = ds_raw.take(NUM_SAMPLES)


# ==== Inizio zona di test ====

jamil_train_ds = JamilBD(train_ds_raw, tokenizer_src, tokenizer_tgt, "en", "it", 320)
my_train_ds = MyBD(train_ds_raw, tokenizer_src, tokenizer_tgt, "en", "it", 320, dynamic_masking=DYNAMIC_MASKING)

jamil_train_dataloader = JamilDL(jamil_train_ds, batch_size=BATCH_SIZE, shuffle=False)
my_train_dataloader = MyDL(my_train_ds, batch_size=BATCH_SIZE, shuffle=False, my_stack=MY_STACK)
my_alter_dataloader = MyDL(my_train_ds, batch_size=BATCH_SIZE, shuffle=False, my_stack=not MY_STACK)

jamil_batch_iterator = tqdm(jamil_train_dataloader, desc=f"Testing...")
my_batch_iterator = tqdm(my_train_dataloader, desc=f"Testing...")
alter_batch_iterator = tqdm(my_alter_dataloader, desc=f"Testing...")


if COMPARE_MY_DSS:
    for batch in zip(alter_batch_iterator, my_batch_iterator):
        j_encoder_input = batch[0]['encoder_input'] # (b, seq_len)
        j_decoder_input = batch[0]['decoder_input'] # (B, seq_len)
        j_encoder_mask = batch[0]['encoder_mask'] # (B, 1, 1, seq_len)
        j_decoder_mask = batch[0]['decoder_mask'] # (B, 1, seq_len, seq_len)
        j_target = batch[0]["target"] # (batch, seq_len)

        my_encoder_input = batch[1]['encoder_input'] # (b, seq_len)
        my_decoder_input = batch[1]['decoder_input'] # (B, seq_len)
        my_encoder_mask = batch[1]['encoder_mask'] # (B, 1, 1, seq_len)
        my_decoder_mask = batch[1]['decoder_mask'] # (B, 1, seq_len, seq_len)
        my_target = batch[1]["target"] # (batch, seq_len)

        assert np.allclose(j_encoder_input.data, my_encoder_input.data)
        assert np.allclose(j_decoder_input.data, my_decoder_input.data)
        assert np.allclose(j_encoder_mask.data, my_encoder_mask.data)
        assert np.allclose(j_decoder_mask.data, my_decoder_mask.data)
        assert np.allclose(j_target.data, my_target.data)

count = 0
for batch in zip(jamil_batch_iterator, my_batch_iterator):
    if count < BATCH_TO_SKIP:
        count += 1 
        continue

    j_encoder_input = batch[0]['encoder_input'] # (b, seq_len)
    j_decoder_input = batch[0]['decoder_input'] # (B, seq_len)
    j_encoder_mask = batch[0]['encoder_mask'] # (B, 1, 1, seq_len)
    j_decoder_mask = batch[0]['decoder_mask'] # (B, 1, seq_len, seq_len)
    j_target = batch[0]["label"] # (batch, seq_len)

    my_encoder_input = batch[1]['encoder_input'] # (b, seq_len)
    my_decoder_input = batch[1]['decoder_input'] # (B, seq_len)
    my_encoder_mask = batch[1]['encoder_mask'] # (B, 1, 1, seq_len)
    my_decoder_mask = batch[1]['decoder_mask'] # (B, 1, seq_len, seq_len)
    my_target = batch[1]["target"] # (batch, seq_len)

    my_max_enc_len = my_encoder_input.shape()[-1]
    my_max_dec_len = my_decoder_input.shape()[-1]
    

    compare_tensors(j_encoder_input[:, :my_max_enc_len], my_encoder_input)
    compare_tensors(j_decoder_input[:, :my_max_dec_len], my_decoder_input)
    compare_tensors(j_encoder_mask[:, :, :, :my_max_enc_len], my_encoder_mask)
    compare_tensors(j_decoder_mask[:, :, :my_max_dec_len, :my_max_dec_len], my_decoder_mask)
    compare_tensors(j_target[:, :my_max_dec_len], my_target)

    if DEBUG_COMPARISON: 
        print("Encoder input")
        print(j_encoder_input[:, :my_max_enc_len], f"\n{my_encoder_input}")
        print("\nDecoder input")
        print(j_decoder_input[:, :my_max_enc_len], f"\n{my_decoder_input}")
        print("\nEncoder mask")
        print(j_encoder_mask[:, :, :, :my_max_enc_len], f"\n{my_encoder_mask}")
        print("\nDecoder mask")
        print(j_decoder_mask[:, :, :my_max_dec_len, :my_max_dec_len], f"\n{my_decoder_mask}")
        print("\nTarget")
        print(j_target[:, :my_max_dec_len], f"\n{my_target}")

        break