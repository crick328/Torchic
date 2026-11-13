import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torchic
from torchic import Tensor, nn, draw_graph
from torchic.nn.models import transformer
from torchic.data.dataloaders.base_classes import DataLoader
import numpy as np


BATCH = 4
SEQ_LEN = 2
VOCAB_SIZE = 3
D_MODEL = 2
DROPOUT = 0.0
HEAD_NUM = 2



def param_state(module: nn.Module, _name: str=""):
    for name, param in module:
        if isinstance(param, Tensor) and param.requires_grad:
            print(f"{_name}.{name}:\n{param.grad}")
        elif isinstance(param, nn.Module):
            param_state(param, name)


list_of_sentences = []

for _ in range(0, BATCH):
    sentence_pair = {
        'encoder_input': torchic.randint((SEQ_LEN, ), high=VOCAB_SIZE, dtype=torchic.int64),
        'decoder_input': torchic.randint((SEQ_LEN, ), high=VOCAB_SIZE, dtype=torchic.int64),
        'target': torchic.randint((SEQ_LEN, ), high=VOCAB_SIZE, dtype=torchic.int64),
        'src_text': ["src_text"],
        'trg_text': ["trg_text"]
    }

    encoder_mask = Tensor(sentence_pair['encoder_input'].data != 1, dtype=torchic.int64)
    # aggiungo le dim di batch e d_model (non sicuro di quest'ultimo)
    encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)

    decoder_mask = Tensor(sentence_pair['decoder_input'].data != 1, dtype=torchic.int64)
    decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)
    decoder_mask.data = decoder_mask.data & Tensor(np.triu( np.ones([1, sentence_pair['decoder_input'].size(0), sentence_pair['decoder_input'].size(0),]) ), dtype=torchic.int64).data == 0
    
    sentence_pair["encoder_mask"] = encoder_mask
    sentence_pair["decoder_mask"] = decoder_mask

    list_of_sentences.append(sentence_pair)

data = DataLoader.stack_on_batch_dim(list_of_sentences)


# modello
embeddings = transformer.TokenEmbeddings(D_MODEL, VOCAB_SIZE)
positional_encoding = transformer.PositionalEncoding(D_MODEL, SEQ_LEN, DROPOUT)
self_attention = transformer.MultiHeadAttention(D_MODEL, HEAD_NUM, DROPOUT)


stacked_encoder_input = data["encoder_input"] # (batch, seq_len)
stacked_decoder_input = data["decoder_input"] # (batch, seq_len)
stacked_encoder_mask = data["encoder_mask"]   # (batch, 1, 1, seq_len)
stacked_decoder_mask = data["decoder_mask"]   # (batch, 1, seq_len, seq_len)
stacked_target = data["target"]               # (batch, seq_len)


# forward
output = embeddings(stacked_encoder_input)
output = positional_encoding(output)
output = self_attention(output, output, output, stacked_encoder_mask)

#draw_graph(output)
output.mean().backward()


for param in embeddings.parameters():
    print(param.grad)

print("========================")

for param in positional_encoding.parameters():
    print(param.grad)

print("========================")
# alcune volte i primi quattro grad sono nulli
param_state(self_attention)