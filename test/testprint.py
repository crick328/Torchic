import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic.nn.models.transformer import create_transformer

import torchic
import torch.nn as nn


# Definizione di un modello Transformer
model = nn.Transformer(
    d_model=512, 
    nhead=8, 
    num_encoder_layers=6, 
    num_decoder_layers=6, 
    dim_feedforward=2048, 
    dropout=0.1, 
    activation='relu'
)

# Stampa del modello
print(model)


my_model = create_transformer(
    encoder_vocab_size=30,
    dec_vocab_size=40,
    d_model=512,
    max_len=69
)
print(my_model)