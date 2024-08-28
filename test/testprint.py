import torch
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