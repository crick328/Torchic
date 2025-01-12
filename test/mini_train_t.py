import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
import torchic
from torchic import Tensor, nn, draw_graph
from torchic.nn.models import transformer, transformer_config
from torchic.data.dataloaders.base_classes import DataLoader
import numpy as np


sys.setrecursionlimit(1500)
config = transformer_config.get_config()

BATCH = config['batch_size']
SEQ_LEN = config['seq_len']
ENC_VOCAB_SIZE = 15698
DEC_VOCAB_SIZE = 22463
D_MODEL = 512
DROPOUT = 0.2
HEAD_NUM = 8



def param_state(module: nn.Module, _name: str=""):
    for name, param in module:
        if isinstance(param, Tensor) and param.requires_grad:
            print(f"{_name}.{name}:\n{param.grad}")
        elif isinstance(param, nn.Module):
            param_state(param, name)


def get_model(model_config, vocab_src_len, vocab_trg_len):
    model = transformer.Transformer(d_model=model_config["d_model"],
                        encoder_vocab_size=vocab_src_len,
                        dec_vocab_size=vocab_trg_len,
                        max_len=model_config["seq_len"])
    return model


list_of_sentences = []

for _ in range(0, BATCH):
    sentence_pair = {
        'encoder_input': torchic.randint((SEQ_LEN, ), high=ENC_VOCAB_SIZE, dtype=torchic.int64),
        'decoder_input': torchic.randint((SEQ_LEN, ), high=DEC_VOCAB_SIZE, dtype=torchic.int64),
        'target': torchic.randint((SEQ_LEN, ), high=DEC_VOCAB_SIZE, dtype=torchic.int64),
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


model = get_model(config, ENC_VOCAB_SIZE, DEC_VOCAB_SIZE)

loss_fn = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)


stacked_encoder_input = data["encoder_input"] # (batch, seq_len)
stacked_decoder_input = data["decoder_input"] # (batch, seq_len)
stacked_encoder_mask = data["encoder_mask"]   # (batch, 1, 1, seq_len)
stacked_decoder_mask = data["decoder_mask"]   # (batch, 1, seq_len, seq_len)
stacked_target = data["target"]               # (batch, seq_len)


prediction = model(
    stacked_encoder_input,
    stacked_decoder_input,
    stacked_encoder_mask,
    stacked_decoder_mask
    )


loss = loss_fn(prediction.view(-1, DEC_VOCAB_SIZE), stacked_target.view(-1))

loss.backward()


with open("bacwkard.txt", 'a') as file:
    for name, param in model.parameters(with_names=True):
        file.write(f"{name}, {param.grad}\n\n")