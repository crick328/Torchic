import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.0 Beta")
from torchic.nn.models.save_load import load_architecture
from torchic.data.dataloaders.transformer_dataset import causal_mask
from torchic import Tensor
from torchic.tools import helpers
from test_train_transformer import bleu_score, get_or_build_tokenizer
from tokenizers import Tokenizer
import math
import os


SEQ_LEN = 320
# se vuoi testare un modello specifico, metti il nome e verra'
# runnato solo quello
MODEL_NAME = "transformer_6"
test_sentence = "My god I ejaculated"
tokenizer_en = Tokenizer.from_file("tokenizer_en.json")
tokenizer_it = Tokenizer.from_file("tokenizer_it.json")
sos_token = Tensor([tokenizer_en.token_to_id("[SOS]")])
eos_token = Tensor([tokenizer_en.token_to_id("[EOS]")])
pad_token = Tensor([tokenizer_en.token_to_id("[PAD]")])

results = {}



# converto la frase in tokens
tok_sentence = Tensor(tokenizer_en.encode(test_sentence).ids)
# aggiungo i tokens speciali
encoder_input = helpers.cat(
    [
        sos_token,
        tok_sentence,
        eos_token,
        Tensor(pad_token._device.tile(pad_token.data, SEQ_LEN - len(tok_sentence) -2))
    ]
)
# creo la mask in modo che non crei self-attention con i pad tokens
encoder_mask = Tensor(encoder_input.data != pad_token.data)
encoder_mask = encoder_mask.unsqueeze(0).unsqueeze(0)


if encoder_input.all() == 0:
    unk_words = []
    indices = encoder_input._device.where(encoder_input.data == 0)
    for i in indices[0]:
        unk_words.append(test_sentence.split()[int(i-1)])

    raise ValueError(
        f"Unknown words found: {unk_words}"
    )


def test_zoo(zoo_dir):
    for subdir, dirs, files in os.walk(zoo_dir):
        for file in files:
            name, extension = file.split(".")

            if MODEL_NAME:
                if MODEL_NAME != name:
                    continue

            if extension == "safetensors" and "transformer" in name.lower():
                path = os.path.join(subdir, file)
                os.chdir(subdir)
                model = load_architecture(name)

                # ==== Generazione della traduzione ====
                encoder_output = model.encoder0(encoder_input, encoder_mask)

                # inizializzo la frase che deve generare
                decoder_input = Tensor([sos_token.data])

                # do while
                while True:
                    # ?
                    decoder_mask = Tensor(causal_mask(decoder_input, 1))

                    out = model.decoder1(decoder_input, encoder_output, encoder_mask, decoder_mask)
                    # prendo solo la prediction prodotta con l'ultima frase data
                    # (batch_size, n_tokens, trg_vocab_size)
                    prob = model.linear2(out[:, -1])
                    # greedy search
                    _value, next_token = helpers.max(prob, dim=-1)
                    print(decoder_input.shape(), Tensor([next_token]).shape())
                    decoder_input = helpers.cat([decoder_input, Tensor([next_token])], dim=1)

                    # se raggiungo la seq_len massima
                    if decoder_input.size(1) == SEQ_LEN:
                        break
                    # se il modello decide di terminare la frase
                    if next_token == eos_token.data:
                        print("INTERROMPO!")
                        break

                # rimuovo la dimension della batch_size
                predicted_tokens = decoder_input.data.squeeze(0)
                #print(predicted_tokens.data)
                predicted_text = tokenizer_it.decode(predicted_tokens.tolist())

                score = bleu_score(predicted_text, test_sentence)
                model_name = path.split("Model zoo")[-1]
                results[model_name] = score

                print(f"Model {path.split('Model zoo')[-1]} translation: {predicted_text}.\n BLEU score: {score}\n")