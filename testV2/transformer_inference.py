from preset import torchic
from torchic.nn.models.save_load import load_architecture
#from torchic.tools.sequences import causal_mask
from torchic import Tensor
from torchic.tools import helpers
from tokenizers import Tokenizer
from torchic.tools.sequences import greedy_decode
from os.path import join
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

torchic.set_default_device("cuda")


VISUALIZE_ENCODER_HEADS = True
NORMALIZE_ATTN = True
SEQ_LEN = 320
# se vuoi testare un modello specifico, metti il nome e verra'
# runnato solo quello
# I'm a terrible teacher, I should starve under a bridge for my sins
MODEL_NAME = "checkpoint_best"#"checkpoint_best"#"checkpoint_e44"
MODEL_FOLDER = "SG_BPE_Take3"#"SG_BPE_Take3"#"SG_BPE"#"SG_WordLevel"
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


    for name, param in model.parameters(trainable_only=False, with_names=True):
        assert param.device == 'cuda'
        assert type(param.data) == cp.ndarray, f"{name}, {type(param.data)}"


    encoder_input = encoder_input.view(1, encoder_input.size())
    assert encoder_input.shape()[0] == 1, "Validation batch size must be 1"

    # predicted_tokens
    model_out, _raw_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_it, SEQ_LEN)

    # predicted text
    model_out_text = tokenizer_it.decode(model_out.data.tolist(), skip_special_tokens=True)
    print(f"Frase inserita: {text}")
    print(f"Traduzione: {model_out_text}")

    return encoder_input, model_out


# visualizzazione embeddings?
# visualizzazione n parole piu' probabili?
# NON HAI DISATTIVATO IL DROPOUT PERCHE' NON HAI LA EVAL MODE!

# - implementa .train() .eval() (o come fa pytorch), magari anche introducendo
#   with no_grad (implementalo a livello di Module)
#   (self.training lo metterei solo nei layer che servono, senza scomodare Module, che
#   lo settera' solo se self c'e' l'ha, senno' si fotte. Puo' essere una flag
#   di classe dei tensori? torchic.no_grad() modificherebbe un setting in settings.py,
#   ma, al posto di andarlo a leggere ad ogni operazione, va subito ad aggiornare
#   la flag dei tensori. NO, mi piace la soluzione di GPT)
#   (per no_grad, hai azzeccato: basta mettere una flag prima di valutare se
#   i parents vogliono il grad)
# - voglio visualizzare gli attention weights e usare il modello



def visualize_attention_heads(attn_weights, query_tokens, key_tokens=None, normalize=False, title="Attention Heads"):
    """
    Visualizes attention weights for each head with token labels.
    
    Args:
        attn_weights (Tensor): The attention weights (shape [1, heads, seq_len, seq_len]).
        query_tokens (list): The list of query tokens.
        key_tokens (list, optional): The list of key tokens (for cross-attention).
        title (str): Title for the plot.
    """
    # (1, num_heads, seq_len_q, seq_len_k) -> (num_heads, seq_len_q, seq_len_k)
    attn = attn_weights.squeeze(0).detach().to("cpu").numpy()
    #num_heads, seq_len_q, seq_len_k = attn.shape
    num_heads, seq_len_y, seq_len_x = attn.shape

    x_labels = query_tokens[:seq_len_x]
    # se non ricevo dei key tokens, assumo si tratti di una self-attention
    y_labels = key_tokens[:seq_len_y] if key_tokens else query_tokens[:seq_len_y]

    #print(attn.shape)
    #print(x_labels, y_labels)

    # Create a subplot for each head
    #fig, axes = plt.subplots(1, num_heads, figsize=(15, 6), constrained_layout=True, sharey=True, sharex=True)
    #fig.suptitle(title, fontsize=16)

    # Create a subplot for each head
    rows = (num_heads + 3) // 4  # Max 4 heads per row
    fig, axes = plt.subplots(rows, min(4, num_heads), figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()[:num_heads]  # Ensure axes is iterable
    fig.suptitle(title, fontsize=16)

    #imshow_kwargs = {"vmin": 0, "vmax": 1} if normalize else {"vmin": None, "vmax": None}
    imshow_kwargs = {"vmin": np.min(attn), "vmax": np.max(attn)} if normalize else {}
    
    # per non far incazzare matplotlib, rendo axes un iterable
    if num_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.set_title(f"Head {i+1}")
        im = ax.imshow(attn[i, :, :], cmap='viridis', **imshow_kwargs)#, aspect='auto')
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
        ax.set_yticklabels(y_labels, fontsize=8)
    
    # Add common labels for the entire figure
    fig.supxlabel("Key Tokens", fontsize=12)
    fig.supylabel("Query Tokens", fontsize=12)

    # Add a colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical")#, shrink=0.8, aspect=30)
    cbar.set_label("Attention Score", fontsize=12)

    plt.show()



if __name__ == "__main__":
    while True:
        user_input = input("Insert a sentence to translate:")
        encoder_input, model_out = inference(user_input)

        if encoder_input is None and model_out is None:
            continue

        encoder_tokens = [tokenizer_en.id_to_token(int(tok)).replace("Ġ", "", 1) for tok in encoder_input.squeeze(0)]
        decoder_tokens = [tokenizer_it.id_to_token(int(tok)).replace("Ġ", "", 1) for tok in model_out]

        print(f"Encoder tokens: {encoder_tokens}")
        print(f"Decoder tokens: {decoder_tokens}")

        #encoder_tokens = tokenizer_en.decode(encoder_input.squeeze(0).data.tolist(), skip_special_tokens=True)
        #decoder_tokens = tokenizer_it.decode(model_out.data.tolist(), skip_special_tokens=True)

        #print(f"Encoder tokens: {encoder_tokens[0]}")
        #print(f"Decoder tokens: {decoder_tokens}")

        attn_weights = model.get_saved_attn_weights()

        if VISUALIZE_ENCODER_HEADS:
            # Visualizza le matrici di attenzione per l'encoder
            for layer_idx, attentions in attn_weights["encoder"].items():
                for attn_type, weights in attentions.items():
                    visualize_attention_heads(
                        attn_weights=weights,
                        query_tokens=encoder_tokens,
                        title=f"Encoder Layer {layer_idx} - {attn_type.capitalize()} Attention",
                        normalize=NORMALIZE_ATTN
                    )

        # Visualizza le matrici di attenzione per il decoder
        for layer_idx, attentions in attn_weights["decoder"].items():
            for attn_type, weights in attentions.items():

                if attn_type == "self_attention":
                    visualize_attention_heads(
                        attn_weights=weights,
                        query_tokens=decoder_tokens,  # Decoder tokens for self-attention
                        title=f"Decoder Layer {layer_idx} - {attn_type.capitalize()} Attention",
                        normalize=NORMALIZE_ATTN
                    )

                elif attn_type == "cross_attention":
                    visualize_attention_heads(
                        attn_weights=weights,
                        query_tokens=encoder_tokens,
                        key_tokens=decoder_tokens,
                        title=f"Decoder Layer {layer_idx} - {attn_type.capitalize()} Attention",
                        normalize=NORMALIZE_ATTN
                    )
