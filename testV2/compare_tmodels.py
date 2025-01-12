from jamil.jamil_transformer import build_transformer
from preset import torchic
from torchic.nn.models.transformer import create_transformer
from torch.nn import Transformer as TorchTransformer


# controlla anche lo shuffle
# la mask la fai bene?
# Se vuoi mimicare le performance di Jamil, allora devi anche
# mimicare modello e lr! Ha usato batch_size=8

D_MODEL = 64
SRC_VOCAB_SIZE = 100
TRG_VOCAB_SIZE = 90
SEQ_LEN = 80
N_LAYERS = 1
NUM_HEADS = 2
DROPOUT = 0.1
D_FF = 128


jt = build_transformer(
    SRC_VOCAB_SIZE,
    TRG_VOCAB_SIZE,
    SEQ_LEN,
    SEQ_LEN,
    D_MODEL,
    N_LAYERS,
    NUM_HEADS,
    DROPOUT,
    D_FF
)

mt = create_transformer(
    D_MODEL,
    SRC_VOCAB_SIZE,
    TRG_VOCAB_SIZE,
    SEQ_LEN,
    DROPOUT,
    N_LAYERS,
    NUM_HEADS,
    D_FF
)

tt = TorchTransformer(
    D_MODEL,
    NUM_HEADS,
    N_LAYERS,
    N_LAYERS,
    D_FF,
    DROPOUT
)


name_mapping = {
    "0": "",
    "weight": "weights",
    "encoder": "encoder0",
    "decoder": "decoder1",
    "src_embed": "encoder0.emb.token_embeddings",
    "self_attention_block": "self_attention",
    "cross_attention_block": "cross_attention",
    "w_q": "query_layer",
    "w_k": "key_layer",
    "w_v": "value_layer",
    "w_o": "output_layer",
    "feed_forward_block": "ffb",
    "linear_1": "linear1",
    "linear_2": "linear2",
    "projection_layer": "linear2",
    "proj": "",
    "norm"

    "src_embed.embedding": "encoder/0/attention/query_weight",
    # Aggiungi altre mappature
}

jt_params = jt.named_parameters()
my_params = mt.parameters(with_names=True)

#print(len(list(jt_params)), len(list(my_params)))


#with open('jt_names.txt', 'w', newline='') as f:
    #f.writelines([f"{row[0]}\n" for row in jt_params])

with open('my_names.txt', 'w', newline='') as f:
    f.writelines([f"{row[0]}\n" for row in my_params])

#with open('tt_names.txt', 'w', newline='') as f:
    #f.writelines([f"{row[0]}\n" for row in tt.named_parameters()])


exit()
for j_param, my_param in zip(jt.named_parameters(), mt.parameters(with_names=True)):
    if j_param[0].startswith("encoder"):
        assert my_param[0].startswith("encoder")

        j_name = j_param[0][17:]
        my_name = None
        #print(j_name)
    print(j_param[0], j_param[1].shape, my_param[0], my_param[1].shape())