import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic.nn.models.transformer import PositionalEncoding, DecoderLayer, LayerNormalization
from torchic.nn.layers.linears import Embedding
from torchic.nn.initializers import normal
from torchic.nn.module import Module, ModuleList
from torchic import Tensor



class LanguageTransformer(Module):
    __slots__ = ("embeddings", "decoder")

    def __init__(
            self, embedding_dim: int, vocab_size:int,
            max_len: int, n_layers: int,
            drop_prob: float=0.1, head_num: int=8,
            dim_feed_forward: int=2048, allow_fusion=True
        ):
        
        self.embeddings = Embedding(vocab_size, embedding_dim, _init_funct=normal)
        self.embedding_dim_sqrt = Tensor(embedding_dim).sqrt()
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len, drop_prob)

        self.layers = ModuleList(
            DecoderLayer(embedding_dim, head_num, dim_feed_forward, drop_prob, allow_fusion=allow_fusion)
            for _ in range(n_layers)
        )
        self.norm = LayerNormalization(embedding_dim)
    
    def __call__(self, dec_input: Tensor, enc_out: Tensor, src_mask: Tensor=None, trg_mask: Tensor=None) -> Tensor:
        dec_input = self.emb(dec_input)

        for layer in self.layers.modules:
            dec_input = layer(dec_input, enc_out, src_mask, trg_mask)
        return self.norm(dec_input)

        
    def __call__(self, x: Tensor, trg_mask: Tensor):
        x = self.embeddings(x) * self.embedding_dim_sqrt
        x = self.positional_encoding(x)

        for layer in self.layers.modules:
            x = layer(x, x, None, trg_mask)
        x = self.norm(x)

        # weight tying: reciclo la matrice di embedding per riproiettare
        # i vettori nello spazio dei tokens
        return x @ self.embeddings.weights.T