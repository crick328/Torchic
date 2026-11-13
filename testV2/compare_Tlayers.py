from jamil import jamil_transformer as jamil_T
from preset import torchic, create_random_tensors, backward_and_compare, equalize_layer, compare_tensors, compare_modules
from torchic.nn.models import transformer as my_T
from torchic.data.dataloaders.bilingual_dataloader import causal_mask
from torchic import randint
import torch
from torch.nn import Transformer as SalsicciaAffumicataBuonaBuona
from torch.nn import Sequential

torch.set_default_dtype(torch.float64)
torchic.set_default_precision(64)
torchic.set_default_device("cpu")


D_MODEL = 4
BATCH_SIZE = 2
SEQ_LEN = 4
NUM_HEADS = 2
D_FF = 2048
DROPOUT = 0.0



# ===== Embeddings =====
print("Testing Embeddings...")
VOCAB_SIZE = 20
jamil_emb = jamil_T.InputEmbeddings(D_MODEL, VOCAB_SIZE)
my_emb = my_T.TokenEmbeddings(D_MODEL, VOCAB_SIZE)

jamil_emb.embedding.weight = torch.nn.Parameter(torch.from_numpy(my_emb.embedding.weights.detach().clone().to("cpu").data), requires_grad=True)
#print(jamil_emb.embedding.weight.shape, my_emb.embedding.weights.shape())

#j_x, my_x = create_random_tensors((BATCH_SIZE, SEQ_LEN))
my_x = randint((BATCH_SIZE, SEQ_LEN), high=VOCAB_SIZE)
j_x = torch.from_numpy(my_x.to("cpu").data)

j_res = jamil_emb(j_x)
my_res = my_emb(my_x)

backward_and_compare(
    [j_res, j_x],
    [my_res, my_x],
    show_graph=False
)
compare_modules(jamil_emb, my_emb, transpose=False)



# ALCUNE VOLTE DA PROBLEMI! Mismatch in data of index 0.
# ===== PositionalEncoding =====
print("Testing PositionalEncoding...")
jamil_PE = jamil_T.PositionalEncoding(D_MODEL, SEQ_LEN, DROPOUT)
my_PE = my_T.PositionalEncoding(D_MODEL, SEQ_LEN, DROPOUT)

j_x, my_x = create_random_tensors((BATCH_SIZE, SEQ_LEN, D_MODEL))

j_res = jamil_PE(j_x)
my_res = my_PE(my_x)


backward_and_compare(
    [j_res, j_x],
    [my_res, my_x]
)



# ===== LayerNormalization =====
print("Testing LayerNorm...")
jamil_LN = jamil_T.LayerNormalization(D_MODEL)
my_LN = my_T.LayerNormalization(D_MODEL, 10 ** -6)

jamil_LN.alpha = torch.nn.Parameter(torch.from_numpy(my_LN.gamma.detach().clone().to("cpu").data), requires_grad=True)
jamil_LN.bias = torch.nn.Parameter(torch.from_numpy(my_LN.bias.detach().clone().to("cpu").data), requires_grad=True)

j_x, my_x = create_random_tensors((BATCH_SIZE, SEQ_LEN, D_MODEL))

# Jamil forward
'''
j_mean = j_x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
# Keep the dimension for broadcasting
j_std = j_x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
# eps is to prevent dividing by zero or when std is very small
j_res = jamil_LN.alpha * (j_x - j_mean) / (j_std + jamil_LN.eps) + jamil_LN.bias

# My forward
my_mean = my_x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
my_std = my_x.std(dim=-1, keepdim=True)
my_res = my_LN.gamma * (my_x - my_mean) / (my_std + my_LN.eps) + my_LN.bias
'''
j_res = jamil_LN(j_x)
my_res = my_LN(my_x)

backward_and_compare(
    #[j_res, j_std, j_mean, j_x],
    #[my_res, my_std, my_mean, my_x],
    [j_res, j_x],
    [my_res, my_x],
    show_graph=False
)
compare_modules(jamil_LN, my_LN)



# ===== FeedForward =====
print("Testing FeedForwardBlock...")
jamil_FFB = jamil_T.FeedForwardBlock(D_MODEL, D_FF, DROPOUT)
my_FFB = my_T.FeedForwardBlock(D_MODEL, D_FF, DROPOUT)

j_x, my_x = create_random_tensors((BATCH_SIZE ,SEQ_LEN, D_MODEL))

jamil_FFB.linear_1.weight = torch.nn.Parameter(torch.from_numpy(my_FFB.linear1.weights.detach().clone().to("cpu").data.T), requires_grad=True)
jamil_FFB.linear_1.bias = torch.nn.Parameter(torch.from_numpy(my_FFB.linear1.bias.detach().clone().to("cpu").data), requires_grad=True)
jamil_FFB.linear_2.weight = torch.nn.Parameter(torch.from_numpy(my_FFB.linear2.weights.detach().clone().to("cpu").data.T), requires_grad=True)
jamil_FFB.linear_2.bias = torch.nn.Parameter(torch.from_numpy(my_FFB.linear2.bias.detach().clone().to("cpu").data), requires_grad=True)

j_res = jamil_FFB(j_x)
my_res = my_FFB(my_x)

backward_and_compare(
    [j_res, j_x],
    [my_res, my_x],
    show_graph=False
)
compare_modules(jamil_FFB, my_FFB)




# ===== MultiHeadAttention =====
print("Testing MultiHeadAttention...")
jamil_MHA = jamil_T.MultiHeadAttentionBlock(D_MODEL, NUM_HEADS, DROPOUT)
my_MHA = my_T.MultiHeadAttention(D_MODEL, NUM_HEADS, DROPOUT, bias=False, clip=False, allow_fusion=False)

j_x, my_x = create_random_tensors((BATCH_SIZE, SEQ_LEN, D_MODEL))

# Jamil nel dataset usa dim 0, ma nel decode usa la 1...
my_mask = causal_mask(my_x, dim=1)
j_mask = torch.from_numpy(my_mask.to("cpu").data)

jamil_MHA.w_q.weight = torch.nn.Parameter(torch.from_numpy(my_MHA.query_layer.weights.detach().clone().to("cpu").data.T), requires_grad=True)
jamil_MHA.w_k.weight = torch.nn.Parameter(torch.from_numpy(my_MHA.key_layer.weights.detach().clone().to("cpu").data.T), requires_grad=True)
jamil_MHA.w_v.weight = torch.nn.Parameter(torch.from_numpy(my_MHA.value_layer.weights.detach().clone().to("cpu").data.T), requires_grad=True)
jamil_MHA.w_o.weight = torch.nn.Parameter(torch.from_numpy(my_MHA.output_layer.weights.detach().clone().to("cpu").data.T), requires_grad=True)

j_res = jamil_MHA(j_x,j_x,j_x, j_mask)
my_res = my_MHA(my_x,my_x,my_x, my_mask)

#print(jamil_MHA.w_q.weight)
#print(my_MHA.query_layer.weights)
#exit()

backward_and_compare(
    [j_res, j_x],
    [my_res, my_x]
)
compare_modules(jamil_MHA, my_MHA)


print("Testing Fused MultiHeadAttention...")
from torchic.tools.helpers import chunk, cat
from torchic import Tensor
import numpy as np
STEP_BY_STEP = True
MANUAL_SPLIT = True

my_MHA = my_T.MultiHeadAttention(D_MODEL, NUM_HEADS, DROPOUT, bias=False, clip=False, allow_fusion=False)
my_fused_MHA = my_T.MultiHeadAttention(D_MODEL, NUM_HEADS, DROPOUT, bias=False, clip=False, allow_fusion=True)

my_fused_MHA.qkv_layer.weights.data[:, :D_MODEL] = my_MHA.query_layer.weights.detach().clone().data
my_fused_MHA.qkv_layer.weights.data[:, D_MODEL:2*D_MODEL] = my_MHA.key_layer.weights.detach().clone().data
my_fused_MHA.qkv_layer.weights.data[:, 2*D_MODEL:] = my_MHA.value_layer.weights.detach().clone().data
my_fused_MHA.output_layer.weights.data = my_MHA.output_layer.weights.detach().clone().data

my_x = torchic.rand((BATCH_SIZE, SEQ_LEN, D_MODEL))
my_fused_x = my_x.detach().clone()

# assert di sanity check
assert np.allclose(my_MHA.query_layer.weights.data, my_fused_MHA.qkv_layer.weights[:, :D_MODEL].data)
assert np.allclose(my_MHA.key_layer.weights.data, my_fused_MHA.qkv_layer.weights[:, D_MODEL:2*D_MODEL].data)
assert np.allclose(my_MHA.value_layer.weights.data, my_fused_MHA.qkv_layer.weights[:, 2*D_MODEL:].data)
assert np.allclose(my_MHA.output_layer.weights.data, my_fused_MHA.output_layer.weights.data)
assert np.allclose(my_x.data, my_fused_x.data)


my_mask = None#causal_mask(my_x, dim=1)
my_fused_mask = None#causal_mask(my_fused_x, dim=1)

if STEP_BY_STEP:
    # Forward pt.1
    my_q = my_MHA.query_layer(my_x)
    my_k = my_MHA.key_layer(my_x)
    my_v = my_MHA.value_layer(my_x)

    my_qkv = my_fused_MHA.qkv_layer(my_fused_x)

    assert np.allclose(my_q.data, my_qkv[:, :, :D_MODEL].data)
    assert np.allclose(my_k.data, my_qkv[:, :, D_MODEL:2*D_MODEL].data)
    assert np.allclose(my_v.data, my_qkv[:, :, 2*D_MODEL:].data)

    # Forward pt.2
    if MANUAL_SPLIT:
        my_q_presplit = my_q
        my_q_split = my_q.view(BATCH_SIZE, SEQ_LEN, my_MHA.head_num, D_MODEL // my_MHA.head_num).contiguous()
        my_k_split = my_k.view(BATCH_SIZE, SEQ_LEN, my_MHA.head_num, D_MODEL // my_MHA.head_num).contiguous()
        my_v_split = my_v.view(BATCH_SIZE, SEQ_LEN, my_MHA.head_num, D_MODEL // my_MHA.head_num).contiguous()

        _, _, fused_d_model = my_qkv.data.shape
        fused_head_dim = fused_d_model // my_fused_MHA.head_num
        my_qkv_presplit = my_qkv
        my_qkv_split = my_qkv.view(BATCH_SIZE, SEQ_LEN, my_MHA.head_num, fused_head_dim).contiguous()
        #my_qkv_split = Tensor(my_qkv.data.transpose(BATCH_SIZE, SEQ_LEN, my_MHA.head_num, fused_head_dim)).contiguous()

        print(my_q_presplit.sum(), my_q_split.sum())
        print(my_qkv_presplit[:, :, :D_MODEL].sum(), my_qkv_split[:, :, :, :fused_head_dim // 3].sum())
        assert np.allclose(my_qkv_presplit[:, :, :D_MODEL].sum(), my_qkv_split[:, :, :, :fused_head_dim // 3].sum())
        assert np.allclose(my_q_split.data, my_qkv_split[:, :, :, :fused_head_dim // 3].data)

        my_q_split = my_q_split.transpose(1, 2).contiguous()
        my_k_split = my_k_split.transpose(1, 2).contiguous()
        my_v_split = my_v_split.transpose(1, 2).contiguous()

        my_qkv_split = my_qkv_split.transpose(1, 2).contiguous()

        assert np.allclose(my_q_split.data, my_qkv_split[:, :, :, :fused_head_dim // 3].data)

        
    else:
        my_q_split = my_MHA.split(my_q)
        my_k_split = my_MHA.split(my_k)
        my_v_split = my_MHA.split(my_v)

        my_qkv_split = my_fused_MHA.split(my_qkv)
        batch_size, seq_len, fused_d_model = my_qkv.data.shape
        fused_head_dim = fused_d_model // my_fused_MHA.head_num

    print(my_q_split == my_qkv_split[:, :, :, :fused_head_dim//3])
    assert np.allclose(cat([my_q_split, my_k_split, my_v_split], dim=-1).data, my_qkv_split.data)
    assert np.allclose(my_q_split.data, my_qkv_split[:, :, :, :fused_head_dim//3].data)
    assert np.allclose(my_k_split.data, my_qkv_split[:, :, :, fused_head_dim//3:2*fused_head_dim//3].data)  
    assert np.allclose(my_v_split.data, my_qkv_split[:, :, :, 2*fused_head_dim//3:].data)   

    my_q_pf, my_k_pf, my_v_pf = chunk(my_qkv_split, 3, dim=-1)
    #my_q_pf, my_k_pf, my_v_pf = np.array_split(my_qkv_split.data, 3, axis=-1)
    #my_q_pf, my_k_pf, my_v_pf = Tensor(my_q_pf, requires_grad=True), Tensor(my_k_pf, requires_grad=True), Tensor(my_v_pf, requires_grad=True)

    print(my_q_split == my_q_pf)
    assert my_q_split == my_q_pf
    assert my_k_split == my_k_pf
    assert my_v_split == my_v_pf
    assert np.allclose(my_q_split.data, my_q_pf.data)
    assert np.allclose(my_k_split.data, my_k_pf.data)   
    assert np.allclose(my_v_split.data, my_v_pf.data)

    # Self-attention
    '''
    my_attn_scores = my_q_split.matmul(my_k_split.transpose(-2, -1)) / my_MHA.head_dim_sqrt
    my_fused_attn_scores = my_q_pf.matmul(my_k_pf.transpose(-2, -1)) / my_fused_MHA.head_dim_sqrt

    assert my_attn_scores == my_fused_attn_scores

    if my_mask is not None:
        my_attn_scores = my_attn_scores.masked_fill(my_mask, -1e9)
    if my_fused_mask is not None:
        my_fused_attn_scores = my_fused_attn_scores.masked_fill(my_fused_mask, -1e9)

    assert my_attn_scores == my_fused_attn_scores

    my_attn_scores = my_attn_scores @ my_v_split
    my_fused_attn_scores = my_fused_attn_scores @ my_v_pf

    assert my_attn_scores == my_fused_attn_scores
    '''
    my_attn_scores = my_MHA.attention(my_q_split, my_k_split, my_v_split, my_mask)
    my_fused_attn_scores = my_fused_MHA.attention(my_q_pf, my_k_pf, my_v_pf, my_fused_mask)

    assert my_attn_scores == my_fused_attn_scores

    # Merge heads
    my_merged = my_MHA.concat(my_attn_scores)
    my_fused_merged = my_fused_MHA.concat(my_fused_attn_scores)

    assert my_merged == my_fused_merged

    # Projection
    my_res = my_MHA.output_layer(my_merged)
    my_fused_res = my_fused_MHA.output_layer(my_fused_merged)

    my_res.mean().backward()
    my_fused_res.mean().backward()


    fused_to_check = [
        my_q_pf, my_k_pf, my_v_pf, my_fused_attn_scores, my_fused_merged, my_res
    ]
    to_check = [
        my_q_split, my_k_split, my_v_split, my_attn_scores, my_merged, my_res
    ]

    count = 0
    for my_fused_t, my_t in zip(fused_to_check, to_check):

        assert my_fused_t.shape() == my_t.shape(), count
        assert my_fused_t.dtype == my_t.dtype, count
        assert my_t._device.allclose(my_fused_t.data, my_t.data), count
        assert my_fused_t.requires_grad == my_t.requires_grad, count
        assert my_fused_t.is_leaf == my_t.is_leaf, count

        if my_fused_t.is_leaf and my_fused_t.requires_grad:
            if my_fused_t.grad is not None:
                assert my_t.grad is not None, count
            else:
                assert my_t.grad is None
            assert my_t._device.allclose(my_fused_t.grad, my_t.grad), count

        count += 1

else:
    my_res = my_MHA(my_x, my_x, my_x, my_mask)
    my_fused_res = my_fused_MHA(my_fused_x, my_fused_x, my_fused_x, my_fused_mask)

    my_res.mean().backward()
    my_fused_res.mean().backward()

    count = 0
    for my_fused_t, my_t in zip([my_fused_res, my_fused_x], [my_res, my_x]):

        assert my_fused_t.shape() == my_t.shape(), count
        assert my_fused_t.dtype == my_t.dtype, count
        assert my_t._device.allclose(my_fused_t.data, my_t.data), count
        assert my_fused_t.requires_grad == my_t.requires_grad, count
        assert my_fused_t.is_leaf == my_t.is_leaf, count

        if my_fused_t.is_leaf and my_fused_t.requires_grad:
            if my_fused_t.grad is not None:
                assert my_t.grad is not None, count
            else:
                assert my_t.grad is None
            assert my_t._device.allclose(my_fused_t.grad, my_t.grad), count

        count += 1
