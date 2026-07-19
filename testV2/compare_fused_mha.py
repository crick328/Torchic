# preset e' un modulo di test che importa torhic ed implementa funzioni
# aggiuntive di comparison e utility
from preset import torchic, create_random_tensors, backward_and_compare, compare_tensors
from torchic import nn
import torch


torch.manual_seed(42)

#torch.nn.modules.TransformerEncoderLayer

D_MODEL = 4
HEAD_NUM = 2
DROP=0
BATCH_SIZE = 2
SEQ_LEN = 4


#tTensor, myTensor = create_random_tensors((BATCH_SIZE, SEQ_LEN, D_MODEL))
tTensorQ = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL), dtype=torch.float64, requires_grad=True)
myTensorQ = torchic.Tensor(tTensorQ.detach().numpy(), requires_grad=True)

tTensorKV = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL), dtype=torch.float64, requires_grad=True)
myTensorKV = torchic.Tensor(tTensorKV.detach().numpy(), requires_grad=True)




myMHA = torchic.nn.models.transformer.MultiHeadAttention(D_MODEL, HEAD_NUM, DROP, allow_fusion=True, bias=False)
# PyTorch fonde automaticamente se le dim K=Q=V (pare lo faccia anche per le non self attention...)
# PORCODIO BATCH_FIRST
tMHA = torch.nn.modules.activation.MultiheadAttention(D_MODEL, HEAD_NUM, DROP, bias=False, batch_first=True)

myMHA.qkv_layer.weights = torchic.Tensor(tMHA.in_proj_weight.detach().numpy().T, requires_grad=True) 
myMHA.output_layer.weights = torchic.Tensor(tMHA.out_proj.weight.detach().numpy().T, requires_grad=True) 
#myMHA.qkv_layer.weights.data = tMHA.in_proj_weight.detach().numpy().T.copy()
#myMHA.output_layer.weights.data = tMHA.out_proj.weight.detach().numpy().T.copy()


print("\n testing QKV Fused...\n")
tOut = tMHA(tTensorQ, tTensorQ, tTensorQ)
myOut = myMHA(myTensorQ, myTensorQ, myTensorQ)

# prendo l'out zero perche' PyTorch restituisce l'out e pesi dell'attention
compare_tensors(tOut[0], myOut)
tOut[0].mean().backward()
myOut.mean().backward()
assert torch.allclose(torch.tensor(myMHA.qkv_layer.weights.grad), tMHA.in_proj_weight.grad.T)




print("\n testing KV Fused:\n")
tOut = tMHA(tTensorQ, tTensorKV, tTensorKV)
myOut = myMHA(myTensorQ, myTensorKV, myTensorKV)

compare_tensors(tOut[0], myOut)
tOut[0].mean().backward()
myOut.mean().backward()
assert torch.allclose(torch.tensor(myMHA.qkv_layer.weights.grad), tMHA.in_proj_weight.grad.T)


#print(tOut[0], "\n\n", myOut)