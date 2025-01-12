from preset import torchic, create_random_tensors, backward_and_compare
import torch

torch.set_default_dtype(torch.float64)
torchic.set_default_precision(64)
#torchic.set_default_device("dynamic")


# PROBLEMA: se faccio normalmente la backward ma ho
# un keepdim, il gradiente della funzione con keepdim e' piu'
# piccolo del dovuto di due o tre ordini di grandezza
# Non trovo problemi nella derivata di std... Che il
# problema sia il prodotto? Non sembra... Deve esserci
# qualche interazione che rompe tutto
# E PARE ESSERE IL + my_t. Le operazioni svolte (tutte somme
# o moltiplicazioni) NON cambiano il problema
# AGGIORNAMENTO: il problema nasce quando si usa un risultato
# ottenuto con una funzione di riduzione avente keepdim=True
# e lo si usa insieme al tensore originale.
# ES:
# a = t.sum(keepdim=True)
# res = a + t
# => t avra' un gradiente sbagliato



DIM = 1
KEEPDIM = True


torch_t, my_t = create_random_tensors((2,3,4))
#torch_tt, my_tt = create_random_tensors((2,3,1))
#my_t = my_t.to("cuda")

# sum e std ok per ora
torch_a = torch_t.mean(dim=DIM, keepdim=KEEPDIM)
torch_b = torch_t.std(dim=DIM, keepdim=KEEPDIM)
#torch_b = torch_t.var(keepdim=KEEPDIM)
#torch_b = torch_t.sum(dim=DIM, keepdim=KEEPDIM)
#torch_res = torch_b + torch_t + torch_tt
torch_res = torch_a+torch_b + torch_t

#print(torch_b.shape)

my_a = my_t.mean(dim=DIM, keepdim=KEEPDIM)
my_b = my_t.std(dim=DIM, keepdim=KEEPDIM)
#my_b = my_t.var(keepdim=KEEPDIM)
#my_b = my_t.sum(dim=DIM, keepdim=KEEPDIM)
#my_res = my_tt + my_b + my_t
my_res = my_a+my_b + my_t


#torch_b.mean().backward()
#my_b.mean().backward()

#print(torch_t.grad)
#print(my_t.grad)


backward_and_compare(
    #[torch_res],#, torch_b, torch_a, torch_t],
    #[my_res],#, my_b, my_a, my_t],
    [torch_res, torch_b, torch_a, torch_t],
    [my_res, my_b, my_a, my_t],
    show_graph=False
)


#print(torch_t.grad)
#print(my_t.grad)