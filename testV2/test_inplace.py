from preset import torchic, compare_tensors, create_random_tensors, backward_and_compare
from torchic import nn, draw_graph
from torchic.settings import set_seed

set_seed(42)

torch_a, my_a = create_random_tensors((2,3))
torch_b, my_b = create_random_tensors((2,3))
torch_c, my_c = create_random_tensors((2,3))

# non si lamenta se richiedi il gradiente ma non sei leaf...
# Certo! Perche' tanto i tuoi parametri non devono essere aggiornati! Sei solo
# un tensore di mezzo!
# Non ci sono restrizioni sul tensore di destra!

torch_a.requires_grad=True
torch_b.requires_grad=True
torch_c.requires_grad=True
my_a.requires_grad=True
my_b.requires_grad=True
my_c.requires_grad=True


torch_d = torch_a / torch_b
torch_e = torch_d + torch_c
torch_e *= torch_d


my_d = my_a / my_b
my_e = my_d + my_c
my_e *= my_d


#draw_graph(my_e)

torch_e.mean().backward()
my_e.mean().backward()


print(torch_c.grad, my_c.grad)
print("+++")
print(torch_b.grad, my_b.grad)
print(torch_a.grad, my_a.grad)




'''
backward_and_compare(
    [torch_c, torch_b, torch_a],
    [my_c, my_b, my_a]
)
'''
