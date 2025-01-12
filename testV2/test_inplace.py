from preset import torchic, compare_tensors, create_random_tensors, backward_and_compare
from torchic import nn, draw_graph

torch_a, my_a = create_random_tensors((2,3))
torch_b, my_b = create_random_tensors((2,3))

torch_b.requires_grad=False
my_b.requires_grad=False

torch_c = torch_a * torch_b
my_c = my_a * my_b

torch_c += torch_b
my_c += my_b

#draw_graph(my_c)

backward_and_compare(
    [torch_c, torch_b, torch_a],
    [my_c, my_b, my_a]
)