from preset import torchic, compare_tensors, create_random_tensors, backward_and_compare
from torchic import nn
from torch.nn import LogSoftmax, Sigmoid, Softmax


# ==== LogSoftmax ====
print("Testing LogSoftmax...")

for dim in range(0, 3):
    print(f"Testing on dim {dim}")

    torch_soft = LogSoftmax(dim)
    my_soft = nn.LogSoftmax(dim)

    torch_t, my_t = create_random_tensors((4,2,5))

    torch_res = torch_soft(torch_t)
    my_res = my_soft(my_t)

    backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )



# ===== SIGMOID =====
print(f"Testing Sigmoid...")

torch_sigmoid = Sigmoid()
my_sigmoid = nn.Sigmoid()

torch_t, my_t = create_random_tensors((2,4,5))

torch_res = torch_sigmoid(torch_t)
my_res = my_sigmoid(my_t)

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )



# ==== Softmax ====
print("Testing Softmax...")
for dim in range(0, 3):
    print(f"Testing on dim {dim}")

    torch_soft = Softmax(dim)
    my_soft = nn.Softmax(dim)

    torch_t, my_t = create_random_tensors((4,2,5))

    torch_res = torch_soft(torch_t)
    my_res = my_soft(my_t)

    backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )


#print(torch_t.grad)
#print(my_t.grad)