from preset import torchic, compare_tensors, create_random_tensors, backward_and_compare, test_tensor_function
#from torchic.tools.helpers import chunk
import torch


'''
# ==== CHUNK ====
print("testing chunk...")
NUM_CHUNKS = 2
for dim in range(0, 3):
    torch_t, my_t = create_random_tensors((2, 3, 7))

    torch_res = torch_t.chunk(NUM_CHUNKS, dim=dim)
    my_res = chunk(my_t, NUM_CHUNKS, dim=dim)

    backward_and_compare(
        [*torch_res, torch_t],
        [*my_res, my_t]
    )
'''
    

# ==== CLAMP ====
print("Testing clamp...")
min_value, max_value = -10, 10
torch_t, my_t = create_random_tensors((3, 2, 4))

torch_res = torch_t.clamp(min_value, max_value)
my_res = my_t.clamp(min_value, max_value)

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )


# ==== MASKED_FILL ====
size = 4
torch_t, my_t = create_random_tensors((3, size, size))
torch_mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) == 0
my_mask = torchic.Tensor(my_t._device.triu(my_t._device.ones((1, size, size), dtype=int), k=1))
my_mask.data = my_mask.data == 0


compare_tensors(torch_mask, my_mask)

torch_res = torch_t.masked_fill(torch_mask, -1e09)
my_res = my_t.masked_fill(my_mask, -1e09)

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )





# ===== MATMUL =====
print("Testing mamtul...")
# caso vettore @ matrice (usato nelle NN)
torch_vec, my_vec = create_random_tensors((5,))
torch_mat, my_mat = create_random_tensors((5,3))

torch_res = torch_vec @ torch_mat
my_res = my_vec @ my_mat

backward_and_compare(
    [torch_res, torch_mat, torch_vec],
    [my_res, my_mat, my_vec]
    )

# caso matrice @ vettore
torch_mat, my_mat = create_random_tensors((5,3))
torch_vec, my_vec = create_random_tensors((3,))

torch_res = torch_mat @ torch_vec
my_res = my_mat @ my_vec

backward_and_compare(
    [torch_res, torch_mat, torch_vec],
    [my_res, my_mat, my_vec]
    )

# caso matrice @ matrice
torch_mat1, my_mat1 = create_random_tensors((5,3))
torch_mat2, my_mat2 = create_random_tensors((3,4))

torch_res = torch_mat1 @ torch_mat2
my_res = my_mat1 @ my_mat2

backward_and_compare(
    [torch_res, torch_mat1, torch_mat2],
    [my_res, my_mat1, my_mat2]
    )
print("Done.")


# ===== MEAN ====
print("Testing mean...")
torch_t, my_t = create_random_tensors((2,3,4))

torch_res = torch_t.mean(dim=-1)
my_res = my_t.mean(dim=-1)

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )
print("Done.")



# ==== PERMUTE ====
print("Testing permute...")
torch_t, my_t = create_random_tensors((2,3,4,5))

torch_res = torch_t.permute(0,3,1,2)
my_res = my_t.permute(0,3,1,2)

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )
print("Done.")


# ===== POW =====
print("Testing power...")
exponent = 2
torch_t, my_t = create_random_tensors((2,3,4))

torch_res = torch_t**exponent
my_res = my_t**exponent

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )
print("Done.")



# ==== SQRT ====
print("Testing sqrt...")
test_tensor_function("sqrt", (2,3,4), op_with_self=False)




# ==== STD ====
print("Testing std...")
# test delle dimensioni
for dim in range(0, 3):
    torch_t, my_t = create_random_tensors((2,3,4))

    torch_res = torch_t.std(dim, keepdim=True)
    my_res = my_t.std(dim, keepdim=True)

    backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )
    
# test per keepdim
torch_t, my_t = create_random_tensors((2,3,4))

torch_res = torch_t.std(keepdim=True)
my_res = my_t.std(keepdim=True)

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )
print("Done.")





# ==== VAR ====
print("Testing var...")
# test delle dimensioni
for dim in range(0, 3):
    torch_t, my_t = create_random_tensors((2,3,4))

    torch_res = torch_t.var(dim)
    my_res = my_t.var(dim)

    backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )
    
# test per keepdim
torch_t, my_t = create_random_tensors((2,3,4))

torch_res = torch_t.var(keepdim=True)
my_res = my_t.var(keepdim=True)

backward_and_compare(
    [torch_res, torch_t],
    [my_res, my_t]
    )