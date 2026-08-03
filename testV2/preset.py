import sys
import torch
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 3.2")
import torchic
from torchic.tools.code_analysis.gradient_checking import check_gradient
from torchic.tools.visualization.graphs import draw_graph

torch.set_default_dtype(torch.float64)
torchic.set_default_precision(64)


def compare_tensors(torch_t: torch.Tensor, my_t: torchic.Tensor, _curr_idx: int=None):
    '''
    Compares shape, type, values, internal status (.requires_grad, .is_leaf) and grad
    '''
    assert torch_t.shape == my_t.shape(), f"Mismatch in shape of index {_curr_idx}.\n Torch: {torch_t.shape},\nTorchic: {my_t.shape()}"
    assert torch_t.detach().numpy().dtype == my_t.dtype, f"Mismatch in dtype of index {_curr_idx}.\n Torch: {torch_t.numpy().dtype},\nTorchic: {my_t.dtype}"
    assert torch.allclose(torch.tensor(my_t.data), torch_t), f"Mismatch in data of index {_curr_idx}.\n Torch: {torch_t},\nTorchic: {my_t}"
    assert torch_t.requires_grad == my_t.requires_grad, f"Mismatch in requires_grad of index {_curr_idx}.\n Torch: {torch_t.requires_grad},\nTorchic: {my_t.requires_grad}"
    assert torch_t.is_leaf == my_t.is_leaf, f"Mismatch in is_leaf of index {_curr_idx}.\n Torch: {torch_t.is_leaf},\nTorchic: {my_t.is_leaf}"

    if torch_t.is_leaf and torch_t.requires_grad:
        if torch_t.grad is not None:
            assert my_t.grad is not None, f"Gradient is None in index {_curr_idx}.\n Torch: {torch_t.grad},\nTorchic: {my_t.grad}"
        else:
            assert my_t.grad is None, f"Gradient is not None in index {_curr_idx}.\n Torch: {torch_t.grad},\nTorchic: {my_t.grad}"
        assert torch.allclose(torch.tensor(my_t.grad), torch_t.grad), f"Mismatch in grad of index {_curr_idx}.\n Torch: {torch_t.grad},\nTorchic: {my_t.grad}"
        
        #assert my_t.data.c_contiguous
        #assert my_t.grad.data.c_contiguous


def compare_modules(torch_module, my_module, transpose=True):
    torch_params = torch_module.named_parameters()
    my_params = my_module.parameters(with_names=True)
    # questo brucia i generatori (d'altronde sono usa e getta)
    #assert len(list(torch_params)) == len(list(my_params))

    # ho dovuto trasporre tutti i params, quindi lo rifaccio
    for j_param, my_param in zip(torch_params, my_params):
        print(f"Testing parameters {j_param[0]}, {my_param[0]}")

        # mi sa che .T traspone anche il grad
        torch_t = j_param[1].T if transpose and j_param[1].dim() == 2 else j_param[1]
        my_t = my_param[1]

        assert torch.allclose(torch.tensor(my_t.data), torch_t), f"Mismatch in data.\n Torch: {torch_t},\nTorchic: {my_t}"
        
        if torch_t.grad is not None:
            assert torch.allclose(torch.tensor(my_t.grad), torch_t.grad), f"Mismatch in grad.\n Torch: {torch_t.grad},\nTorchic: {my_t.grad}"


def create_random_tensors(shape: tuple) -> tuple[torch.Tensor, torchic.Tensor]:
    my_t = torchic.rand(shape, requires_grad=True)
    torch_t = torch.tensor(my_t.data.copy(), requires_grad=True)

    return torch_t, my_t


def equalize_layer(torch_layer, my_layer):
    my_params = {}

    for param in my_layer.parameters():
        if param.shape() in my_params:
            raise RuntimeError(
                "The chosen layers has one or more parameters that "
                "shares the same shape, so it's not safe to "
                "automatically init them to the same values."
            )
        data = param.detach().clone().data
        if data.ndim > 1:
            data = data.T
        my_params[data.shape] = torch.from_numpy(data)
    
    with torch.no_grad():
        for param in torch_layer.parameters():
            print(param.shape, my_params)
            if any(param.shape == shape for shape in my_params):
                param.data.copy_(my_params[param.shape])
                my_params.pop(param.shape)

    return torch_layer, my_layer


def get_class_instances(cls_name: str, kwargs):
    try:
        torch_class = getattr(torch.nn, cls_name)
        my_class = getattr(torchic.nn, cls_name)

        if kwargs:
            if "layer_shape" in kwargs:
                in_f, out_f = kwargs["layer_shape"]
                kwargs.pop("layer_shape")

                torch_layer = torch_class(in_f, out_f, **kwargs)
                my_layer = my_class(in_f, out_f, **kwargs)

                # devono condividere gli stessi
                # valori di partenza

                my_params = {}

                for param in my_layer.parameters():
                    if param.shape() in my_params:
                        raise RuntimeError(
                            "The chosen layers has one or more parameters that "
                            "shares the same shape, so it's not safe to "
                            "automatically init them to the same values."
                        )
                    data = param.detach().clone().data
                    if data.ndim > 1:
                        data = data.T
                    my_params[data.shape] = torch.from_numpy(data)
                
                with torch.no_grad():
                    for param in torch_layer.parameters():
                        print(param.shape, my_params)
                        if any(param.shape == shape for shape in my_params):
                            param.data.copy_(my_params[param.shape])
                            my_params.pop(param.shape)

                return torch_layer, my_layer
            else:
                return torch_class(**kwargs), my_class(**kwargs)
        else:
            return torch_class(), my_class()
    except Exception as e:
        raise e


def backward_and_compare(torch_tensors: list, my_tensors: list, show_graph: bool=False):
    '''
    Takes two LISTS of tensors, computes the gradients and compares them
    '''
    torch_root = torch_tensors[0]
    my_root = my_tensors[0]

    #compare_tensors(torch_root, my_root)
    if show_graph:
        draw_graph(my_root)

    torch_root.mean().backward()
    my_root.mean().backward()

    count = 0
    for torch_t, my_t in zip(torch_tensors, my_tensors):
        compare_tensors(torch_t, my_t, count)
        count += 1




def test_tensor_function(fn_name: str, shape: tuple, op_with_self=False):
    torch_t, my_t = create_random_tensors(shape)

    try:
        torch_func = getattr(torch_t, fn_name)
        my_func = getattr(my_t, fn_name)
    except Exception as e:
        print(e)
        exit()
    
    if not op_with_self:
        torch_res = torch_func()
        my_res = my_func()
    else:
        # in caso di operazioni come add, faccio self + self
        torch_res = torch_func(torch_t)
        my_res = my_func(my_t)

    torch_res.mean().backward()
    my_res.mean().backward()

    compare_tensors(torch_res, my_res)
    compare_tensors(torch_t, my_t)


def test_class(cls_name: str, shape: tuple, **kwargs):
    '''
    Test anything that is a class, such as loss functions, layers
    and activation functions. Models are not supported.
    If the class is a loss function, a target tensor is required
    in order to calculate the loss without bugs.
    '''
    torch_instance, my_instance = get_class_instances(cls_name, kwargs)
    torch_t, my_t = create_random_tensors(shape)

    # modo casereccio per capire se e' una loss
    if my_instance.__class__.__name__[-4:] == "Loss":
        assert "target" in kwargs, "A 'target' tensor is required in order to calculate losses without bugs."
        
        my_trg = kwargs["target"]
        torch_trg = torch.tensor(my_trg.data, dtype=my_trg.dtype)

        torch_res = torch_instance(torch_t, torch_trg)
        my_res = my_instance(my_t, my_trg)

    else:

        #compare_tensors(torch_t, my_t)
        #compare_tensors(torch_instance.weight, my_instance.weights)

        #torch_res = torch_t.matmul(torch_instance.weight.T)
        #my_res = my_t.matmul(my_instance.weights)
        #print(torch_res)
        #print("----")
        #print(my_res)

        #torch_res = torch_res.add(torch_instance.bias)
        #my_res = my_res.__add__(my_instance.bias)

        print("------")
        print(torch_instance.weight.shape)
        print(my_instance.weights.shape())
        print("------")


        torch_res = torch_instance(torch_t)
        my_res = my_instance(my_t)
        #print(torch_res)
        #print("----")
        #print(my_res)

    backward_and_compare([torch_res, torch_t], [my_res, my_t])
    #torch_res.mean().backward()
    #my_res.mean().backward()

    # questi vanno bene
    #print(torch_instance.weight.grad)
    #print(my_instance.weights.grad)
    #print(torch_t.grad)
    #print(my_t.grad)

    # NOTE: devi chiamare zero_grad quando lavori con questo
    #if isinstance(my_instance, torchic.nn.Module):
        # posso usare il gradient check
        #check_gradient(my_instance, (my_t,))



    

# TENSOR_FN_TO_TEST = []
#test_tensor_function("__matmul__", (2,2), op_with_self=True)
#test_class("Linear", (3,), layer_shape = (3,2))