from preset import torchic, compare_tensors, create_random_tensors, backward_and_compare, get_class_instances
from torch import zeros, nn


# testerei embeddings e PosEncoding (init e forward). Idealmente anche Attention, ma dopo.
# Prima di tutto, confronta il codice con quello di Jamil


#torch_drop, my_drop = get_class_instances("Dropout", {"p": 0.5})
'''
Il dropout e' randomico, ops

torch_drop = nn.Dropout(0.5)
my_drop = torchic.nn.Dropout(0.5)

torch_t, my_t = create_random_tensors((2,3,4))

torch_res = torch_drop(torch_t)
my_res = my_drop(my_t)

backward_and_compare(
        [torch_res, torch_t],
        [my_res, my_t]
        )
backward_and_compare(
        torch_drop.parameters(),
        my_drop.parameters()
        )
'''