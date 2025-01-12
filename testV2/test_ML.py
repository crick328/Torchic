from preset import torchic
from torchic import nn

module_list = nn.ModuleList(
    [nn.Linear(3,2, bias=False),
    nn.Linear(3,2, bias=False)]
)

print(module_list.parameters())
print(module_list.number_of_params())
print(module_list._set_training_mode(False))