from torchic.cuda_tensor import Tensor
from torchic.nn.layers import Linear

# si puo' creare una classe Module nel momento in cui vedi che ci sono delle
# cose che sono uguali per ogni classe. Ad esempio, Linear e Sequential hanno
# entrambi bisogno di un metodo che cambi device e che permetta di salvare su
# file. Quando ne avrai bisongo, implementalo.
# Esempio:


class Sequential:
    '''
    Questa classe si può usare in due modi:
    - chiami la classe e gli passi in input una lista/tupla contenente i layer e
      e le activation function in ordine e io gestisco in automatico la scrittura
      della forward;
    - creare una nuova classe che eredita da questa, creare variabili per layer
      e activation function, ma così devi scriverti la forward tu.
    Questo funzionamento mimica Pytorch.
    '''
    def __init__(self, *args):
        for i, element in enumerate(args):
            self.__dict__[f"{element.__class__.__name__.lower()}{i}"] = element

    def forward(self, x: Tensor) -> Tensor:
        for key in self.__dict__:
            x = self.__dict__[key](x)
        return x 
    
    def to(self, device: str) -> None:
        for elem in self.__dict__.values():
            if isinstance(elem, Linear):
                elem.weights.to(device)
                if elem.has_bias:
                    elem.bias.to(device)

    def parameters(self) -> list[Tensor]:
        """
        returns all trainable tensors of the model
        """
        params = []

        for elem in self.__dict__.values():
            if isinstance(elem, Linear):
                if elem.weights.requires_grad:
                    params.append(elem.weights)

                if elem.has_bias and elem.bias.requires_grad:
                    params.append(elem.bias)
        return params

    def number_of_params(self, learnable_only=False) -> int:
        number_of_params = 0

        for elem in self.__dict__.values():
            if isinstance(elem, Linear):
                number_of_params += elem.weights.data.size
                if elem.has_bias:
                    number_of_params += elem.bias.data.size
        return number_of_params
    
    def state_dict(self) -> dict:
        Sequential_class = self.__class__.__name__
        output = {Sequential_class: []}

        for elem in self.__dict__.values():
            output[Sequential_class].append(elem)
        return
    
    def __repr__(self):
        output = "Sequential("
        for elem in self.__dict__.values():
            output = output + f"{repr(elem)}, "
        return output[0:-2] + ")"

    def __call__(self, x) -> Tensor:
        return self.forward(x)