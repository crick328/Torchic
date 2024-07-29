from torchic.cuda_tensor import Tensor
from torchic.nn.layers.linears import Linear

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
    
    def __iter__(self):
        for layer in self.__dict__.values():
            yield layer
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = layer(x)
        return x
    
    def to(self, device: str) -> None:
        for layer in self:
            if isinstance(layer, Linear):
                layer.weights.to(device)
                if layer.has_bias:
                    layer.bias.to(device)

    def parameters(self) -> list[Tensor]:
        """
        returns all trainable tensors of the model
        """
        params = []

        for layer in self:
            if isinstance(layer, Linear):
                if layer.weights.requires_grad:
                    params.append(layer.weights)

                if layer.has_bias and layer.bias.requires_grad:
                    params.append(layer.bias)
        return params

    def number_of_params(self, learnable_only=False) -> int:
        number_of_params = 0

        for layer in self:
            if isinstance(layer, Linear):
                number_of_params += layer.weights.data.size
                if layer.has_bias:
                    number_of_params += layer.bias.data.size
        return number_of_params
    

    def state_dict(self) -> dict:
        '''
        Returns the state of the trainable params.
        '''
        params_dict = {}

        for name, layer in self.__dict__.items():
            if isinstance(layer, Linear):
                if layer.weights.requires_grad:
                    params_dict[f"{name}.weights"] = layer.weights.data
                if layer.has_bias and layer.bias.requires_grad:
                    params_dict[f"{name}.bias"] = layer.bias.data
        
        return params_dict
    
    
    def __str__(self):
        '''
        Print d'esempio (uguale a PyTorch):
        Sequential(
                (0): Linear(n_input=784, n_output=500, bias=True),
                (1): ReLU(),
                (2): Linear(n_input=500, n_output=250, bias=True),
                (3): ReLU(),
                (4): Linear(n_input=250, n_output=100, bias=True),
                (5): ReLU(),
                (6): Linear(n_input=100, n_output=10, bias=True),
                (7): Softmax(dim=0),
        )
        '''
        output = "Sequential(\n"
        counter = 0
        for layer in self:
            output = output + f"\t({counter}): {layer},\n"
            counter += 1
        return output + ")"

    
    # chiamato in save_model() nei metadata
    def __repr__(self):
        output = "nn.Sequential(\n"
        for layer in self:
            output = output + f"\tnn.{repr(layer)},\n"
        return output + ")"

    def __call__(self, x) -> Tensor:
        return self.forward(x)
    
    '''
    def structure(self):
        from torchic.nn.activations import ActivationFunction
        output = "Sequential\n"

        for elem in self.__dict__.values():
            if isinstance(elem, Linear):
                output = output + f"{elem.__class__.__name__}:\n\tweights: {elem.weights.shape()}, requires_grad={elem.weights.requires_grad}\n"
                
                if elem.has_bias:
                    output = output + f"\tbias: {elem.bias.shape()}, requires_grad={elem.bias.requires_grad}\n"
            
            elif isinstance(elem, ActivationFunction):
                act_fun_str = f"{repr(elem)}\n".replace(")", "")
                act_fun_str = act_fun_str.replace("(dim", ": dim") if "dim" in act_fun_str else act_fun_str.replace("(", " ")
                output = output + act_fun_str
        return output
    '''