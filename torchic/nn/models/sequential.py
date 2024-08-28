from torchic.cuda_tensor import Tensor
from torchic.nn.module import Module, ModuleList

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
            if isinstance(layer, Module):
                layer.to(device)

    def parameters(self, trainable_only: bool=True, with_names: bool=False):
        """
        returns all trainable tensors of the model
        """

        for name, layer in self.__dict__.items():
            if isinstance(layer, Module):
                yield from layer.parameters(trainable_only, with_names, f"{name}.")
    
    

    def number_of_params(self, trainable_only=False) -> int:
        n = 0

        for layer in self:
            if isinstance(layer, Module):
                for param in layer.parameters(trainable_only):
                    n += param.size()
        return n
    
    def state_dict(self) -> dict:
        '''
        Returns the state of the trainable params.
        
        Example::

            >>> print(model.state_dict())
            {'linear0.weights': array([[ 0.09000251,  0.03145568, -0.7834164 ],
                            [-0.4419116 , -0.02768573, -1.3669962 ],
                            [ 0.36691672, -0.23904948, -0.5104246 ],
                            [ 0.1395965 ,  0.5443452 , -0.47324494]], dtype=float32),
            'linear0.bias': array([-0.8548351 , -0.39590812,  0.35631013], dtype=float32),
            'linear2.weights': array([[ 0.4408978 , -0.26182708],
                            [ 0.7175888 , -0.998319  ],
                            [ 0.79341125, -0.5995235 ]], dtype=float32),
            'linear2.bias': array([1.2450343 , 0.93363947], dtype=float32),
            'linear4.weights': array([[0.8089074 ], [0.26144603]], dtype=float32),
            'linear4.bias': array([0.04495028], dtype=float32)}
        '''
        params_dict = {}

        # NOTE: funziona come previsto per layer multistrato?
        # Restituisce davvero dict della forma
        # {'encoder0.embeddings0.weights': [0.2, 0.45]}?
        for name, layer in self.__dict__.items():
            if isinstance(layer, Module):
                for key, value in layer.state_dict(name):
                    params_dict[key] = value
                    
        return params_dict
    

    def __getitem__(self, index):
        return next( elem for i, elem in enumerate(self) if i == index )
    
    
    def __str__(self):
        '''
        Print d'esempio (uguale a PyTorch):

        Example::

            >>> print(model)
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
        '''
        output = "Sequential(\n"
        counter = 0
        for layer in self:
            output = output + f"\t({counter}): {layer},\n"
            counter += 1
        return output + ")"
        '''
        def build_fstring(elem, depth: int):
            # incolonno bene
            tab = "  "
            tabs = tab * depth

            if isinstance(elem, ModuleList):
                output = f"{elem.__class__.__name__}(\n"

                # se ModuleList contiene layer tutti uguali
                if len(set(repr(module) for module in elem)) == 1:
                    output += (
                        f"{tabs}(0-{len(elem) - 1}): {len(elem)} x "
                        + build_fstring(elem[0], depth + 1)
                        )
                else:
                    for i, module in enumerate(elem):
                        output += (
                            f"{tabs}({i}): "
                            + build_fstring(module, depth + 1)
                            )
                return output + f"{tab * (depth - 1)})\n"
            
            elif isinstance(elem, Module):
                # se il Module non ha altri Module come attributi
                if all(not isinstance(param, Module) for _, param in elem):
                    return f"{repr(elem)}\n"
                else:
                    output = f"{elem.__class__.__name__}(\n"

                    for name, param in elem:
                        if isinstance(param, Module):
                            output += (
                                f"{tabs}({name}): "
                                + build_fstring(param, depth + 1)
                                )
                    return output + f"{tab * (depth - 1)})\n"


        output = f"{self.__class__.__name__}(\n"
        for name, layer in self.__dict__.items():
            output += f"  ({name}): " + build_fstring(layer, 2)
        return output + ")"

    
    # chiamato in save_model() nei metadata
    def __repr__(self):
        output = "nn.Sequential(\n"
        for layer in self:
            output = output + f"\tnn.{repr(layer)},\n"
        return output + ")"

    def __call__(self, x) -> Tensor:
        return self.forward(x)