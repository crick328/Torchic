from torchic.cuda_tensor import Tensor
from torchic.nn.module import Module

# si puo' creare una classe Module nel momento in cui vedi che ci sono delle
# cose che sono uguali per ogni classe. Ad esempio, Linear e Sequential hanno
# entrambi bisogno di un metodo che cambi device e che permetta di salvare su
# file. Quando ne avrai bisongo, implementalo.
# Esempio:

# questa classe non ha senso che erediti da Module perche' non ha
# bisogno della sua struttura. Questa deve essere una classe dinamica
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
    
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self:
            x = layer(x)
        return x
    
    def to(self, device: str) -> None:
        for layer in self:
            if isinstance(layer, Module):
                layer.to(device)

    def number_of_params(self, trainable_only=False) -> int:
        n = 0
        for layer in self:
            if isinstance(layer, Module):
                n += layer.number_of_params()
        return n

    def parameters(self, trainable_only: bool=True, with_names: bool=False, tensor_only: bool=True):
        """
        returns all trainable tensors of the model
        """

        for name, layer in self.__dict__.items():
            if isinstance(layer, Module):
                yield from layer.parameters(trainable_only, with_names, tensor_only, name)
    
    # chiamato in save() per salvare i parametri
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
        model_state = {}

        for name, layer in self.__dict__.items():
            if isinstance(layer, Module):
                #print(list(layer.state_dict(name))[0])
                #for key, value in layer.state_dict(name):
                    #model_state[key] = value
                # unisco i dizionari
                model_state |= layer.state_dict(name)
        return model_state
    
    def train(self):
        for layer in self:
            if isinstance(layer, Module):
                layer.train()

    def eval(self):
        for layer in self:
            if isinstance(layer, Module):
                layer.eval()
    

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

        output = f"{self.__class__.__name__}(\n"
        for name, layer in self.__dict__.items():
            #output += f"  ({name}): " + build_fstring(layer, 2)
            output += f"  ({name}): " + Module._build_fstring(layer, 2)
        return output + ")"

    # chiamato in save() nei metadata
    def __repr__(self):
        output = f"nn.{self.__class__.__name__}(\n"
        for layer in self:
            output = output + f"\tnn.{repr(layer)},\n"
        return output + ")"