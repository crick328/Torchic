import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.requires_grad = requires_grad
        self._grad = None
        self._device = np  # Mocking device for simplicity

    def shape(self):
        return self.data.shape
    
    def random_grad(self):
        self._grad = np.random.randn(*self.shape())


class Module():
    def parameters(self, trainable_only: bool = True):
        for attr_name, param in self:
            if isinstance(param, Tensor):
                if not trainable_only or param.requires_grad:
                    yield param
            elif isinstance(param, Module):
                yield from param.parameters(trainable_only=trainable_only)

    def state_dict(self, layer_name: str):
        state = {}
        for name, param in self:
            if isinstance(param, Tensor) and param.requires_grad:
                state[f"{layer_name}.{name}"] = param.data
            elif isinstance(param, Module):
                state.update(param.state_dict(f"{layer_name}.{name}"))
        return state

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __iter__(self):
        if isinstance(self.__slots__, str):
            yield self.__slots__, self.__getattribute__(self.__slots__)
        else:
            for attr_name in self.__slots__:
                yield attr_name, self.__getattribute__(attr_name)

    def __repr__(self):
        params = list(self.parameters())
        param_info = ', '.join(f"{param.data.shape}" for param in params)
        return f"{self.__class__.__name__}({param_info})"

class Linear(Module):
    __slots__ = ('weights', 'bias', 'has_bias')

    def __init__(self, n_input: int, n_output: int, bias: bool = True):
        self.weights = Tensor(np.random.randn(n_input, n_output))
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros((n_output,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        y = np.dot(x.data, self.weights.data)
        if self.has_bias:
            y += self.bias.data
        return Tensor(y)

class Sequential(Module):

    def __init__(self, *args):
        for i, element in enumerate(args):
            self.__dict__[f"{element.__class__.__name__.lower()}{i}"] = element
    
    def __iter__(self):
        for layer in self.__dict__.values():
            yield layer

    def __getitem__(self, index):
        return next( elem for i, elem in enumerate(self) if i == index )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = layer.forward(x)
        return x

    def parameters(self, trainable_only: bool = True):
        for layer in self:
            if isinstance(layer, Module):
                yield from layer.parameters(trainable_only=trainable_only)

class SGD:
    __slots__ = ("parameters", "lr")

    def __init__(self, trainable_params: list[Tensor], lr=1e-3):
        self.parameters = list(trainable_params)
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for param in self.parameters:
            param._grad = None if set_to_none else np.zeros(param._grad.shape, param.data.dtype)

    def step(self):
        for param in self.parameters:
            if param._grad is not None:
                param.data -= self.lr * param._grad

# Esempio di utilizzo
linear_layer = Linear(n_input=784, n_output=500, bias=True)
sequential_model = Sequential(linear_layer)

# Esempio di Tensore di Input
x = Tensor(np.random.randn(1, 784))
output = sequential_model.forward(x)

# Esempio di ottimizzatore
optimizer = SGD(sequential_model.parameters(), lr=0.01)

# Aggiungiamo un gradiente NEL MODELLO per il testing
for param in sequential_model.parameters():
    param.random_grad()
# verifica che l'aggiornamento sia stato registrato
print(sequential_model[0].weights._grad)

# Eseguiamo uno step dell'ottimizzatore
optimizer.step()
optimizer.zero_grad()

# Controlliamo se i parametri sono stati aggiornati
for param in sequential_model.parameters():
    print(param._grad)
