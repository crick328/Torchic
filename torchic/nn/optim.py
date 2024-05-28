class SGD():
    __slots__ = ("parameters", "lr")

    def __init__(self, trainable_params, lr=1e-3, momentum=0):
        # NOTA: questo funziona perché i miei oggetti sono incapsulati
        # all'interno di un oggetto mutabile (lista)
        self.parameters = trainable_params
        self.lr = lr
        
    def zero_grad(self, set_to_none=True):
        for param in self.parameters:
            param.grad = None if set_to_none else param._device.zeros(param.grad.shape, param.data.dtype)

    # qua ci perdo 97s... Conta che
    # outer chiede 52.5s, mentre
    # matmul 28.7s...

    def step(self): 
        for param in self.parameters:
            param.data -= self.lr*param.grad
            #print(param.grad)
    # NOTA: dentro questa classe credo che sei libero di usare i return!