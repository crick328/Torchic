class SGD():
    def __init__(self, trainable_params, lr=1e-3, momentum=0):
        # NOTA: questo funziona perché i miei oggetti sono incapsulati
        # all'interno di un oggetto mutabile (lista)
        self.parameters = trainable_params
        self.lr = lr
        

    def zero_grad_(self, set_to_none=True):
        for i in range (0, len(self.parameters)):
            self.parameters[i].grad = None if set_to_none else self.parameters[i].grad.data.shape.zeros() #np.zeros(self.parameters[i].grad.data.shape)

    def zero_grad(self, set_to_none=True):
        for param in self.parameters:
            param.grad = None if set_to_none else param.grad.data.shape.zeros() #np.zeros(self.parameters[i].grad.data.shape)

    # qua ci perdo 97s... Conta che
    # outer chiede 52.5s, mentre
    # matmul 28.7s...
    def step_(self):
        for i in range(0, len(self.parameters)):
            self.parameters[i].data -= self.lr*self.parameters[i].grad.data

    def step(self):
        for param in self.parameters:
            param.data -= self.lr*param.grad.data

    # FUNZIONA!!
    def _test_step(self):
        tmp = []
        for param in self.parameters:
            param.data -= self.lr*param.grad.data
            tmp.append(param)
        self.parameters = tmp
    # NOTA: dentro questa classe credo che sei libero di usare i return!