# bella libreria per vedere lo usage di memoria: psutil. https://stackoverflow.com/questions/67010386/delete-reference-of-variable-passed-to-a-function-in-python
import numpy as np


class SGD():
    def __init__(self, trainable_params, lr=1e-3, momentum=0):
        self.parameters = trainable_params
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for i in range (0, len(self.parameters)):
            self.parameters[i].weights.grad = None if set_to_none else np.zeros(self.parameters[i].weights.grad.data.shape)
            self.parameters[i].bias.grad = None if set_to_none else np.zeros(self.parameters[i].bias.grad.data.shape)

    def step(self):
        for i in range(0, len(self.parameters)):
            self.parameters[i].weights.data = self.parameters[i].weights.data - self.lr*self.parameters[i].weights.grad.data
            self.parameters[i].bias.data = self.parameters[i].bias.data - self.lr*self.parameters[i].bias.grad.data