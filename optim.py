# bella libreria per vedere lo usage di memoria: psutil. https://stackoverflow.com/questions/67010386/delete-reference-of-variable-passed-to-a-function-in-python
import numpy as np
# tutti questi self.parameters[i]


class SGD():
    def __init__(self, trainable_params, lr=1e-3, momentum=0):
        self.parameters = trainable_params
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for i in range (0, len(self.parameters)):
            self.parameters[i].grad = None if set_to_none else np.zeros(self.parameters[i].grad.data.shape)
    
    def step(self):
        for i in range(0, len(self.parameters)):
            self.parameters[i].data -= self.lr*self.parameters[i].grad.data