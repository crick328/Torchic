# bella libreria per vedere lo usage di memoria: psutil. https://stackoverflow.com/questions/67010386/delete-reference-of-variable-passed-to-a-function-in-python
import numpy as np
from test_NNet import LLayer
from test10_tensor import Tensor

class SGD():
    def __init__(self, trainable_params, lr=1e-3, momentum=0):
        self.parameters = trainable_params
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for i in range (0, len(self.parameters)):
            self.parameters[i].w_matrix.grad = None if set_to_none else np.zeros(self.parameters[i].w_matrix.grad.data.shape)
            self.parameters[i].b_vector.grad = None if set_to_none else np.zeros(self.parameters[i].b_vector.grad.data.shape)

    def step(self):
        '''
        for i in range(0, len(self.parameters)):
            if type(self.parameters[i]) == LLayer:
                print(f"grad matrice: {self.parameters[i].w_matrix.grad}")
                print(f"grad bias: {self.parameters[i].b_vector.grad}")
        '''
        for i in range(0, len(self.parameters)):
            self.parameters[i].w_matrix.data = self.parameters[i].w_matrix.data - self.lr*self.parameters[i].w_matrix.grad.data
            self.parameters[i].b_vector.data = self.parameters[i].b_vector.data - self.lr*self.parameters[i].b_vector.grad.data