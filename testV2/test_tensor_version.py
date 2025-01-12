import numpy as np


class DummyTensor:
    def __init__(self, data):
        self._data = data
        self._version = 0
        self.grad = None
        self.grad_fn = None
        self.requires_grad = False

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        self._version += 1

    def inplace_add(self, other: 'DummyTensor') -> 'DummyTensor':
        self.data += other.data

        if self.requires_grad or other.requires_grad:
            pass

    