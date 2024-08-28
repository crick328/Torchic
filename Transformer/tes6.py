import numpy as np
import time

class Tensor(np.ndarray):
    @staticmethod
    def random(shape):
        return np.random.rand(*shape)
    
class Dropout:
    def __init__(self, probability: float):
        self.p = probability
        self.training = True  # Per controllare se siamo in modalità training o eval

    def forward_variant1(self, x):
        if self.training:
            mask = Tensor.random(x.shape) > self.p
            scale = 1.0 / (1.0 - self.p)
            return x * mask * scale
        else:
            return x

    def forward_variant2(self, x):
        if self.training:
            mask = Tensor.random(x.shape) > self.p
            scale = 1.0 / (1.0 - self.p)
            x *= mask
            x *= scale
            return x
        else:
            return x

# Esempio di utilizzo
dropout = Dropout(probability=0.5)
x = np.random.rand(1000, 1000)

# Benchmarking
start_time = time.time()
for _ in range(1000):
    dropout.forward_variant1(x.copy())
print("Variante 1 tempo medio:", (time.time() - start_time) / 1000)

start_time = time.time()
for _ in range(1000):
    dropout.forward_variant2(x.copy())
print("Variante 2 tempo medio:", (time.time() - start_time) / 1000)
