from optimizers import SGD
from test10_tensor import Tensor

class oggetto_del_cazzo:
    def __init__(self):
        self.lista = [Tensor([1,2,4], requires_grad=True), Tensor([1,2,4,5])]
    
    def dai_i_cazzo_di_grad(self):
        self.lista[0].grad = [-2000,-345,-245345]
        self.lista[1].grad = [92503427,34857,43554,434543]

cazzo = oggetto_del_cazzo()
print(cazzo.lista)

optimizer = SGD(cazzo.lista, lr=1)
optimizer.step()

print(cazzo.lista)
