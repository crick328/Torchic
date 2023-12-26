from test2_NNet import NNetwork, LLayer, Sigmoid, ReLU, SELoss, save_model_type1, load_model_type1
from test11_tensor import Tensor
from optim import SGD


#import time
#t = time.time()
#a = Tensor([np.ones((2300,5400))])
#print(f"Tempo impiegato: {((time.time() - t)):.10f} secondi")


model = NNetwork(
    LLayer(2, 500),
    ReLU(),
    LLayer(500, 10),
    Sigmoid()
)
print(model(Tensor([1,4])))
save_model_type1(model, "test")
model = load_model_type1("test")
print("ok")

print(model(Tensor([1,4])))