from test_NNet import NNetwork, LLayer, SELoss, Sigmoid, load_model_type1, save_model_type1, ReLU
from test10_tensor import Tensor
from optimizers import SGD
import numpy as np
# servono + commenti in generale, più spiegazioni

# softmax e categoriocal cross entropy loss.
# Ricrea i tensor step by step su pytorch per un confronto. Cambia i famosi valori di x, W e b, anche di dimensioni.
# I nuovi tensor falli più grandi, cos' magari inizi a vedere qualcosa sull'uso della memoria e la velocità effettiva.
# Sorpassato il test della rete neurale ReLU+softmax+CCELoss, fai un training su MNIST 16x16

# Rettifico: se hai voglia di vedere un training, fallo pure, ma non speedrunnare l'implementazione di softmax e CCE.
# Tanto anche le reti più basiche riescono ad imparare mnist 16x16, non sono necessarie per ora.
# Implementa solo una funzione per salvare e caricare il modello, altrimenti non puoi testarlo.
# Puramente opzionale: confronta le velocità delle dumb_matmul contro numpy

W1 = Tensor([[0.5794, 0.6730, 0.8875, 0.9021, 0.9662, 0.8482],
        [0.5435, 0.7482, 0.7800, 0.9295, 0.6639, 0.9301],
        [0.2938, 0.0957, 0.4389, 0.5478, 0.2955, 0.3213],
        [0.3429, 0.4262, 0.1190, 0.1769, 0.3686, 0.7412],
        [0.6553, 0.2470, 0.0884, 0.1499, 0.8788, 0.7797],
        [0.3195, 0.6272, 0.6759, 0.3161, 0.8730, 0.8096],
        [0.0216, 0.4054, 0.3292, 0.8136, 0.7822, 0.6545],
        [0.7702, 0.4767, 0.7654, 0.8625, 0.6182, 0.6024],
        [0.4077, 0.1423, 0.0370, 0.8075, 0.4355, 0.0881],
        [0.0561, 0.0848, 0.2390, 0.1369, 0.3916, 0.9968],
        [0.5500, 0.7078, 0.2318, 0.1214, 0.8642, 0.1792],
        [0.0261, 0.6829, 0.4548, 0.6723, 0.8068, 0.5544]], requires_grad=True)
B1 = Tensor([0.8918, 0.3249, 0.6569, 0.0247, 0.9285, 0.0806], requires_grad=True)
W2 = Tensor([[0.5671, 0.5108, 0.6357],
        [0.7820, 0.9338, 0.9623],
        [0.8650, 0.4352, 0.3510],
        [0.4005, 0.7835, 0.2309],
        [0.9725, 0.4179, 0.6448],
        [0.7971, 0.2383, 0.8744]], requires_grad=True)
B2 = Tensor([0.8601, 0.5030, 0.2870], requires_grad=True)

#np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


model = NNetwork((
            LLayer(12, 6),
                   #_w_matrix=W1,
                   #_b_vector=B1),
            Sigmoid(),
            LLayer(6, 3),
                   #_w_matrix=W2,
                   #_b_vector=B2),
            Sigmoid()
))


# al primo forward pass si comporta bene, ma al secondo cancella i gradients

optimizer = SGD(model.parameters())
output = model([0,4,3,4,5,6,4,3,56,6,3,4])
loss = SELoss(output, [7.,-10.,3.])
loss.mean().backward()

print(model.structure[-2].b_vector.grad)

optimizer.step()
optimizer.zero_grad()

print(model.structure[-2].b_vector.grad)


output = model([45,4,9,4,0,6,4,3,6,6,7,4])
loss = SELoss(output, [1.,0.,-6.])
loss.mean().backward()

print(model.structure[-2].b_vector.grad)

optimizer.step()
optimizer.zero_grad()

print(model.structure[-2].b_vector.grad)






#optimizer.step()
#optimizer.zero_grad()

#print(f"vector grad: {model.structure[-2].b_vector} (dovrebbe essere aggiornato)")

#output = model([0,4,3,4,5,6,4,3,56,6,3,4])
#loss = SELoss(output, [7.,-10.,3.])
#loss.mean().backward()



for i in range(0, len(model.structure)):
    if type(model.structure[i]) == LLayer:
        print(f"grad matrice: {model.structure[i].w_matrix.grad}")
        print(f"grad bias: {model.structure[i].b_vector.grad}")




#save_model_type1(model)

