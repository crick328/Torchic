# la funzione da approssimare sara' sqrt.
# mi consente di usare la ReLU e mi fa sperimentare
# cosa fara' la rete quando ricevera' un punto che non
# fa parte del suo dominio

from preset import torchic
from torchic import nn, Tensor, set_default_precision
from torchic.nn.models.save_load import save, load_architecture
import matplotlib.pyplot as plt
import numpy as np
import imageio



EPOCHS = 200
BATCH_SIZE = 20
TRAINING_SAMPLES = 580
TEST_SAMPLES = 80
LR = 0.001
CHOSEN_FUNCTION = lambda x: np.cos(2*x) #np.sqrt(x)
DOMAIN = (-4, 4)

FRAME_SAMPLES_RATE = 200
SAMPLE_POINTS = np.linspace(DOMAIN[0], DOMAIN[1], TEST_SAMPLES)
PATH = r"testV2\sqrt"
frames = []


set_default_precision(64)


# La rete prendera' e restituira' uno scalare float perche'
# voglio complicarle la vita
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)



def visualize_fit(model, x_values: list, subname: str):
    x_values = list(x_values)
    true_y = [CHOSEN_FUNCTION(x) for x in x_values]
    y_preds = [float(model(Tensor([x]))[0].data) for x in x_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, true_y, label='True Function', color='blue', linewidth=2)
    plt.plot(x_values, y_preds, label='Fit', color='orange', linewidth=1)
    
    # Titoli, legenda e labels
    plt.title(f'{subname}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Salva il plot ad ogni epoch
    plt.savefig(f'{PATH}\plot {subname}.png')
    plt.close()

    #plt.show()
    
    # Aggiunge l'immagine al frame
    frames.append(imageio.imread(f'{PATH}\plot {subname}.png'))
    #exit()



training_points = set()
test_points = []

unique_x = np.random.uniform(DOMAIN[0], DOMAIN[1], TRAINING_SAMPLES)
#unique_x = np.linspace(0, 100, TRAINING_SAMPLES)
for x in unique_x:
    y = CHOSEN_FUNCTION(x)
    training_points.add( (Tensor([[x]]), Tensor([[y]])) )

#print(f"Generazione del dataset finita. Punto randomico: {training_points.pop()}")

while len(test_points) != TEST_SAMPLES:
    test_points.append(training_points.pop())

training_points = list(training_points)



loss_fn = nn.MSELoss()
optim = nn.SGD(model.parameters(), lr=LR)


# Training loop
indices = np.linspace(0, len(training_points)-1, num=len(training_points), dtype=np.int32)

for epoch in range(0, EPOCHS):
    counter = 0
    np.random.shuffle(indices)

    for i in indices:
        counter += 1

        point = training_points[i]
        predicted = model(point[0])
        
        loss = loss_fn(predicted, point[1])
        loss.backward()

        if counter % BATCH_SIZE == 0:
            optim.step()
            optim.zero_grad()
            print(f"Epoch: {epoch+1}/{EPOCHS}, loss: {float(loss.data) :.2f}", flush=True)
        
        if counter % FRAME_SAMPLES_RATE == 0 or counter == 1:
            visualize_fit(model, SAMPLE_POINTS, f"Epoch {epoch}, iter {counter}")

        #optim.lr = optim.lr/10



# Validation
correct=0
ATOL = 0.1

for x, y in test_points:
    y_predicted = model(x)

    if np.abs(y.data - y_predicted.data) < ATOL:
        correct +=1
print(f"Validation score: {correct/TEST_SAMPLES * 100 :0.1f}% correct points")


print(model(Tensor([[4]])))
print(model(Tensor([[2]])))
print(model(Tensor([[0]])))
save(model, filename=f"{PATH}\cos_net[{DOMAIN[0]}-{DOMAIN[1]}]")


# Salva la GIF
output_file = 'training_progress.gif'
imageio.mimsave(output_file, frames, duration=0.5)  # `duration` regola la velocità dei frame
