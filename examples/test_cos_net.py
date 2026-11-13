import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic import nn, Tensor, set_default_precision
from torchic.nn.models.save_load import save, load_architecture
import matplotlib.pyplot as plt
from os.path import exists
from os import mkdir
import numpy as np
import imageio


# ====== Settings ======

EPOCHS = 45
GRAD_ACCUM = 20
TOTAL_SAMPLES = 880
TEST_SAMPLES = 80 # presi dal totale
LR = 0.001
# metto il +1 per poter sfruttare la ReLU
CHOSEN_FUNCTION = lambda x: np.cos(2*x) + 1
DOMAIN = (-3, 3)

FRAME_SAMPLES_RATE = 200
SAMPLE_POINTS = np.linspace(DOMAIN[0], DOMAIN[1], TEST_SAMPLES)
PATH = r"examples/gif_frames"

#set_default_precision(32)

if not exists(PATH):
    mkdir(PATH)


# La rete prendera' e restituira' uno scalare float perche'
# voglio complicarle la vita
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    #nn.ReLU(),
    #nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(32, 1)
)


# ====== Helper function ======

frames = []
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

    frames.append(imageio.imread(f'{PATH}\plot {subname}.png'))


# ==== Dataset Generation ====

training_points = []
test_points = []
unique_x = np.random.uniform(DOMAIN[0], DOMAIN[1], TOTAL_SAMPLES)

for x in unique_x:
    y = CHOSEN_FUNCTION(x)
    training_points.append( (Tensor([[x]]), Tensor([[y]])) )

while len(test_points) != TEST_SAMPLES:
    test_points.append(training_points.pop())


# ====== Training loop ======

loss_fn = nn.MSELoss()
optim = nn.SGD(model.parameters(), lr=LR)

indices = np.linspace(0, len(training_points)-1, num=len(training_points), dtype=np.int32)

for epoch in range(0, EPOCHS):
    counter = 0
    epoch_loss = 0
    np.random.shuffle(indices)

    for i in indices:
        counter += 1

        point = training_points[i]
        predicted = model(point[0])
        
        loss = loss_fn(predicted, point[1])
        loss.backward()
        epoch_loss += loss.data

        if counter % GRAD_ACCUM == 0:
            optim.step()
            optim.zero_grad()
        
        if counter % FRAME_SAMPLES_RATE == 0 or counter == 1:
            visualize_fit(model, SAMPLE_POINTS, f"Epoch {epoch}, iter {counter}")

        #optim.lr = optim.lr/10
    print(f"Epoch: {epoch+1}/{EPOCHS}, epoch mean loss: {float(loss.data.mean()) :.2f}", flush=True)


# ====== Validation ======
correct=0
ATOL = 0.1

for x, y in test_points:
    y_predicted = model(x)

    if np.abs(y.data - y_predicted.data) < ATOL:
        correct +=1
print(f"Validation score: {correct/TEST_SAMPLES * 100 :0.1f}% correct points")


# Salva la GIF
print("Preparing training animation gif...")
output_file = 'examples/training_progress.gif'
imageio.mimsave(output_file, frames, duration=0.5)  # `duration` regola la velocità dei frame
print("Done! You can see it in examples/training_progress.gif!")

#save(model, filename=f"{PATH}\cos_net[{DOMAIN[0]}-{DOMAIN[1]}]")