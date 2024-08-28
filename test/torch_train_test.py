import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.3")
from torchic.datasets.my_mnist import AutoMnist
import time
import random
from torch import nn, optim, tensor


# https://ml4a.github.io/ml4a/looking_inside_neural_nets/
DEVICE = "cuda"
PROFILER = True
EPOCHS = 1
BATCH_SIZE = 300


# 6.12 minuti con Adam w profiler (32306507 function calls (31346505 primitive calls)) (non traina)

# 3.19 minuti con Adam w cuda e profiler


train_images = AutoMnist.train_images(preprocess=True)
train_labels = AutoMnist.train_labels(one_hot_encode=True)

test_images = AutoMnist.test_images(preprocess=True)
test_labels = AutoMnist.test_labels(one_hot_encode=True)

total_train_samples = len(train_images)
total_test_samples = len(test_images)
height, width = 28, 28


# conversione dei tensori
for i in range(0, total_train_samples):
    train_images[i] = tensor(train_images[i].data, device=DEVICE)
    train_labels[i] = tensor(train_labels[i].data, device=DEVICE)
for i in range(0, total_test_samples):
    test_images[i] = tensor(test_images[i].data, device=DEVICE)
    test_labels[i] = tensor(test_labels[i].data, device=DEVICE)


model = nn.Sequential(
    nn.Linear(height*width, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Softmax(0)
)
print(model.parameters())
print(nn.Linear(height*width, 500))

model.to(DEVICE)
loss_fn = nn.MSELoss()
#optimizer = nn.optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)
N_BATCH = int(total_train_samples/BATCH_SIZE)


if PROFILER:
    import cProfile, pstats
    import subprocess

    profiler = cProfile.Profile()
    profiler.enable()
start_time = time.time()


for epoch in range(1, EPOCHS+1):
    epoch_loss = 0
    current_iter = 0
    correct_predictions = 0

    while current_iter < total_train_samples:
        for i in range(current_iter, current_iter + BATCH_SIZE):
            predicted = model(train_images[i])

            loss = loss_fn(predicted, train_labels[i])
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            #print("DBN")

            epoch_loss += loss.data

            if predicted.data.argmax() == train_labels[i].data.argmax(): #train_labels[i].tolist().index(1):
                correct_predictions += 1

        #optimizer.step()
        #optimizer.zero_grad()

        current_iter += BATCH_SIZE
        print(f"Epoch {epoch}/{EPOCHS}, batch {(current_iter/BATCH_SIZE) :.0f}/{N_BATCH}, loss: {(epoch_loss/current_iter*100):.1f}%, accuracy: {(correct_predictions/current_iter*100) :.1f}%")

    # rimescolo il dataset con le labels
    temp = list(zip(train_images, train_labels))
    random.shuffle(temp)
    train_images, train_labels = zip(*temp)


print(f"Fine training! Tempo impiegato: {((time.time() - start_time)/60):.2f} minuti")


if PROFILER:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
    # Remove dir names
    stats.strip_dirs()
    stats.print_stats()
    profiler.dump_stats("profiler_data2.prof")

    subprocess.run(["snakeviz", "profiler_data2.prof"])