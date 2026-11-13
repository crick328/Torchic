import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic import nn
from torchic.data.dataloaders.cifar import AutoCifar
from torchic.tools.training import metrics
#from torchic.tools.visualization import show
from torchic.tools.visualization.graphs import draw_graph
import time
import random
import torchic
from torchic.nn.models.save_load import save, load_architecture
from torchic.autograd.cuda.funct_cost import branch_time_estimator


# https://ml4a.github.io/ml4a/looking_inside_neural_nets/
DEVICE = "cpu"
PROFILER = False
EPOCHS = 1
BATCH_SIZE = 300
LIB_TRAINING_LOOP=False

torchic.set_default_precision(32)
torchic.set_default_device(DEVICE)



train_images, train_labels = AutoCifar.load_train_data()
test_images, test_labels = AutoCifar.load_test_data()

total_train_samples = len(train_images)
total_test_samples = len(test_images)
height, width = 32, 32


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

#print(list(model.parameters()))

#model.to(DEVICE)

loss_fn = nn.MSELoss()
optimizer = nn.optim.SGD(model.parameters(), lr=0.01)
#optimizer = nn.optim.AdamV3(model.parameters(), lr=0.01)
N_BATCH = int(total_train_samples/BATCH_SIZE)


if PROFILER:
    import cProfile, pstats
    import subprocess

    profiler = cProfile.Profile()
    profiler.enable()
start_time = time.time()

if LIB_TRAINING_LOOP:
    metrics = metrics.supervised_training(
        model=model,
        train=train_images,
        train_labels=train_labels,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eval=test_images,
        eval_labels=test_labels
    )
    metrics.show_metrics()
else:
    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0
        current_iter = 0
        correct_predictions = 0

        while current_iter < total_train_samples:
            for i in range(current_iter, current_iter + BATCH_SIZE):
                predicted = model(train_images[i])

                loss = loss_fn(predicted, train_labels[i])
                loss = loss.mean()
                #draw_graph(predicted)
                #print(branch_time_estimator(loss))
                #exit()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.data

                if predicted.data.argmax() == train_labels[i].data.argmax(): #train_labels[i].tolist().index(1):
                    correct_predictions += 1
                #break
            #break
        
            #optimizer.step()
            #optimizer.zero_grad()

            current_iter += BATCH_SIZE
            print(f"Epoch {epoch}/{EPOCHS}, batch {(current_iter/BATCH_SIZE) :.0f}/{N_BATCH}, loss: {(epoch_loss/current_iter*100):.1f}%, accuracy: {(correct_predictions/current_iter*100) :.1f}%")
        
        # rimescolo il dataset con le labels
        temp = list(zip(train_images, train_labels))
        random.shuffle(temp)
        train_images, train_labels = zip(*temp)
        # TODO: scrivi su foglio le stats dell'epoch
        # TODO: introduci gli optimizers seri!
        #optimizer.lr = optimizer.lr * 0.1


    print(f"Fine training! Tempo impiegato: {((time.time() - start_time)/60):.2f} minuti")

if PROFILER:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
    # Remove dir names
    stats.strip_dirs()
    stats.print_stats()
    profiler.dump_stats("profiler_data2.prof")

    subprocess.run(["snakeviz", "profiler_data2.prof"])

#save(model, optim=None, filename="cazzogigante")
#model = load_architecture("cazzogigante")

metrics.auto_eval(model, test_images, test_labels)