import sys
sys.path.insert(1, "C:/Users/admin/Desktop/NN Lab/Projects/Torchic/Torchic 2.3")
from torchic import nn
from torchic.datasets.my_mnist import AutoMnist
from torchic.tools import training
from torchic.tools.visualization import show
from torchic.tools.visualization.graphs import draw_graph
import time
import random
import torchic


# https://ml4a.github.io/ml4a/looking_inside_neural_nets/
DEVICE = "cuda"
PROFILER = False
EPOCHS = 1
BATCH_SIZE = 300
LIB_TRAINING_LOOP=False

torchic.set_default_precision(32)
torchic.set_default_device(DEVICE)

# NOTE: IL BATCHING E' IMPLEMENTATO IN MODO NAIVE!
# Dovresti manipolare i tensori per creare una batch dim,
# proprio come nel transformer!
# Magari correggi in un train_test_v2.py 



# 3.7 min su cpu no profiler np.float64 (grad come tensor)
# 3.7 min su cpu no profiler np.float64 (grad come np|cp)
# 3.7 min su cpu no profiler np.float64 (Sequential con slot e forward con e senza .values())
# 5.4 min su cpu no profiler np.float32 (ma gradient con np.float64)
# 2.2! min su cpu no profiler np.float32 (all)
# 3.7 min su gpu no profiler np.float32 nell'init
# 3.7 min su gpu no profiler cp.float32 nell'init
# 3.7 min su cpu no profiler np.float64 (settato da set_default_dtype)

# 8.24 minuti con AdamV2 w profiler
# 8.37 minuti con Adam w profiler (18505417 function calls (16885417 primitive calls)) (non traina)
# 7.37 minuti con AdamV2 (no vettori) w profiler
# 9.84 minuti con AdamV2 (vettori) w profiler
# 9.87 minuti con AdamV2 (vettori e zip) w profiler
# 8.36 minuti con AdamV3 (vettori, zip e in-place ops) w profiler
# 7.62 minuti con AdamV3 (vettori, zip, in-place ops e inizio di op fuori da loop) w profiler
# 7.19 minuti con AdamV3 (vettori, zip, in-place ops e op fuori da loop) w profiler
# 7.01 minuti con AdamV3 (vettori, zip, in-place ops x2 e op fuori da loop) w profiler
# 7.68 minuti con AdamV3 (vettori, zip, in-place ops tranne m per testing e op fuori da loop) w profiler
# 6.98 minuti con AdamV3 (vettori, zip, in-place ops x2, op fuori da loop, square)
# 7.23 minuti con AdamV3 (vettori, zip, in-place ops x2, op fuori da loop, square, m_hat e v_hat come val intermedi)
# 7.16 minuti con AdamV3 (all + variabile device)
# 7.08 minuti con AdamV3 (all)
# 7.19 minuti con AdamV3 (all + betas tensors)

# 6.10 minuti con AdamV3 (all) w cuda e profiler

# 3.05 minuti con SGD su cuda no profiler (no __data_manager)



train_images = AutoMnist.train_images(preprocess=True)
train_labels = AutoMnist.train_labels(one_hot_encode=True)

test_images = AutoMnist.test_images(preprocess=True)
test_labels = AutoMnist.test_labels(one_hot_encode=True)

total_train_samples = len(train_images)
total_test_samples = len(test_images)
height, width = 28, 28


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
print(model)

model.to(DEVICE)

loss_fn = nn.SELoss()
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
    metrics = training.supervised_training(
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
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

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

training.auto_eval(model, test_images, test_labels)