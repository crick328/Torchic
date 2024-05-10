import mnist    #https://github.com/datapythonista/mnist, pip install mnist
from torchic import nn, optim, Tensor, helper
import time
import random
import torchic
torchic.set_defalt_dtype(torchic.float32)


# Fai in modo che la transizione da codice tuo a Torch sia piu' liscia
# Adatta la sintassi delle Backward Functions?


DEVICE = "cpu"
PROFILER = False
EPOCHS = 2
BATCH_SIZE = 300


# 3.7 min su cpu no profiler np.float64 (grad come tensor)
# 3.7 min su cpu no profiler np.float64 (grad come np|cp)
# 3.7 min su cpu no profiler np.float64 (Sequential con slot e forward con e senza .values())
# 5.4 min su cpu no profiler np.float32 (ma gradient con np.float64)
# 2.2! min su cpu no profiler np.float32 (all)
# 3.7 min su gpu no profiler np.float32 nell'init
# 3.7 min su gpu no profiler cp.float32 nell'init
# 3.7 min su cpu no profiler np.float64 (settato da set_default_dtype)


def preprocess_images(images):
    output = []
    for image in images:
        output.append(Tensor(image.flatten()/255, device=DEVICE))
    return output


# vorrei renderle delle tuple ordinate con (img, label)
train_images = mnist.train_images()
train_labels = mnist.train_labels()
#total_train_samples = train_images.shape[0]

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# preprocessing (entrambi liste di tensor)
train_images = preprocess_images(train_images)
train_labels = helper.labels_to_OHE(train_labels, device=DEVICE)
total_train_samples = len(train_images)

total_test_samples, height, width = test_images.shape
test_images = preprocess_images(test_images)
test_labels = helper.labels_to_OHE(test_labels, device=DEVICE)


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




#print(model.state_dict())
model.to(DEVICE)

print(model.parameters()[0].data.dtype)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.SELoss()


N_BATCH = total_train_samples/BATCH_SIZE
print("Inizio training...")
print(f"batch per epochs da fare: {N_BATCH : .0f}")

if PROFILER:
    import cProfile, pstats
    import subprocess

    profiler = cProfile.Profile()
    profiler.enable()
start_time = time.time()

# PERDI ANCORA TROPPO TEMPO NELL'OPTIMIZER!
for epoch in range(1, EPOCHS+1):
    epoch_loss = 0
    current_iter = 0
    correct_predictions = 0
    t = time.time()

    while current_iter < total_train_samples:
        for i in range(current_iter, current_iter + BATCH_SIZE):
            predicted = model(train_images[i])

            loss = loss_fn(predicted, train_labels[i])
            loss = loss.mean()
            loss.backward()

            #optimizer.step()
            #optimizer.zero_grad()

            epoch_loss += loss.data

            if predicted.data.argmax() == train_labels[i].data.argmax(): #train_labels[i].tolist().index(1):
                correct_predictions += 1
            #break
        #break

        optimizer.step()
        optimizer.zero_grad()

        current_iter += BATCH_SIZE
        print(f"Epoch {epoch}/{EPOCHS}, batch {(current_iter/BATCH_SIZE) :.0f}/{BATCH_SIZE}, loss: {(epoch_loss/current_iter*100):.1f}%, accuracy: {(correct_predictions/current_iter*100) :.1f}%")

    # rimescolo il dataset con le labels
    temp = list(zip(train_images, train_labels))
    random.shuffle(temp)
    train_images, train_labels = zip(*temp)
    # TODO: scrivi su foglio le stats dell'epoch
    # TODO: introduci gli optimizers seri!
    optimizer.lr = optimizer.lr * 0.1

    epoch_min = (time.time() - t) / 60
    print(f"ETA: { epoch_min * (EPOCHS - epoch) : .1f} minuti")


print(f"Fine training! Tempo impiegato: {((time.time() - start_time)/60):.1f} minuti")

if PROFILER:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
    # Remove dir names
    stats.strip_dirs()
    stats.print_stats()
    profiler.dump_stats("profiler_data2.prof")

    subprocess.run(["snakeviz", "profiler_data2.prof"])


#save_model_type1(model, "mnist 28x28")
#model = load_model_type1("mnist 28x28")



# auto eval
correct_predictions = 0

for index in range(0, len(test_images)):
    predicted = model(test_images[index])

    if predicted.data.argmax() == test_labels[index].data.argmax():
        correct_predictions += 1

print(f"Prediction corrette: {correct_predictions}/{total_test_samples}. Test accuracy: {correct_predictions/total_test_samples*100}%")