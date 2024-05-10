import mnist    #https://github.com/datapythonista/mnist, pip install mnist
from torchic import nn, optim, Tensor, helper
import time
import random
import torchic
torchic.set_defalt_dtype(torchic.float32)

# https://ml4a.github.io/ml4a/looking_inside_neural_nets/
DEVICE = "cpu"
PROFILER = False
EPOCHS = 1
BATCH_SIZE = 300


def preprocess_images(images):
    output = []
    for image in images:
        output.append(Tensor(image.flatten()/255, device=DEVICE))
    return output


train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = preprocess_images(train_images)
train_labels = helper.labels_to_OHE(train_labels, device=DEVICE)
total_train_samples = len(train_images)

total_test_samples, height, width = test_images.shape
test_images = preprocess_images(test_images)
test_labels = helper.labels_to_OHE(test_labels, device=DEVICE)


model = nn.Sequential(
    nn.Linear(height*width, 10, bias=False),
    nn.Softmax(0)
    #nn.LogSoftmax(0)
)

print(model.linear0.weights.requires_grad)

print(model.number_of_params())
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.SELoss()


print("Inizio training...")
print(f"batch per epochs da fare: {total_train_samples/BATCH_SIZE : .0f}")

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

    while current_iter < total_train_samples:
        for i in range(current_iter, current_iter + BATCH_SIZE):
            predicted = model(train_images[i])

            loss = loss_fn(predicted, train_labels[i])
            loss = loss.mean()
            loss.backward()

            epoch_loss += loss.data

            if predicted.data.argmax() == train_labels[i].data.argmax(): #train_labels[i].tolist().index(1):
                correct_predictions += 1

        optimizer.step()
        optimizer.zero_grad()

        current_iter += BATCH_SIZE
        print(f"Epoch {epoch}, batch {(current_iter/BATCH_SIZE) :.0f}, loss: {(epoch_loss/current_iter*100):.1f}%, accuracy: {(correct_predictions/current_iter*100) :.1f}%")

    # rimescolo il dataset con le labels
    temp = list(zip(train_images, train_labels))
    random.shuffle(temp)
    train_images, train_labels = zip(*temp)
    # TODO scrivi su foglio le stats dell'epoch

    # diminuisci il lr ogni N epochs
    if epoch % 2 == 0:
        optimizer.lr = optimizer.lr * 0.1


print(f"Fine training! Tempo impiegato: {((time.time() - start_time)/60):.1f} minuti")

if PROFILER:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
    # Remove dir names
    stats.strip_dirs()
    stats.print_stats()
    profiler.dump_stats("profiler_data2.prof")

    subprocess.run(["snakeviz", "profiler_data2.prof"])


# auto eval
predicted_list = []
ground_truth = []
correct_predictions = 0

for index in range(0, len(test_images)):
    predicted = model(test_images[index])

    if predicted.data.argmax() == test_labels[index].data.argmax():
        correct_predictions += 1

    predicted_list.append(predicted.data.argmax())
    ground_truth.append(test_labels[index].data.argmax())

print(f"Prediction corrette: {correct_predictions}/{total_test_samples}. Test accuracy: {correct_predictions/total_test_samples*100}%")


helper.show_out_connections(model.__dict__['linear0'].weights, 'out_connections')
helper.show_confusion_matrix(predicted_list, ground_truth, "confusion_matrix.png")
#show_matrix_values(model.__dict__['linear0'].weights, 'w_matrix_values')
#show_matrix_distribution(model.__dict__['linear0'].weights)