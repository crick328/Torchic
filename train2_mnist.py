import mnist    #https://github.com/datapythonista/mnist, pip install mnist
from replica import nn, optim, CudaTensor, labels_to_OHE
import time
import random


DEVICE = "cpu"
PROFILER = False
EPOCHS = 1
BATCH_SIZE = 300

#4.5 min su gpu (+-) w profiler (3.3 min no profiler float32 w nuova softmax)
#3.8 min su cpu (?) w profiler


def preprocess_images(images):
    output = []
    for image in images:
        output.append(CudaTensor(image.flatten()/255, device=DEVICE))
    return output


# vorrei renderle delle tuple ordinate con (img, label)
train_images = mnist.train_images()
train_labels = mnist.train_labels()
#total_train_samples = train_images.shape[0]

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# preprocessing (entrambi liste di tensor)
# prova tensor di tensor!
train_images = preprocess_images(train_images)
train_labels = labels_to_OHE(train_labels, device=DEVICE)
total_train_samples = len(train_images)

total_test_samples, height, width = test_images.shape
test_images = preprocess_images(test_images)
test_labels = labels_to_OHE(test_labels, device=DEVICE)


model = nn.NNetwork(
    nn.LLayer(height*width, 500),
    nn.ReLU(),
    nn.LLayer(500, 250),
    nn.ReLU(),
    nn.LLayer(250, 100),
    nn.ReLU(),
    nn.LLayer(100, 10),
    nn.Softmax(0)
)
#print(model.state_dict())
model.to(DEVICE)


start = 0
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
for epoch in range(0, EPOCHS):
    print(f"=== Epoch {epoch+1} ===")
    while start < total_train_samples:
        loss_sum = 0
        correct_predictions = 0

        for i in range(start, start+BATCH_SIZE):
            predicted = model(train_images[i])
            # su pytorch è una classe!
            loss = loss_fn(predicted, train_labels[i])
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss

            if predicted.data.argmax() == train_labels[i].data.argmax(): #train_labels[i].tolist().index(1):
                correct_predictions += 1

        #optimizer.step()
        #optimizer.zero_grad()

        start += BATCH_SIZE
        print(f"batch {(start/BATCH_SIZE) :.0f}, loss: {((loss_sum/BATCH_SIZE*100).data):.1f}%, accuracy: {(correct_predictions/BATCH_SIZE*100) :.1f}%")

    # rimescolo il dataset con le labels
    temp = list(zip(train_images, train_labels))
    random.shuffle(temp)
    train_images, train_labels = zip(*temp)
    # train_images e train_labels sono restituite come tuple, quindi li ricasto
    # train_images posso anche non riconvertirlo in lista... Tupla va bene
    #train_images, train_labels = list(train_images), list(train_labels)

    start = 0
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


#save_model_type1(model, "mnist 28x28")
#model = load_model_type1("mnist 28x28")



# auto eval
correct_predictions = 0

for index in range(0, len(test_images)):
    predicted = model(test_images[index])

    if predicted.data.argmax() == test_labels[index].data.argmax():
        correct_predictions += 1

print(f"Prediction corrette: {correct_predictions}/{total_test_samples}. Test accuracy: {correct_predictions/total_test_samples*100}%")