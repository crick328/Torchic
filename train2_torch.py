import mnist    #https://github.com/datapythonista/mnist, pip install mnist
from torch import nn, optim, Tensor


one_hot_encoding_dict = {
    0: [1,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0],
    8: [0,0,0,0,0,0,0,0,1,0],
    9: [0,0,0,0,0,0,0,0,0,1]
}
DEVICE = "cpu"
PROFILER = True

#4.5 min su gpu (+-) w profiler
#3.8 min su cpu (?) w profiler

def labels_to_OHE(labels):
    output = []
    for label in labels:
        output.append(one_hot_encoding_dict[label])
    return output

def preprocess_images(images):
    output = []
    for image in images:
        output.append(Tensor(image.flatten()/255, device=DEVICE))
    return output


# vorrei renderle delle tuple ordinate con (img, label)
train_images = mnist.train_images()
train_labels = mnist.train_labels()
total_train_samples = train_images.shape[0]

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# preprocessing
train_images = preprocess_images(train_images)
train_labels = Tensor(labels_to_OHE(train_labels), device=DEVICE)


total_test_samples, height, width = test_images.shape
print(height)


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


import time
import random

epochs = 1
batch_size = 300
start = 0
print(model._parameters)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


print("Inizio training...")
print(f"batch per epochs da fare: {total_train_samples/batch_size : .0f}")

if PROFILER:
    import cProfile, pstats
    import subprocess

    profiler = cProfile.Profile()
    profiler.enable()
start_time = time.time()


for epoch in range(0, epochs):
    print(f"=== Epoch {epoch+1} ===")
    while start < total_train_samples:
        loss_sum = 0
        correct_predictions = 0

        for i in range(start, start+batch_size):
            predicted = model(train_images[i])
            # su pytorch è una classe!
            loss = loss_fn(predicted, train_labels[i])
            loss = loss.mean()
            loss.backward()

            #optimizer.step()
            #optimizer.zero_grad()

            loss_sum += loss

            if predicted.data.argmax() == train_labels[i].argmax(): #train_labels[i].tolist().index(1):
                correct_predictions += 1

        optimizer.step()
        optimizer.zero_grad()

        start += batch_size
        print(f"batch {(start/batch_size) :.0f}, loss: {((loss_sum/batch_size*100).data):.1f}%, accuracy: {(correct_predictions/batch_size*100) :.1f}%")

    # rimescolo il dataset con le labels
    temp = list(zip(train_images, train_labels.data))
    random.shuffle(temp)
    train_images, train_labels = zip(*temp)
    print(train_labels)
    # train_images and train_labels come out as tuples, and so must be converted to lists.
    #train_images, train_labels = list(train_images), Tensor(train_labels, device=DEVICE)

    start = 0
    #optimizer.lr = optimizer.lr * 0.1


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
test_images = preprocess_images(test_images)
test_labels = Tensor(labels_to_OHE(test_labels), device=DEVICE)

for index in range(0, len(test_images)):
    predicted = model(test_images[index])

    if predicted.data.argmax() == test_labels[index].argmax():
        correct_predictions += 1

print(f"Prediction corrette: {correct_predictions}/{total_test_samples}. Test accuracy: {correct_predictions/total_test_samples*100}%")