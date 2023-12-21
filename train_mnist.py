import mnist    #https://github.com/datapythonista/mnist
from PIL import Image

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

def labels_to_OHE(labels):
    output = []
    for label in labels:
        output.append(one_hot_encoding_dict[label])
    return output

def preprocess_images(images):
    output = []
    for image in images:
        output.append(image.flatten()/255)
    return output

def preprocess_image(image):
    return image.flatten()/255


# vorrei renderle delle tuple ordinate con (img, label)
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# preprocessing
train_images = preprocess_images(train_images)
train_labels = labels_to_OHE(train_labels)
#test_labels = labels_to_OHE(test_labels)

total_test_samples, height, width = test_images.shape


from test_NNet import NNetwork, LLayer, Sigmoid, SELoss, save_model_type1#, load_model_type1
from optimizers import SGD

model = NNetwork((
    LLayer(height*width, 400),
    Sigmoid(),
    LLayer(400, 10),
    Sigmoid()
))


# training (il dataset andrebbe rimescolato dopo ogni epoch)
import time

epochs = 1
batch_size = 60
start = 0
optimizer = SGD(model.parameters())

print("Inizio training...")
start_time = time.time()

for epoch in range(0, epochs):
    loss_sum = 0
    for i in range(start, start+batch_size):
        predicted = model(train_images[i])
        loss = SELoss(predicted, train_labels[i])
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        start += batch_size
        loss_sum += loss
    print(f"batch {start/batch_size + 1}, loss: {loss_sum/batch_size}")

print(f"Fine training! Tempo impiegato: {(time.time() - start_time)/60} minuti")
save_model_type1(model, "mnist 28x28")


# eval
import random

samples = 20

for i in range(0, samples):
    index = random.randint(0, total_test_samples-1)
    preprocessed_image = preprocess_image(test_images[index])
    preticted = model(preprocessed_image)

    img = Image.fromarray(test_images[index])
    print(test_labels[index])
    img.show(test_labels[index])

    input(f"Premi invio per un altro sample... {i}/{samples}")