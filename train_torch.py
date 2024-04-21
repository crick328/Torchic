import mnist    #https://github.com/datapythonista/mnist
from PIL import Image
import torch
from torch import nn, optim, tensor, argmax, float32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
total_train_samples = train_images.shape[0]

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# preprocessing
train_images = preprocess_images(train_images)
train_labels = labels_to_OHE(train_labels)
#test_labels = labels_to_OHE(test_labels)

total_test_samples, height, width = test_images.shape

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = nn.Sequential(
    nn.Linear(height*width, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Softmax() #nn.Sigmoid()
   )
model = model.float()
model.to(device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
#raise Exception("Napoli")
print(count_parameters(model))


import time
import random

epochs = 1
batch_size = 300
start = 0
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("Inizio training...")
print(f"batch per epochs da fare: {total_train_samples/batch_size : .0f}")
start_time = time.time()

for epoch in range(0, epochs):
    print(f"=== Epoch {epoch+1} ===")
    while start < total_train_samples:
        loss_sum = 0
        correct_predictions = 0

        for i in range(start, start+batch_size):
            predicted = model(tensor(train_images[i], device=device, dtype=float32))
            loss = loss_fn(predicted, tensor(train_labels[i], device=device, dtype=float32))
            #loss = loss.mean() # per stampare la loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss
            if argmax(predicted) == train_labels[i].index(1):
                correct_predictions += 1
            
        start += batch_size
        print(f"batch {(start/batch_size) :.0f}, loss: {((loss_sum/batch_size*100).data):.1f}%, accuracy: {(correct_predictions/batch_size*100) :.1f}%")

    # rimescolo il dataset con le labels
    temp = list(zip(train_images, train_labels))
    random.shuffle(temp)
    train_images, retrain_labels = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    train_images, train_labels = list(train_images), list(train_labels)

    start = 0


print(f"Fine training! Tempo impiegato: {((time.time() - start_time)/60):.1f} minuti")
#save_model_type1(model, "mnist 28x28")



# auto eval
correct_predictions = 0
test_images = preprocess_images(test_images)
test_labels = labels_to_OHE(test_labels)

for index in range(0, len(test_images)):
    predicted = model(tensor(test_images[index], dtype=float32))

    if argmax(predicted) == test_labels[index].index(1):
        correct_predictions += 1

print(f"Prediction corrette: {correct_predictions}/{total_test_samples}. Test accuracy: {correct_predictions/total_test_samples*100}%")