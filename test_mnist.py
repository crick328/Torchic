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

test_images = mnist.test_images()
test_labels = mnist.test_labels()
total_test_samples, height, width = test_images.shape

# preprocessing
test_images = preprocess_images(test_images)
test_labels = labels_to_OHE(test_labels)


from test_NNet import load_model_type1, NNetwork

# non funziona, literal_eval non accetta funzioni. Devi organizzare il modello in un
# json facile da ricomporre. Puoi creare un metodo stat_dict() in ogni oggetto, da usare
# per creare il json. Devi solo cambiare la prima riga di save
params = load_model_type1("mnist 28x28")
print(params)
model = NNetwork(params)

# auto eval
correct_predictions = 0

for index in range(0, len(test_images)):
    predicted = model(test_images[index])

    if predicted.index(max(predicted)) == test_labels.index(1):
        correct_predictions += 1

print(f"Prediction corrette: {correct_predictions}/{total_test_samples}. Test accuracy: {correct_predictions/total_test_samples*100}%")