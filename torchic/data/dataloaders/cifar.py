from torchic.cuda_tensor import Tensor
import urllib.request
import numpy as np
import tempfile
import tarfile
import pickle
import os



n_classes = 10
# URL del dataset
url = f"https://www.cs.toronto.edu/~kriz/cifar-{n_classes}-python.tar.gz"
dataset_dir = tempfile.gettempdir()
tar_filename = f"cifar-{n_classes}-python.tar.gz"
dataset_folder = os.path.join(dataset_dir, tar_filename)



class AutoCifar:
    # scarico il dataset se non esiste gia'
    if not os.path.exists(dataset_folder):
        print("Scaricamento di CIFAR-10...")
        # prendo i dati da "url" e li salvo in "dataset_folder"
        urllib.request.urlretrieve(url, dataset_folder)

    # estraggo i file
    #if os.path.exists(os.path.join(dataset_folder)):
        print("Estrazione dei file...")
        with tarfile.open(dataset_folder, "r:gz") as tar:
            tar.extractall()

    @staticmethod
    def load_batch(filename):
        """Carica un singolo batch di CIFAR-10"""
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')

        images = batch[b"data"]
        labels = batch[b"labels"]

        return images, labels


    @staticmethod
    def load_train_data() -> tuple:
        train_images, train_labels = [], []

        for i in range(1, 6):
            filename = os.path.join(dataset_folder, f"data_batch_{i}")
            images, labels = AutoCifar.load_batch(filename)
            train_images.append(images)
            train_labels.append(labels)
        
        # Unisci i batch
        train_images = np.vstack(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_labels = np.hstack(train_labels)

        return Tensor(train_images), Tensor(train_labels)

    @staticmethod
    def load_test_data():
        # Carica il batch di test
        test_images, test_labels = AutoCifar.load_batch(os.path.join(dataset_folder, "test_batch"))
        test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_labels = np.array(test_labels)

        return Tensor(test_images), Tensor(test_labels)