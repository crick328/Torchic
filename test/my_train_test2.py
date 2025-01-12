import sys
sys.path.insert(1, r"C:\Users\admin\Desktop\NN Lab\Projects\Torchic\Torchic 3.0 Beta")
import torchic
from torchic import nn
from torchic.tools.helpers import cat
from torchic.data.dataloaders.my_mnist import AutoMnist
from torchic.data.dataloaders.base_classes import DataLoader, Dataset


DEVICE = "cpu"
PROFILER = False
EPOCHS = 1
BATCH_SIZE = 300
LIB_TRAINING_LOOP=False

torchic.set_default_precision(32)
torchic.set_default_device(DEVICE)


train_images = AutoMnist.train_images(preprocess=True)
train_labels = AutoMnist.train_labels(one_hot_encode=True)

test_images = AutoMnist.test_images(preprocess=True)
test_labels = AutoMnist.test_labels(one_hot_encode=True)

total_train_samples = len(train_images)
total_test_samples = len(test_images)
height, width = 28, 28


model = nn.Sequential(
    nn.Linear(height*width, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax(0)
)
#print(list(model.parameters()))
#print(model.state_dict())


model.to(DEVICE)