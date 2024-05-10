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


# prossimo step: automatizzare questo
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
)
model.to(DEVICE)


if PROFILER:
    import cProfile, pstats
    import subprocess

    profiler = cProfile.Profile()
    profiler.enable()


helper.supervised_training(
    model=model,
    train=train_images,
    train_labels=train_labels,
    loss_fn=nn.SELoss(),
    optimizer=optim.SGD(model.parameters(), lr=0.01),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    eval=test_images,
    eval_labels=test_labels
)


if PROFILER:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
    # Remove dir names
    stats.strip_dirs()
    stats.print_stats()
    profiler.dump_stats("profiler_data2.prof")

    subprocess.run(["snakeviz", "profiler_data2.prof"])


predicted_list = []
ground_truth = []


helper.show_out_connections(model.__dict__['linear0'].weights, 'out_connections')
#helper.show_confusion_matrix(predicted_list, ground_truth, "confusion_matrix.png")
#show_matrix_values(model.__dict__['linear0'].weights, 'w_matrix_values')
#show_matrix_distribution(model.__dict__['linear0'].weights)