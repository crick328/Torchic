import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchic import nn, Tensor
import numpy as np



# https://ml4a.github.io/ml4a/looking_inside_neural_nets/
DEVICE = "cpu"
PROFILER = False
EPOCHS = 1
BATCH_SIZE = 300
LIB_TRAINING_LOOP=False



model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Softmax(0)
)

#for item in model[0].parameters():
    #print(item.shape())
#print(model.state_dict())


model.to(DEVICE)
loss_fn = nn.SELoss()
optimizer = nn.optim.SGD(model.parameters(), lr=0.01)

print(model.state_dict())


x = Tensor(np.random.randn(10), requires_grad=True)
y = model(x)

y.mean().backward()

for param in optimizer.parameters:
    print(param.grad)

optimizer.step()
optimizer.zero_grad()