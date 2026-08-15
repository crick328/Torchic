from preset import torchic
from torchic import nn
from torchic.tools.visualization.graphs import draw_graph
from torchic.nn.models.save_load import save, load_architecture
from torchic.data.dataloaders.base_classes import DataLoader
from torchic.data.dataloaders.new_mnist import get_MNIST_dataset
from torchic.dtypes import float32
from torchic.amp import autocast
from torchic.amp.grad_scaler import GradScaler
from tqdm import tqdm
#import cupy as cp
#cp.cuda.set_allocator(None)

# non fare softmax + SGD o stai chiedendo di avere vanishing gradients!

# global settings
DEVICE = "cpu"
PROFILER = False

# test settings
INPLACE_RELU = True

# CUDA settings
ENABLE_AMP = False
USE_BFLOAT16 = False
SCALING_FACTOR = 2.0**14 # 2.0**14
STREAMS = 0

# training settings
EPOCHS = 1
BATCH_SIZE = 1000
LR = 0.001 #0.001


torchic.set_default_precision(32)
torchic.set_default_device(DEVICE)
torchic.use_brain_float(USE_BFLOAT16)
torchic.set_num_backward_workers(STREAMS)
# non serve a un cazz
#torchic.set_mixed_precision_enabled(ENABLE_AMP)


train, test = get_MNIST_dataset()


train_dataloader = DataLoader(train, BATCH_SIZE)
test_dataloader = DataLoader(test, BATCH_SIZE)
# matrice con soli 1 sulla diagonale
# fanculo ai OHE
#identity_matrix = torchic.Tensor(np.identity(10), dtype=int8)


height, width = 28, 28
model = nn.Sequential(
    nn.Linear(height*width, 500),
    nn.ReLU(INPLACE_RELU),
    nn.Linear(500, 250),
    nn.ReLU(INPLACE_RELU),
    nn.Linear(250, 10),
    # softmax sulla dimensione delle classi: l'ultima
    #nn.Softmax(-1)
    #nn.Sigmoid()
)
print(model.number_of_params())
#for name, param in model.parameters(with_names=True):
    #print(name, param.shape())

#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()
#optimizer = nn.optim.SGD(model.parameters(), lr=LR)
optimizer = nn.optim.AdamV4(model.parameters(), lr=LR)
# chiamare enable_grad non ha impatto, e' gia' on di default
amp_ctx = autocast if ENABLE_AMP else torchic.enable_grad
scaler = GradScaler(SCALING_FACTOR) if ENABLE_AMP and not USE_BFLOAT16 else None


if PROFILER:
    import cProfile, pstats
    import subprocess

    profiler = cProfile.Profile()
    profiler.enable()


model.train()
for epoch in range(0, EPOCHS):
    batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch+1:02d}")
    # stampo i dati parziali dell'epoch ad ogni batch perche'
    # sarebbe inutile stampare quelli delle singole batch
    epoch_loss = 0
    total_samples = 0
    correct_preds = 0

    for batch in batch_iterator:
        inputs = batch["image"].view(batch["image"].size(0), 784).to(float32)
        # normalizzo (si dovrei farlo prima, ma non ho trovato un setup comodo)
        #print(inputs.dtype) -> uint8
        inputs *= 1/255

        for name, param in model.parameters(with_names=True):
            if not param._device.isfinite(param.data).all():
                print(f"Il NaN è apparso nei pesi di {name}!")

        # (batch, 10)
        with amp_ctx():
            #set_grad_enabled(False)
            #print(CONFIG["precision"], CONFIG["mixed_precision"]["enabled"])
            #print(torchic.Tensor.__add__)
            #exit()
            predicted = model(inputs)
            if not predicted._device.isfinite(predicted.data).all():
                print("Il NaN è apparso nel forward!")
            if ENABLE_AMP:
                if USE_BFLOAT16:
                    pass
                    #assert predicted.dtype == torchic.bfloat16
                else:
                    pass
                    #assert predicted.dtype == torchic.float16
            #predicted = predicted.to(torchic.float32)
            # senno' la loss fa .data su un np.ndarray
            #targets = torchic.Tensor(identity_matrix[batch["label"]])
            targets = torchic.Tensor(batch["label"])

            loss = loss_fn(predicted, targets)


        #draw_graph(loss, dpi=30)
        #exit()
        if ENABLE_AMP and not USE_BFLOAT16:
            #print(loss)
            # la CELoss usa LogSoft, che e' safeness=-1
            assert loss.dtype == torchic.float32, loss.dtype
            scaler.scale(loss).backward()

            for name, param in model.parameters(with_names=True):
                assert param.grad.dtype == float32
                if not param._device.isfinite(param._grad).all():
                    print(f"Il NaN è apparso nel grad di {name}!")

            scaler.step(optimizer)
            scaler.update()
        else:
            # calcoli normali
            #loss = loss.mean() # la loss fa gia' .mean()
            loss.backward()

            for name, param in model.parameters(with_names=True):
                assert param.grad.dtype == float32
                if not param._device.isfinite(param._grad).all():
                    print(f"Il NaN è apparso nel grad di {name}!")

            optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.data *BATCH_SIZE
        total_samples += BATCH_SIZE

        # (batch, classes) -> (batch, )
        predicted_classes = predicted.data.argmax(axis=-1)
        correct_preds += (predicted_classes == batch["label"]).sum()
        batch_iterator.set_postfix({
                "accuracy": f"{correct_preds/total_samples : 6.3f}",
                "loss": f"{epoch_loss/total_samples : 6.3f}"
                })


if PROFILER:
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')#sort_stats('tottime')
    # Remove dir names
    stats.strip_dirs()
    stats.print_stats()
    profiler.dump_stats("profiler_data2.prof")

    subprocess.run(["snakeviz", "profiler_data2.prof"])


model.eval()
with torchic.no_grad():
    batch_iterator = tqdm(test_dataloader, desc=f"Eval: ")
    # stampo i dati parziali dell'epoch ad ogni batch perche'
    # sarebbe inutile stampare quelli delle singole batch
    epoch_loss = 0
    total_samples = 0
    correct_preds = 0

    for batch in batch_iterator:
        inputs = batch["image"].view(batch["image"].size(0), 784).to(float32)
        # normalizzo (si dovrei farlo prima, ma non ho trovato un setup comodo)
        #print(inputs.dtype) -> uint8
        inputs *= 1/255

        # (batch, 10)
        predicted = model(inputs)

        targets = torchic.Tensor(batch["label"])
        loss = loss_fn(predicted, targets)

        epoch_loss += loss.data *BATCH_SIZE
        total_samples += BATCH_SIZE

        # (batch, classes) -> (batch, )
        predicted_classes = predicted.data.argmax(axis=-1)
        correct_preds += (predicted_classes == batch["label"]).sum()
        batch_iterator.set_postfix({
                "accuracy": f"{correct_preds/total_samples : 6.3f}",
                "loss": f"{epoch_loss/total_samples : 6.3f}"
                })




#save(model, optim=None, filename="cazzogigante")
#model = load_architecture("cazzogigante")