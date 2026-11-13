from torchic.data.dataloaders.base_classes  import Dataset
from datasets import load_dataset
from torchic import Tensor



def get_MNIST_dataset():
    # DatasetDict avente gia' train/test split.
    # Ogni entry dataset[i] ha un campo "image"
    # e uno "label"
    raw_dataset = load_dataset("ylecun/mnist")
    return Mnist(raw_dataset["train"]), Mnist(raw_dataset["test"])


class Mnist(Dataset):
    # questo invece sei costretto a definirlo a mano
    def __getitem__(self, idx):
        # NOTA: datasets mi fa lo stack ancor prima che mi arrivi
        # come argomento della classe, quindi non ho bisogno di
        # usare universal_stacker (per ora)
        return {
            "image": Tensor(self.dataset[idx]["image"]),
            "label": Tensor(self.dataset[idx]["label"])
        }