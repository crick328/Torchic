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
        # non converto la label in Tensor, sarebbe uno spreco
        return {
            "image": Tensor(self.dataset[idx]["image"]),
            "label": self.dataset[idx]["label"]
        }