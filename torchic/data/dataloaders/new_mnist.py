import os
os.environ['KAGGLE_CONFIG_DIR'] = r'your\path'
from kaggle.api.kaggle_api_extended import KaggleApi

from base_classes import Dataset


class MnistDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__(data)

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)