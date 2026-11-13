import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


import torchic
from datasets import load_from_disk
from torchic.data.dataloaders.base_classes import Dataset
from LLM.data.dataset.SG_dataloader import CausalLMDataLoader


DATASET_PATH = f"LLM/data/SG_ready"

# carico il dataset pre-tokenizzato
# ci vorrebbe un dataset.load_from_disk
dataset = Dataset(load_from_disk(DATASET_PATH))
print(dataset.dataset)
dataloader = CausalLMDataLoader()