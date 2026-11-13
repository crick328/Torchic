from datasets import load_dataset

# Da un file CSV
'''
try:
    hf_dataset = load_dataset("csv", data_files="torchic\datasets\SG_en-it\SG_transaltion_dataset.csv")
except FileNotFoundError:
    print("Non sono in locale")
    hf_dataset = load_dataset("csv", data_files="SG_en-it\SG_translation_dataset.csv")
'''

hf_dataset = load_dataset(
    "csv", 
    data_files="torchic\datas\SG_en-it\SG_transaltion_dataset.csv", 
    delimiter="|",                      # specificaro il separatore
    column_names=["name", "en", "it"],  # (ri)nomino le colonne
    usecols=["en", "it"],               # ignoro la colonna 'name'
    split="train"                       # carico tutto come unico split
)

#hf_dataset.save_to_disk("path/to/saved_dataset")
