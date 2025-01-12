import datasets as ds

#dataset = ds.load_dataset("opus_books", 'en-it')
#train_test_split = dataset['train'].train_test_split(test_size=0.1)
# la maggior parte dei dataset non e' pre splittata. I dati
# vengono messi tutti nello split principale, cioe' "train"
# Alternativamente, si puo' chiamare:
dataset = ds.load_dataset("opus_books", 'en-it', split="train")
train_test_split = dataset.train_test_split(test_size=0.1)
print(type(train_test_split["train"]))
print(type(dataset))