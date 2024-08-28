
# impostazioni del modello
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_trg": "it",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None, #"latest",
        "tokenizer_file": "tokenizer_{0}.json",
        # per tensorboard
        "experiment_name": "runs/tmodel"
    }