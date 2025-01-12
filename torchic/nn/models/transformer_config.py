# per ridurre la memoria occupata, devi:
# 1) creare meno tensori possibili
# 2) avere meno risultati intermedi possibile (operazioni in-place regnano)
# 3) usare precisione mista (da indagare)


# impostazioni del modello
def get_config():
    return {
        "batch_size": 2,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 320,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_trg": "it",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": True, #"latest",
        "tokenizer_file": "tokenizer_{0}.json",
        # per tensorboard
        "experiment_name": "runs/tmodel"
    }