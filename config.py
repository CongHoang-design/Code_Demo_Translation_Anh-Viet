from pathlib import Path


def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 5,
        "lr": 10**-4,
        "seq_len": 64,
        "d_model": 256,
        "parquet_base": "hf://datasets/IWSLT/mt_eng_vietnamese@refs%2Fconvert%2Fparquet/iwslt2015-en-vi",
        "lang_src": "en",
        "lang_tgt": "vi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
