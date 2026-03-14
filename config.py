from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_target": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        'experiment_name': "runs/tmodel",
        "datasource": 'opus_books'
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config: dict) -> str:
    model_folder = config["model_folder"]
    model_basename = f"{config['model_basename']}*"
    weight_files = list(Path(model_folder).glob(model_basename))
    if len(weight_files) == 0:
        return None

    weight_files.sort()

    return str(weight_files[-1])
