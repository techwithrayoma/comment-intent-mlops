import yaml

def load_training_config(path="src/assets/default_training_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
