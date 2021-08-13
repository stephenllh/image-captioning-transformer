import os
import yaml


def load_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)
    return config
