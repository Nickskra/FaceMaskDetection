import yaml
import os

def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(current_dir, "..", "config", "config.yaml")
    config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigurasi tidak ditemukan di: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)