from pathlib import Path

import yaml


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "external_data"


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["config_path"] = str(path.resolve())
    return config


def resolve_root(config):
    return Path(config.get("root", DEFAULT_ROOT))


def run_directory(root, block, partition, seed):
    return Path(root) / "runs" / block / partition.lower() / f"seed_{seed}"

