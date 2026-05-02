#!/usr/bin/env python3
"""
Shared utilities for deltaml4ccsd command-line tools.
"""

import json
from pathlib import Path


def load_config(config_path):
    config_path = Path(config_path)

    with open(str(config_path), "r") as f:
        config = json.load(f)

    dataset_dir = Path(config["dataset_dir"])

    return config, dataset_dir


def require_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError("Required file not found: {}".format(path))

    return path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
