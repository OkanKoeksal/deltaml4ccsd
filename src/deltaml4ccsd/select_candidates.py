#!/usr/bin/env python3
"""
Central wrapper for candidate-structure selection.

Usage:
    PYTHONPATH=src python -m deltaml4ccsd.select_candidates \
        --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json
"""

import argparse
import runpy
from pathlib import Path

from .common import load_config, require_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config, dataset_dir = load_config(args.config)
    dataset_dir = Path(dataset_dir)

    script_path = dataset_dir / "2_select_candidate_structures.py"
    require_file(script_path)

    print("Running candidate-selection workflow:")
    print("  System:      ", config["system_label"])
    print("  Descriptor:  ", config["descriptor_label"])
    print("  Low level:   ", config["low_level_label"])
    print("  Dataset dir: ", dataset_dir)
    print("  Script:      ", script_path)
    print()

    old_cwd = Path.cwd()

    try:
        import os
        os.chdir(str(dataset_dir))
        runpy.run_path("2_select_candidate_structures.py", run_name="__main__")
    finally:
        os.chdir(str(old_cwd))


if __name__ == "__main__":
    main()