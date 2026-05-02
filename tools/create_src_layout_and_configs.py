#!/usr/bin/env python3
"""
Create a centralized src/ layout for deltaml4ccsd and generate JSON config
files linked to the existing datasets.

Run from the repository root:

    python3.6 create_src_layout_and_configs.py

This script does not move or delete dataset files. It only creates:
    - src/deltaml4ccsd/
    - src/deltaml4ccsd/configs/
    - starter Python entry-point modules
    - JSON config files linked to datasets/
    - pyproject.toml if missing
"""

import json
from pathlib import Path


# =====================================================
# Repository paths
# =====================================================

REPO_ROOT = Path.cwd()
DATASETS_DIR = REPO_ROOT / "datasets"
SRC_DIR = REPO_ROOT / "src" / "deltaml4ccsd"
CONFIG_DIR = SRC_DIR / "configs"


# =====================================================
# Known workflow defaults
# =====================================================

RANDOM_STATE_DEFAULTS = {
    ("BPBrBr", "CCSD_MP2", "handcrafted"): 4,
    ("BPBrBr", "CCSD_DFT", "handcrafted"): 92,
    ("BPClCl", "CCSD_MP2", "handcrafted"): 6,
    ("BPClCl", "CCSD_DFT", "handcrafted"): 7,
    ("BPFF", "CCSD_MP2", "handcrafted"): 9,
    ("BPFF", "CCSD_DFT", "handcrafted"): 34,

    ("BPBrBr", "CCSD_MP2", "soap"): 37,
    ("BPClCl", "CCSD_MP2", "soap"): 18,
    ("BPFF", "CCSD_MP2", "soap"): 36,
}


N_TRAIN_DEFAULTS = {
    "BPBrBr": 800,
    "BPClCl": 1027,
    "BPFF": 800,
}


SOAP_SPECIES_DEFAULTS = {
    "BPBrBr": ["B", "P", "Br"],
    "BPClCl": ["B", "P", "Cl"],
    "BPFF": ["B", "P", "F"],
}


SYSTEM_FORMULAS = {
    "BPBrBr": "Br3B-PBr3",
    "BPClCl": "Cl3B-PCl3",
    "BPFF": "F3B-PF3",
}


# =====================================================
# Helpers
# =====================================================

def rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def count_nonempty_lines(path):
    if path is None or not path.exists():
        return None

    with open(str(path), "r") as f:
        return sum(1 for line in f if line.strip())


def first_existing(case_dir, candidates):
    for name in candidates:
        path = case_dir / name
        if path.exists():
            return name
    return None


def require_repo_root():
    if not DATASETS_DIR.exists():
        raise FileNotFoundError(
            "Could not find datasets/. Run this script from the repository root."
        )


def write_text_if_missing(path, content):
    if path.exists():
        print("[SKIP] Existing file: {}".format(rel(path)))
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(path), "w") as f:
        f.write(content)

    print("[WRITE] {}".format(rel(path)))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(path), "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print("[WRITE] {}".format(rel(path)))


def add_warning(config, message):
    if "warnings" not in config:
        config["warnings"] = []
    config["warnings"].append(message)


# =====================================================
# Detection helpers
# =====================================================

def detect_features_file(case_dir):
    return first_existing(
        case_dir,
        [
            "processed_features.txt",
            "processed_features_reordered.txt",
            "processed_features_final_reordered.txt",
            "processed_features_final.txt",
        ],
    )


def detect_xyz_list_file(case_dir):
    return first_existing(
        case_dir,
        [
            "processed_xyz_files.txt",
            "processed_xyz_files_reordered.txt",
            "processed_xyz_files_final.txt",
            "processed_xyz_files_reordered_original_names.txt",
        ],
    )


def detect_ccsd_file(case_dir):
    return first_existing(
        case_dir,
        [
            "CCSD.txt",
            "CCSD_reordered.txt",
            "CCSD_ML_Final_reordered.txt",
            "CCSD_ML_Final.txt",
        ],
    )


def detect_low_level_file(case_dir, pair_name):
    if pair_name == "CCSD_MP2":
        candidates = [
            "MP2.txt",
            "MP2_reordered.txt",
            "MP2_ML_Final_reordered.txt",
            "MP2_ML_Final.txt",
        ]
    elif pair_name == "CCSD_DFT":
        candidates = [
            "DFT.txt",
            "DFT_reordered.txt",
            "DFT_ML_Final_reordered.txt",
            "DFT_ML_Final.txt",
        ]
    else:
        candidates = []

    return first_existing(case_dir, candidates)


def low_level_label_from_pair(pair_name):
    if pair_name == "CCSD_MP2":
        return "MP2"
    if pair_name == "CCSD_DFT":
        return "DFT"
    raise ValueError("Unknown pair name: {}".format(pair_name))


def infer_random_state(system, pair_name, descriptor_type):
    return RANDOM_STATE_DEFAULTS.get((system, pair_name, descriptor_type), 1)


def infer_n_train(system):
    return N_TRAIN_DEFAULTS.get(system, 800)


def infer_soap_species(system):
    return SOAP_SPECIES_DEFAULTS.get(system, ["B", "P"])


def validate_common_files(config, case_dir, needs_features):
    if needs_features and config.get("features_file") is None:
        add_warning(config, "No processed feature file found.")

    if config.get("xyz_list_file") is None:
        add_warning(config, "No processed xyz-list file found.")

    if config.get("low_level_energies_file") is None:
        add_warning(config, "No low-level energy file found.")

    if config.get("ccsd_energies_file") is None:
        add_warning(config, "No CCSD energy file found.")

    structures_dir = case_dir / "Structures"
    if not structures_dir.exists():
        add_warning(config, "Structures/ directory not found.")

    xyz_list_name = config.get("xyz_list_file")
    if xyz_list_name is not None:
        xyz_list_path = case_dir / xyz_list_name
        n_xyz = count_nonempty_lines(xyz_list_path)
        config["n_xyz_list"] = n_xyz

        if config.get("n_total") is not None and n_xyz is not None:
            if n_xyz < config["n_total"]:
                add_warning(
                    config,
                    "XYZ list has fewer rows than n_total: {} < {}".format(
                        n_xyz,
                        config["n_total"],
                    ),
                )

    return config


# =====================================================
# Config generation
# =====================================================

def build_handcrafted_config(case_dir, system, pair_name):
    low_level_label = low_level_label_from_pair(pair_name)

    features_file = detect_features_file(case_dir)
    xyz_list_file = detect_xyz_list_file(case_dir)
    low_level_file = detect_low_level_file(case_dir, pair_name)
    ccsd_file = detect_ccsd_file(case_dir)

    n_total = count_nonempty_lines(case_dir / low_level_file) if low_level_file else None
    n_known = count_nonempty_lines(case_dir / ccsd_file) if ccsd_file else None
    n_train = infer_n_train(system)

    config = {
        "system_label": system,
        "system_formula": SYSTEM_FORMULAS.get(system, system),
        "descriptor_label": "handcrafted",
        "low_level_label": low_level_label,

        "dataset_dir": rel(case_dir),
        "features_file": features_file,
        "xyz_list_file": xyz_list_file,
        "low_level_energies_file": low_level_file,
        "ccsd_energies_file": ccsd_file,

        "n_total": n_total,
        "n_known": n_known,
        "n_train": n_train,
        "random_state": infer_random_state(system, pair_name, "handcrafted"),

        "results_dir": "Predictions",
        "plots_dir": "Plots_HeldOut_Set",
        "found_dir": "Found_Within_MAE_Range",
        "metrics_summary": "all_regression_metrics.txt",

        "add_low_level_as_feature": True,
        "use_tail_weighted_ccsd_loss": True,
        "tail_q_ccsd": 0.10,
        "tail_factor_ccsd": 10.0,
        "use_weights_in_fit": True,
        "use_svr_weights": True,

        "gbr_trials": 96,
        "svr_trials": 96,
        "gpr_trials": 96,

        "gbr_model_file": "pretrained_gbr_delta_ccsd_{}.pkl".format(
            low_level_label.lower()
        ),
        "svr_model_file": "pretrained_svr_delta_ccsd_{}.pkl".format(
            low_level_label.lower()
        ),
        "gpr_model_file": "pretrained_gpr_delta_ccsd_{}.pkl".format(
            low_level_label.lower()
        ),

        "prediction_gbr_file": "predicted_ccsd_from_{}_gbr.txt".format(
            low_level_label.lower()
        ),
        "prediction_svr_file": "predicted_ccsd_from_{}_svr.txt".format(
            low_level_label.lower()
        ),
        "prediction_gpr_file": "predicted_ccsd_from_{}_gpr.txt".format(
            low_level_label.lower()
        ),
    }

    if system == "BPFF" and pair_name == "CCSD_DFT":
        config["add_low_level_as_feature"] = False
        config["use_tail_weighted_ccsd_loss"] = False
        config["use_weights_in_fit"] = False
        config["use_svr_weights"] = False

    return validate_common_files(config, case_dir, needs_features=True)


def build_soap_config(case_dir, system, pair_name):
    low_level_label = low_level_label_from_pair(pair_name)

    xyz_list_file = detect_xyz_list_file(case_dir)
    low_level_file = detect_low_level_file(case_dir, pair_name)
    ccsd_file = detect_ccsd_file(case_dir)

    n_total = count_nonempty_lines(case_dir / low_level_file) if low_level_file else None
    n_known = count_nonempty_lines(case_dir / ccsd_file) if ccsd_file else None
    n_train = infer_n_train(system)

    config = {
        "system_label": system,
        "system_formula": SYSTEM_FORMULAS.get(system, system),
        "descriptor_label": "SOAP",
        "low_level_label": low_level_label,

        "dataset_dir": rel(case_dir),
        "xyz_list_file": xyz_list_file,
        "low_level_energies_file": low_level_file,
        "ccsd_energies_file": ccsd_file,

        "n_total": n_total,
        "n_known": n_known,
        "n_train": n_train,
        "random_state": infer_random_state(system, pair_name, "soap"),

        "results_dir": "Predictions_SOAP",
        "plots_dir": "Plots_HeldOut_Set_SOAP",
        "found_dir": "Found_Within_MAE_Range_SOAP",
        "metrics_summary": "all_regression_metrics_soap.txt",

        "soap_npy_file": "soap_features.npy",
        "soap_n_processes": 4,
        "soap_species": infer_soap_species(system),
        "soap_r_cut": 5.5,
        "soap_n_max": 8,
        "soap_l_max": 6,
        "soap_sigma": 0.4,
        "soap_average": "inner",

        "add_low_level_as_feature": True,

        "use_pca": True,
        "pca_n_components": 10,
        "pca_whiten": False,
        "pca_random_state": 0,

        "scaler_file": "soap_scaler.pkl",
        "pca_file": "soap_pca.pkl",

        "use_tail_weighted_ccsd_loss": True,
        "tail_q_ccsd": 0.10,
        "tail_factor_ccsd": 10.0,
        "use_svr_weights": True,

        "svr_trials": 96,
        "svr_n_jobs": -1,

        "svr_model_file": "pretrained_svr_soap_delta_ccsd_{}.pkl".format(
            low_level_label.lower()
        ),
        "prediction_svr_file": "predicted_ccsd_from_{}_soap_svr.txt".format(
            low_level_label.lower()
        ),
    }

    return validate_common_files(config, case_dir, needs_features=False)


def scan_datasets_and_write_configs():
    descriptor_roots = [
        ("handcrafted", DATASETS_DIR / "Handcrafted_Descriptors"),
        ("soap", DATASETS_DIR / "SOAP_Descriptors"),
    ]

    for descriptor_type, root in descriptor_roots:
        if not root.exists():
            print("[SKIP] Missing directory: {}".format(rel(root)))
            continue

        for system_dir in sorted(root.iterdir()):
            if not system_dir.is_dir():
                continue

            system = system_dir.name

            for pair_dir in sorted(system_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue

                pair_name = pair_dir.name

                if pair_name not in {"CCSD_MP2", "CCSD_DFT"}:
                    continue

                if descriptor_type == "soap" and pair_name == "CCSD_DFT":
                    continue

                if descriptor_type == "handcrafted":
                    config = build_handcrafted_config(pair_dir, system, pair_name)
                else:
                    config = build_soap_config(pair_dir, system, pair_name)

                config_name = "{}_{}_{}.json".format(
                    system,
                    pair_name,
                    descriptor_type,
                )

                write_json(CONFIG_DIR / config_name, config)

                if "warnings" in config:
                    print("[WARN] {} has warnings:".format(config_name))
                    for warning in config["warnings"]:
                        print("       - {}".format(warning))


# =====================================================
# Source file templates
# =====================================================

COMMON_PY = '''#!/usr/bin/env python3
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
'''


TRAIN_HANDCRAFTED_PY = '''#!/usr/bin/env python3
"""
Central entry point for handcrafted-descriptor Delta-ML models.

Usage:
    python -m deltaml4ccsd.train_handcrafted --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json

This file is intentionally a thin entry point. Move the cleaned logic from
your dataset-local 1_train_handcrafted_delta_ml_models.py here and replace
hard-coded constants with values from the JSON config.
"""

import argparse

from .common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config, dataset_dir = load_config(args.config)

    print("Loaded handcrafted config:")
    print("  System:      ", config["system_label"])
    print("  Formula:     ", config.get("system_formula", config["system_label"]))
    print("  Low level:   ", config["low_level_label"])
    print("  Dataset dir: ", dataset_dir)
    print()
    print("This is currently a central entry-point scaffold.")
    print("The existing dataset-local training scripts remain the executable reference.")


if __name__ == "__main__":
    main()
'''


TRAIN_SOAP_PY = '''#!/usr/bin/env python3
"""
Central entry point for SOAP-SVR Delta-ML models.

Usage:
    python -m deltaml4ccsd.train_soap --config src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json

This file is intentionally a thin entry point. Move the cleaned logic from
your dataset-local 1_train_soap_delta_ml_models.py here and replace hard-coded
constants with values from the JSON config.
"""

import argparse

from .common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config, dataset_dir = load_config(args.config)

    print("Loaded SOAP config:")
    print("  System:      ", config["system_label"])
    print("  Formula:     ", config.get("system_formula", config["system_label"]))
    print("  Low level:   ", config["low_level_label"])
    print("  Species:     ", config.get("soap_species"))
    print("  Dataset dir: ", dataset_dir)
    print()
    print("This is currently a central entry-point scaffold.")
    print("The existing dataset-local training scripts remain the executable reference.")


if __name__ == "__main__":
    main()
'''


SELECT_CANDIDATES_PY = '''#!/usr/bin/env python3
"""
Central entry point for candidate-structure selection.

Usage:
    python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json

This file is intentionally a thin entry point. Move the cleaned logic from
your dataset-local 2_select_candidate_structures.py here and replace hard-coded
constants with values from the JSON config.
"""

import argparse

from .common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config, dataset_dir = load_config(args.config)

    print("Loaded candidate-selection config:")
    print("  System:      ", config["system_label"])
    print("  Descriptor:  ", config["descriptor_label"])
    print("  Low level:   ", config["low_level_label"])
    print("  Dataset dir: ", dataset_dir)
    print()
    print("This is currently a central entry-point scaffold.")
    print("The existing dataset-local selection scripts remain the executable reference.")


if __name__ == "__main__":
    main()
'''


REORDER_DATASET_PY = '''#!/usr/bin/env python3
"""
Dataset reordering utility.

This module can later host the cleaned logic from 1_Reorder_dataset_keep_known_fixed.py.
"""
'''


STANDARDIZE_STRUCTURES_PY = '''#!/usr/bin/env python3
"""
Structure standardization utility.

This module can later host the cleaned logic from 2_standardize_structure_names.py.
"""
'''


INIT_PY = '''"""
deltaml4ccsd

Delta-machine-learning correction of low-level electronic-structure energies
toward CCSD quality.
"""

__version__ = "0.1.0"
'''


def write_src_files():
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    write_text_if_missing(SRC_DIR / "__init__.py", INIT_PY)
    write_text_if_missing(SRC_DIR / "common.py", COMMON_PY)
    write_text_if_missing(SRC_DIR / "train_handcrafted.py", TRAIN_HANDCRAFTED_PY)
    write_text_if_missing(SRC_DIR / "train_soap.py", TRAIN_SOAP_PY)
    write_text_if_missing(SRC_DIR / "select_candidates.py", SELECT_CANDIDATES_PY)
    write_text_if_missing(SRC_DIR / "reorder_dataset.py", REORDER_DATASET_PY)
    write_text_if_missing(SRC_DIR / "standardize_structures.py", STANDARDIZE_STRUCTURES_PY)


# =====================================================
# pyproject.toml
# =====================================================

PYPROJECT = '''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "deltaml4ccsd"
version = "0.1.0"
description = "Delta-machine-learning correction of DFT and MP2 energies toward CCSD quality."
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
    "optuna",
    "matplotlib",
    "ase",
    "dscribe",
]

[tool.setuptools.packages.find]
where = ["src"]
'''


def write_pyproject():
    write_text_if_missing(REPO_ROOT / "pyproject.toml", PYPROJECT)


def write_config_manifest():
    manifest_path = CONFIG_DIR / "README.md"

    lines = [
        "# Configuration files",
        "",
        "Each JSON file links one dataset case to the centralized Python modules in",
        "`src/deltaml4ccsd`.",
        "",
        "Example for handcrafted descriptors:",
        "",
        "```bash",
        "python -m deltaml4ccsd.train_handcrafted --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json",
        "python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json",
        "```",
        "",
        "Example for SOAP descriptors:",
        "",
        "```bash",
        "python -m deltaml4ccsd.train_soap --config src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json",
        "python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json",
        "```",
        "",
        "At the moment, these central modules are scaffold entry points.",
        "The dataset-local scripts remain the fully executable reference workflows",
        "until the cleaned logic is moved into `src/deltaml4ccsd/`.",
        "",
    ]

    content = "\n".join(lines)
    write_text_if_missing(manifest_path, content)


# =====================================================
# Main
# =====================================================

def main():
    require_repo_root()

    print("Repository root:")
    print("  {}".format(REPO_ROOT))
    print()

    write_src_files()
    scan_datasets_and_write_configs()
    write_config_manifest()
    write_pyproject()

    print()
    print("Done.")
    print("Created or updated:")
    print("  {}".format(SRC_DIR))
    print("  {}".format(CONFIG_DIR))
    print("  {}".format(REPO_ROOT / "pyproject.toml"))


if __name__ == "__main__":
    main()

