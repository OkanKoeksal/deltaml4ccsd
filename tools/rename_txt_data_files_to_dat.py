#!/usr/bin/env python3
"""
Rename core dataset .txt files to .dat and update references in scripts/configs.

Run from repository root.

Dry run:
    python3.6 tools/rename_txt_data_files_to_dat.py

Execute:
    python3.6 tools/rename_txt_data_files_to_dat.py --execute

This script only renames selected core data files, not every .txt file.
It updates references inside:
    - *.py
    - *.json
    - README.md
    - *.md

It intentionally does NOT rename output/report files such as:
    - Values_Within_Range_*.txt
    - all_regression_metrics*.txt
    - best_hyperparameters*.txt
    - val_indices*.txt
    - test_indices*.txt
"""

import argparse
from pathlib import Path


REPO_ROOT = Path.cwd()
DATASETS_DIR = REPO_ROOT / "datasets"
SRC_DIR = REPO_ROOT / "src"


RENAME_MAP = {
    "CCSD.txt": "CCSD.dat",
    "MP2.txt": "MP2.dat",
    "DFT.txt": "DFT.dat",
    "processed_features.txt": "processed_features.dat",
    "processed_xyz_files.txt": "processed_xyz_files.dat",

    # Include these only in case older names still exist somewhere.
    "CCSD_ML_Final.txt": "CCSD.dat",
    "MP2_ML_Final.txt": "MP2.dat",
    "DFT_ML_Final.txt": "DFT.dat",
    "processed_features_final.txt": "processed_features.dat",
    "processed_xyz_files_final.txt": "processed_xyz_files.dat",

    "CCSD_ML_Final_reordered.txt": "CCSD.dat",
    "MP2_ML_Final_reordered.txt": "MP2.dat",
    "DFT_ML_Final_reordered.txt": "DFT.dat",
    "processed_features_final_reordered.txt": "processed_features.dat",
    "processed_xyz_files_reordered.txt": "processed_xyz_files.dat",
}


TEXT_SUFFIXES_TO_UPDATE = {
    ".py",
    ".json",
    ".md",
    ".toml",
}


def find_dataset_files_to_rename():
    actions = []

    if not DATASETS_DIR.exists():
        raise FileNotFoundError("datasets/ not found. Run this from the repository root.")

    for path in DATASETS_DIR.rglob("*"):
        if not path.is_file:
            continue

        old_name = path.name

        if old_name not in RENAME_MAP:
            continue

        new_path = path.with_name(RENAME_MAP[old_name])

        if path == new_path:
            continue

        actions.append((path, new_path))

    return actions


def find_text_files_to_update():
    candidates = []

    search_roots = [
        REPO_ROOT / "README.md",
        SRC_DIR,
        DATASETS_DIR,
    ]

    for root in search_roots:
        if root.is_file():
            candidates.append(root)
        elif root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in TEXT_SUFFIXES_TO_UPDATE:
                    candidates.append(path)

    return sorted(set(candidates))


def update_text_file(path, execute):
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return False

    new_text = text

    for old_name, new_name in RENAME_MAP.items():
        new_text = new_text.replace(old_name, new_name)

    if new_text == text:
        return False

    print("[UPDATE REF] {}".format(path.relative_to(REPO_ROOT)))

    if execute:
        path.write_text(new_text)

    return True


def rename_files(actions, execute):
    for old_path, new_path in actions:
        rel_old = old_path.relative_to(REPO_ROOT)
        rel_new = new_path.relative_to(REPO_ROOT)

        if new_path.exists():
            print("[SKIP RENAME] {} -> {} because target already exists".format(rel_old, rel_new))
            continue

        print("[RENAME] {} -> {}".format(rel_old, rel_new))

        if execute:
            old_path.rename(new_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files and update references. Without this, only prints actions.",
    )
    args = parser.parse_args()

    print("Repository root:")
    print("  {}".format(REPO_ROOT))
    print()

    actions = find_dataset_files_to_rename()

    if not actions:
        print("No matching dataset .txt files found for renaming.")
    else:
        print("Planned file renames:")
        rename_files(actions, execute=args.execute)

    print()
    print("Reference updates:")
    changed_count = 0

    for path in find_text_files_to_update():
        if update_text_file(path, execute=args.execute):
            changed_count += 1

    print()
    if args.execute:
        print("Done. Renamed files and updated references.")
    else:
        print("Dry run only. No files were changed.")
        print("Run again with --execute to apply changes.")

    print("Files with updated references:", changed_count)


if __name__ == "__main__":
    main()