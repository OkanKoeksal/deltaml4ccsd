# deltaml4ccsd

Machine-learning correction of low-level electronic-structure energies toward CCSD quality for homohalogenated borane-phosphine donor-acceptor adducts.

This repository accompanies the publication:

**Δ-Machine Learning toward CCSD Accuracy for Homohalogenated Borane--Phosphine Adducts: Screening Low-Energy Structures from DFT and MP2 Libraries**  
O. Köksal, *Physical Chemistry Chemical Physics*, 2026, Accepted Manuscript.  
DOI: [10.1039/D6CP00985A](https://doi.org/10.1039/D6CP00985A)

---

## Overview

This repository contains the datasets, molecular structures, pretrained models, descriptor files, and Python scripts used to reproduce the Δ-machine-learning workflow described in the publication.

The central idea is to correct low-level electronic-structure energies toward CCSD quality. The machine-learning model learns the energy difference

ΔE = E<sub>CCSD</sub> − E<sub>low-level</sub>

where E<sub>low-level</sub> is either a DFT or MP2 energy. The final predicted CCSD-quality energy is reconstructed as

E<sub>CCSD,predicted</sub> = E<sub>low-level</sub> + ΔE<sub>predicted</sub>

This repository contains two descriptor/model families:

1. **Handcrafted-descriptor Δ-ML models**

   These models use precomputed geometry-based descriptor vectors. Depending on the dataset, the models correct either DFT or MP2 energies toward CCSD quality.

2. **SOAP-descriptor Δ-ML models**

   These models use Smooth Overlap of Atomic Positions, SOAP, descriptors generated from the `.xyz` structures. The SOAP workflow uses support vector regression together with train-only scaling and optional PCA dimensionality reduction.

For both descriptor families, the workflow consists of two steps:

1. **Model training or loading**
2. **Candidate low-energy structure selection**

The dataset-local scripts remain fully executable. In addition, the repository now contains a centralized `src/` layout with wrapper modules and JSON configuration files so that workflows can also be launched from the repository root.

---

## Systems

The repository contains datasets for the following homohalogenated borane-phosphine donor-acceptor adducts:

- `BPFF` denotes F<sub>3</sub>B-PF<sub>3</sub>.
- `BPClCl` denotes Cl<sub>3</sub>B-PCl<sub>3</sub>.
- `BPBrBr` denotes Br<sub>3</sub>B-PBr<sub>3</sub>.

In the directory names, the shorthand labels are used as follows:

- `BPFF` denotes F<sub>3</sub>B-PF<sub>3</sub>.
- `BPClCl` denotes Cl<sub>3</sub>B-PCl<sub>3</sub>.
- `BPBrBr` denotes Br<sub>3</sub>B-PBr<sub>3</sub>.

The available correction tasks are organized as:

```text
CCSD_DFT    DFT -> CCSD correction
CCSD_MP2    MP2 -> CCSD correction
```

---

## Repository layout

The repository is organized as follows:

```text
deltaml4ccsd/
├── README.md
├── pyproject.toml
├── src/
│   └── deltaml4ccsd/
│       ├── __init__.py
│       ├── common.py
│       ├── train_handcrafted.py
│       ├── train_soap.py
│       ├── select_candidates.py
│       ├── reorder_dataset.py
│       ├── standardize_structures.py
│       └── configs/
│           ├── BPBrBr_CCSD_DFT_handcrafted.json
│           ├── BPBrBr_CCSD_MP2_handcrafted.json
│           ├── BPBrBr_CCSD_MP2_soap.json
│           ├── BPClCl_CCSD_DFT_handcrafted.json
│           ├── BPClCl_CCSD_MP2_handcrafted.json
│           ├── BPClCl_CCSD_MP2_soap.json
│           ├── BPFF_CCSD_DFT_handcrafted.json
│           ├── BPFF_CCSD_MP2_handcrafted.json
│           └── BPFF_CCSD_MP2_soap.json
└── datasets/
    ├── Handcrafted_Descriptors/
    │   ├── BPFF/
    │   │   ├── CCSD_DFT/
    │   │   └── CCSD_MP2/
    │   ├── BPClCl/
    │   │   ├── CCSD_DFT/
    │   │   └── CCSD_MP2/
    │   └── BPBrBr/
    │       ├── CCSD_DFT/
    │       └── CCSD_MP2/
    │
    └── SOAP_Descriptors/
        ├── BPFF/
        │   └── CCSD_MP2/
        ├── BPClCl/
        │   └── CCSD_MP2/
        └── BPBrBr/
            └── CCSD_MP2/
```

The `datasets/` directory contains the actual data, molecular structures, pretrained models, and dataset-local scripts.

The `src/deltaml4ccsd/` directory contains centralized Python entry points and configuration files. These wrappers allow workflows to be started from the repository root while still using the dataset-local scripts and files.

---

## Dataset directory contents

### Handcrafted-descriptor directories

A typical handcrafted-descriptor directory contains files such as:

```text
CCSD.dat
DFT.dat or MP2.dat
processed_features.dat
processed_xyz_files.dat
Structures/
pretrained_gbr_delta_ccsd_*.pkl
pretrained_svr_delta_ccsd_*.pkl
pretrained_gpr_delta_ccsd_*.pkl
1_train_handcrafted_delta_ml_models.py
2_select_candidate_structures.py
```

Typical output directories/files after running the scripts are:

```text
Predictions/
Plots_HeldOut_Set/
Found_Within_MAE_Range/
all_regression_metrics.txt
Values_Within_Range_*.txt
```

### SOAP-descriptor directories

A typical SOAP-descriptor directory contains files such as:

```text
CCSD.dat
MP2.dat
processed_xyz_files.dat
Structures/
soap_features.npy
soap_scaler.pkl
soap_pca.pkl
pretrained_svr_soap_delta_ccsd_mp2.pkl
1_train_soap_delta_ml_models.py
2_select_candidate_structures.py
```

Typical output directories/files after running the scripts are:

```text
Predictions_SOAP/
Plots_HeldOut_Set_SOAP/
Found_Within_MAE_Range_SOAP/
all_regression_metrics_soap.txt
Values_Within_Range_SOAP_SVR_SOAP_MP2.txt
```

---

## Important file conventions

All relevant files inside a dataset directory are row-aligned. This means that row `i` in the following files refers to the same molecular structure:

```text
processed_features.dat
processed_xyz_files.dat
DFT.dat or MP2.dat
CCSD.dat, where available
```

The file `processed_xyz_files.dat` defines the authoritative structure order. The structures have been standardized to a uniform naming convention:

```text
structure0.xyz
structure1.xyz
structure2.xyz
...
```

For each dataset, the first part of the file contains structures with known CCSD reference energies. These rows are used for model training and held-out testing. The remaining rows are treated as the low-level structure library for which CCSD-quality energies are predicted.

The prediction files are written such that prediction line 1 corresponds to dataset row `N_TRAIN`.

```text
prediction line 1 -> dataset row N_TRAIN
prediction line 2 -> dataset row N_TRAIN + 1
```

The values of `N_TOTAL`, `N_CCSD` or `N_KNOWN`, and `N_TRAIN` are defined in the dataset-local scripts and in the corresponding JSON configuration files under `src/deltaml4ccsd/configs/`.

---

## Running workflows from the dataset directories

Each dataset directory is self-contained. A workflow can therefore be reproduced by entering the corresponding directory and running the two local scripts.

### Handcrafted descriptors

Example for an MP2-to-CCSD workflow:

```bash
cd datasets/Handcrafted_Descriptors/BPBrBr/CCSD_MP2

python 1_train_handcrafted_delta_ml_models.py
python 2_select_candidate_structures.py
```

Example for a DFT-to-CCSD workflow:

```bash
cd datasets/Handcrafted_Descriptors/BPFF/CCSD_DFT

python 1_train_handcrafted_delta_ml_models.py
python 2_select_candidate_structures.py
```

### SOAP descriptors

Example for a SOAP-based MP2-to-CCSD workflow:

```bash
cd datasets/SOAP_Descriptors/BPFF/CCSD_MP2

python 1_train_soap_delta_ml_models.py
python 2_select_candidate_structures.py
```

The SOAP script uses `soap_features.npy` if it is already present. If the SOAP feature file is removed, the descriptors are regenerated from the `.xyz` structures listed in `processed_xyz_files.dat`.

---

## Running workflows from the repository root

The repository also provides centralized wrappers under `src/deltaml4ccsd/`.

From the repository root, use:

```bash
PYTHONPATH=src python -m deltaml4ccsd.train_handcrafted --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json
PYTHONPATH=src python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json
```

For a SOAP workflow:

```bash
PYTHONPATH=src python -m deltaml4ccsd.train_soap --config src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json
PYTHONPATH=src python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json
```

The JSON configuration files specify the dataset directory, descriptor type, low-level method, row counts, model filenames, prediction filenames, and other workflow settings.

Example configuration files include:

```text
src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json
src/deltaml4ccsd/configs/BPBrBr_CCSD_DFT_handcrafted.json
src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_soap.json
src/deltaml4ccsd/configs/BPClCl_CCSD_MP2_handcrafted.json
src/deltaml4ccsd/configs/BPFF_CCSD_MP2_soap.json
```

---

## Training or loading the Δ-ML model

The training script performs the following operations:

1. Loads the descriptor matrix or SOAP descriptor cache.
2. Loads the ordered low-level and CCSD energy files.
3. Splits the known CCSD block into training and held-out test data.
4. Loads pretrained `.pkl` models if present.
5. Retrains models only if the pretrained model files are absent.
6. Predicts CCSD-quality energies for rows `N_TRAIN:N_TOTAL`.
7. Writes prediction files and summary metrics.

For handcrafted descriptors, the available models are:

```text
Gradient Boosting Regressor
Support Vector Regressor
Gaussian Process Regressor
```

For SOAP descriptors, the current cleaned workflow uses:

```text
SOAP + Support Vector Regressor
```

---

## Candidate low-energy structure selection

After the training/prediction step, run the candidate-selection script.

Dataset-local form:

```bash
python 2_select_candidate_structures.py
```

Central wrapper form:

```bash
PYTHONPATH=src python -m deltaml4ccsd.select_candidates --config src/deltaml4ccsd/configs/BPBrBr_CCSD_MP2_handcrafted.json
```

The candidate-selection script performs the following operations:

1. Reads the prediction file.
2. Compares the first held-out prediction block against the corresponding CCSD reference block.
3. Computes regression and error metrics.
4. Uses the measured MAE as an energy window around the most negative predicted CCSD energy.
5. Finds structures whose predicted CCSD energies fall within that window.
6. Writes the selected values to a text file.
7. Copies the corresponding `.xyz` structures into an output directory.

The main outputs are:

```text
all_regression_metrics*.txt
Values_Within_Range*.txt
Plots_HeldOut_Set*/
Found_Within_MAE_Range*/
```

---

## Dependencies

The scripts require Python 3 and the following Python packages:

```text
numpy
pandas
scikit-learn
joblib
optuna
matplotlib
ase
dscribe
```

The SOAP-based workflows require `dscribe`.

A minimal installation can be performed with:

```bash
pip install numpy pandas scikit-learn joblib optuna matplotlib ase dscribe
```

For older local environments, the scripts were also tested in Cygwin/Python workflows. If using Cygwin, make sure that the correct Python environment and package installation paths are active before running the scripts.

---

## Reproducibility notes

- Core numerical input files use the `.dat` extension.
- Human-readable logs, summaries, and candidate lists generally use the `.txt` extension.
- The training scripts define `N_TOTAL`, `N_CCSD` or `N_KNOWN`, `N_TRAIN`, and `RANDOM_STATE`.
- The JSON configuration files under `src/deltaml4ccsd/configs/` mirror the dataset-specific workflow settings.
- Pretrained model files are included where applicable and are loaded automatically.
- If a pretrained model file is removed, the corresponding script will retrain the model using the parameters defined in the script.
- SOAP descriptors are cached in `soap_features.npy` to avoid recomputation.
- Candidate structures are copied based on the row order in `processed_xyz_files.dat`.
- The selected candidates depend on the model prediction file and the MAE window computed from the held-out CCSD block.

---


## Citation

If you use this repository, the trained models, or the accompanying datasets, please cite:

```bibtex
@article{Koeksal2026DeltaMLCCSD,
  author  = {Koeksal, O.},
  title   = {Delta-Machine Learning toward CCSD Accuracy for Homohalogenated Borane--Phosphine Adducts: Screening Low-Energy Structures from DFT and MP2 Libraries},
  journal = {Physical Chemistry Chemical Physics},
  year    = {2026},
  doi     = {10.1039/D6CP00985A}
}
```

---

## License

Please see the `LICENSE` file for licensing information.
