# deltaml4ccsd

Machine-learning correction of low-level electronic-structure energies toward CCSD quality for homohalogenated borane--phosphine donor--acceptor adducts.

This repository accompanies the publication:

**Δ-Machine Learning toward CCSD Accuracy for Homohalogenated Borane--Phosphine Adducts: Screening Low-Energy Structures from DFT and MP2 Libraries**  
O. Köksal, *Physical Chemistry Chemical Physics*, 2026, Accepted Manuscript.  
DOI: [10.1039/D6CP00985A](https://doi.org/10.1039/D6CP00985A)

---

## Overview

This repository contains the datasets, molecular structures, pretrained models, and Python scripts used to reproduce the Δ-machine-learning workflow described in the publication.

The central idea is to correct low-level electronic-structure energies toward CCSD quality. The machine-learning model learns the energy difference

```text
ΔE = E_CCSD - E_low-level
```

where `E_low-level` is either a DFT or MP2 energy. The final predicted CCSD-quality energy is then reconstructed as

```text
E_CCSD,predicted = E_low-level + ΔE_predicted
```

The repository contains two descriptor/model families:

1. **Handcrafted-descriptor Δ-ML models**

   These models use precomputed descriptor vectors based on molecular geometry information. Depending on the dataset, the models correct either DFT or MP2 energies toward CCSD quality.

2. **SOAP-descriptor Δ-ML models**

   These models use Smooth Overlap of Atomic Positions (SOAP) descriptors generated from the `.xyz` structures. The SOAP workflow uses support vector regression together with train-only scaling and optional PCA dimensionality reduction.

For both descriptor families, the workflow consists of two steps:

1. **Model training or loading**
2. **Candidate low-energy structure selection**

The scripts are intentionally stored inside each dataset directory so that each case can be reproduced directly from that directory without changing global paths.

---

## Systems

The repository contains datasets for the following homohalogenated borane--phosphine adducts:

```text
BPFF
BPClCl
BPBrBr
```

The available correction tasks are organized as:

```text
CCSD_DFT    DFT -> CCSD correction
CCSD_MP2    MP2 -> CCSD correction
```

---

## Repository layout

The main data are stored under `datasets/`.

```text
datasets/
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

The `src/` directory is reserved for possible future command-line wrappers or shared helper routines. The current reproducible workflow uses the self-contained scripts inside each dataset directory.

---

## Dataset directory contents

### Handcrafted-descriptor directories

A typical handcrafted-descriptor directory contains files such as:

```text
CCSD.txt
DFT.txt or MP2.txt
processed_features.txt
processed_xyz_files.txt
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
CCSD.txt
MP2.txt
processed_xyz_files.txt
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
processed_features.txt
processed_xyz_files.txt
DFT.txt or MP2.txt
CCSD.txt, where available
```

The file `processed_xyz_files.txt` defines the authoritative structure order. The structures have been standardized to a uniform naming convention:

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

The value of `N_TRAIN` is defined at the top of each training and candidate-selection script.

---

## Workflow

### 1. Train or load the Δ-ML model

Enter a dataset directory and run the training script.

For handcrafted descriptors:

```bash
python 1_train_handcrafted_delta_ml_models.py
```

For SOAP descriptors:

```bash
python 1_train_soap_delta_ml_models.py
```

The training script performs the following operations:

1. Loads the descriptor matrix or SOAP descriptor cache.
2. Loads the ordered low-level and CCSD energy files.
3. Splits the known CCSD block into training and held-out test data.
4. Loads pretrained `.pkl` models if present.
5. Retrains models only if the pretrained model files are absent.
6. Predicts CCSD-quality energies for rows `N_TRAIN:N_TOTAL`.
7. Writes prediction files and summary metrics.

For handcrafted descriptors, the models are:

```text
Gradient Boosting Regressor
Support Vector Regressor
Gaussian Process Regressor
```

For SOAP descriptors, the current cleaned workflow uses:

```text
SOAP + SVR
```

### 2. Select candidate low-energy structures

After the training/prediction step, run:

```bash
python 2_select_candidate_structures.py
```

The candidate-selection script performs the following operations:

1. Reads the prediction file.
2. Compares the first held-out prediction block against the corresponding CCSD reference block.
3. Computes regression/error metrics.
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

## Example: handcrafted descriptors

To reproduce a handcrafted-descriptor MP2-to-CCSD workflow, enter the corresponding directory and run:

```bash
cd datasets/Handcrafted_Descriptors/BPBrBr/CCSD_MP2

python 1_train_handcrafted_delta_ml_models.py
python 2_select_candidate_structures.py
```

For a DFT-to-CCSD workflow:

```bash
cd datasets/Handcrafted_Descriptors/BPFF/CCSD_DFT

python 1_train_handcrafted_delta_ml_models.py
python 2_select_candidate_structures.py
```

---

## Example: SOAP descriptors

To reproduce a SOAP-based MP2-to-CCSD workflow:

```bash
cd datasets/SOAP_Descriptors/BPFF/CCSD_MP2

python 1_train_soap_delta_ml_models.py
python 2_select_candidate_structures.py
```

The SOAP script will use `soap_features.npy` if it is already present. If the SOAP feature file is removed, the descriptors are regenerated from the `.xyz` structures listed in `processed_xyz_files.txt`.

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

- The training scripts define `N_TOTAL`, `N_CCSD` or `N_KNOWN`, `N_TRAIN`, and `RANDOM_STATE` at the top of each file.
- Pretrained model files are included where applicable and are loaded automatically.
- If a pretrained model file is removed, the corresponding script will retrain the model using the parameters defined in the script.
- SOAP descriptors are cached in `soap_features.npy` to avoid recomputation.
- Candidate structures are copied based on the row order in `processed_xyz_files.txt`.
- The selected candidates depend on the model prediction file and the MAE window computed from the held-out CCSD block.

---

## GitHub usage notes

The repository is designed so that each dataset directory is self-contained. A typical local update workflow is:

```bash
git status
git add README.md datasets
git commit -m "Update datasets, scripts, and README"
git pull --rebase origin main
git push origin main
```

If Git reports that `README.md` is untracked and would be overwritten during `git pull --rebase`, either add and commit it first or move it temporarily before pulling.

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
