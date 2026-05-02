# DeltaML4CCSD

Machine-learning-based correction of lower-level quantum-chemical energies toward CCSD-quality energies for donor-acceptor complexes.

This repository contains the datasets, molecular structures, descriptors, pretrained models, prediction outputs, and analysis scripts used for delta-machine-learning models of the form

```text
Delta E = E_CCSD - E_low-level
```

where the lower-level method is either MP2 or DFT. The final predicted CCSD energy is reconstructed as

```text
E_CCSD(predicted) = E_low-level + Delta E(predicted)
```

Both handcrafted molecular descriptors and SOAP descriptors are provided.

---

## Contents

The repository is organized by descriptor type, molecular system, and low-level reference method.

```text
deltaml4ccsd/
├── datasets/
│   ├── Handcrafted_Descriptors/
│   │   ├── BPBrBr/
│   │   │   ├── CCSD_MP2/
│   │   │   └── CCSD_DFT/
│   │   ├── BPClCl/
│   │   │   ├── CCSD_MP2/
│   │   │   └── CCSD_DFT/
│   │   └── BPFF/
│   │       ├── CCSD_MP2/
│   │       └── CCSD_DFT/
│   │
│   └── SOAP_Descriptors/
│       ├── BPBrBr/
│       │   ├── CCSD_MP2/
│       │   └── CCSD_DFT/
│       ├── BPClCl/
│       │   ├── CCSD_MP2/
│       │   └── CCSD_DFT/
│       └── BPFF/
│           ├── CCSD_MP2/
│           └── CCSD_DFT/
│
├── src/
└── README.md
```

The three molecular systems are:

```text
BPBrBr
BPClCl
BPFF
```

Each system is available for two energy-pair settings:

```text
CCSD_MP2    # Delta-learning from MP2 to CCSD
CCSD_DFT    # Delta-learning from DFT to CCSD
```

---

## Dataset format

Each dataset directory is self-contained. The important files are:

```text
Structures/
processed_xyz_files.txt
processed_features.txt
MP2.txt or DFT.txt
CCSD.txt
```

The rows of these files are aligned. Row `i` in `processed_xyz_files.txt`, `processed_features.txt`, and the lower-level energy file corresponds to the same molecular structure.

For handcrafted descriptor models:

```text
processed_features.txt
```

contains the handcrafted descriptor matrix.

For SOAP models:

```text
soap_features.npy
```

contains the cached SOAP descriptor matrix generated from the structures listed in

```text
processed_xyz_files.txt
```

The lower-level energy file is either

```text
MP2.txt
```

or

```text
DFT.txt
```

depending on the directory. The CCSD reference values are stored in

```text
CCSD.txt
```

The order of these files must not be changed independently.

---

## Handcrafted descriptor workflows

Handcrafted descriptor datasets are located in:

```text
datasets/Handcrafted_Descriptors/
```

A typical handcrafted descriptor directory contains:

```text
Structures/
processed_features.txt
processed_xyz_files.txt
MP2.txt or DFT.txt
CCSD.txt

1_Reorder_dataset_keep_known_fixed.py
2_standardize_structure_names.py
3_train_handcrafted_delta_ml_models.py
4_select_candidate_structures.py

pretrained_gbr_delta_ccsd_*.pkl
pretrained_svr_delta_ccsd_*.pkl
pretrained_gpr_delta_ccsd_*.pkl

Predictions/
Plots_HeldOut_Set/
Found_Within_MAE_Range/
all_regression_metrics.txt
Values_Within_Range_*.txt
```

The main training and prediction script is:

```text
3_train_handcrafted_delta_ml_models.py
```

It loads or trains the following models:

```text
GradientBoostingRegressor
Support Vector Regressor
Gaussian Process Regressor
```

If pretrained `.pkl` files are present, they are loaded directly. If they are absent, the script trains new models and saves them.

The candidate-selection script is:

```text
4_select_candidate_structures.py
```

It compares the predictions against the held-out CCSD reference block, generates regression and error-distribution plots, identifies structures within the MAE window around the most negative predicted CCSD energy, and copies the corresponding `.xyz` files into:

```text
Found_Within_MAE_Range/
```

### Running a handcrafted descriptor workflow

Run the scripts from inside the corresponding dataset directory. For example:

```bash
cd datasets/Handcrafted_Descriptors/BPBrBr/CCSD_MP2

python3 3_train_handcrafted_delta_ml_models.py
python3 4_select_candidate_structures.py
```

The same procedure applies to the other systems and to the DFT-based workflows.

---

## SOAP descriptor workflows

SOAP descriptor datasets are located in:

```text
datasets/SOAP_Descriptors/
```

A typical SOAP descriptor directory contains:

```text
Structures/
processed_xyz_files.txt
MP2.txt or DFT.txt
CCSD.txt

1_train_soap_delta_ml_models.py
2_select_candidate_structures.py

soap_features.npy
soap_scaler.pkl
soap_pca.pkl
pretrained_svr_soap_delta_ccsd_*.pkl

Predictions_SOAP/
Plots_HeldOut_Set_SOAP/
Found_Within_MAE_Range_SOAP/
all_regression_metrics_soap.txt
Values_Within_Range_*_SOAP_*.txt
```

The main SOAP training and prediction script is:

```text
1_train_soap_delta_ml_models.py
```

It performs the following steps:

1. Reads the `.xyz` structures listed in `processed_xyz_files.txt`.
2. Builds or loads SOAP descriptors from `soap_features.npy`.
3. Optionally appends the lower-level energy as an additional scalar feature.
4. Applies train-only feature scaling.
5. Applies train-only PCA if enabled.
6. Loads or trains a SOAP-SVR delta-learning model.
7. Writes predicted CCSD energies for the screening block.

The SOAP candidate-selection script is:

```text
2_select_candidate_structures.py
```

It evaluates the SOAP-SVR predictions on the held-out CCSD block and copies selected structures into:

```text
Found_Within_MAE_Range_SOAP/
```

### Running a SOAP workflow

Run the scripts from inside the corresponding dataset directory. For example:

```bash
cd datasets/SOAP_Descriptors/BPFF/CCSD_MP2

python3 1_train_soap_delta_ml_models.py
python3 2_select_candidate_structures.py
```

---

## Prediction files

For handcrafted descriptor workflows, prediction files are written to:

```text
Predictions/
```

Typical MP2-based prediction files are:

```text
predicted_ccsd_from_mp2_gbr.txt
predicted_ccsd_from_mp2_svr.txt
predicted_ccsd_from_mp2_gpr.txt
```

Typical DFT-based prediction files are:

```text
predicted_ccsd_from_dft_gbr.txt
predicted_ccsd_from_dft_svr.txt
predicted_ccsd_from_dft_gpr.txt
```

For SOAP workflows, prediction files are written to:

```text
Predictions_SOAP/
```

Typical SOAP prediction files are:

```text
predicted_ccsd_from_mp2_soap_svr.txt
predicted_ccsd_from_dft_soap_svr.txt
```

---

## Candidate selection

The candidate-selection scripts compare the first part of the prediction file against the available held-out CCSD reference energies.

The following quantities are reported:

```text
R2
MSE
MAE
standard error
maximum absolute error
```

The scripts then locate structures whose predicted CCSD energies fall inside an MAE-sized window around the most negative predicted CCSD energy.

Selected structures are copied into:

```text
Found_Within_MAE_Range/
```

for handcrafted descriptors, or

```text
Found_Within_MAE_Range_SOAP/
```

for SOAP descriptors.

The corresponding tabulated results are written as:

```text
Values_Within_Range_*.txt
```

Regression metrics are written as:

```text
all_regression_metrics.txt
all_regression_metrics_soap.txt
```

Plots are written to:

```text
Plots_HeldOut_Set/
Plots_HeldOut_Set_SOAP/
```

or similarly named plot directories inside each dataset folder.

---

## Row-index convention

The training scripts generate predictions for dataset rows

```text
N_TRAIN : N_TOTAL
```

Therefore, prediction line 1 corresponds to dataset row `N_TRAIN`.

For example, if

```text
N_TRAIN = 800
```

then

```text
prediction line 1   -> dataset row 800
prediction line 2   -> dataset row 801
prediction line 100 -> dataset row 899
```

If

```text
N_TRAIN = 1027
```

then

```text
prediction line 1   -> dataset row 1027
prediction line 2   -> dataset row 1028
prediction line 100 -> dataset row 1126
```

This convention is used by the candidate-selection scripts to map predicted energies back to the corresponding `.xyz` structures.

---

## Pretrained models

If pretrained models are present, they are loaded directly.

Typical handcrafted descriptor models are:

```text
pretrained_gbr_delta_ccsd_mp2.pkl
pretrained_svr_delta_ccsd_mp2.pkl
pretrained_gpr_delta_ccsd_mp2.pkl

pretrained_gbr_delta_ccsd_dft.pkl
pretrained_svr_delta_ccsd_dft.pkl
pretrained_gpr_delta_ccsd_dft.pkl
```

Typical SOAP descriptor models are:

```text
pretrained_svr_soap_delta_ccsd_mp2.pkl
pretrained_svr_soap_delta_ccsd_dft.pkl
```

If a pretrained model is missing, the corresponding training script trains a new model and writes the resulting `.pkl` file.

---

## Requirements

The scripts require Python 3 and the following main packages:

```text
numpy
pandas
matplotlib
scikit-learn
joblib
optuna
ase
dscribe
```

A minimal installation can be created with:

```bash
pip install numpy pandas matplotlib scikit-learn joblib optuna ase dscribe
```

For older Python environments, especially Python 3.6, compatible package versions may be required.

---

## Reproducibility notes

The dataset directories contain the files needed to reproduce the reported training and selection workflows.

The scripts use fixed random states for the accepted train/test splits and model training runs. These values are set directly in the individual scripts.

For the handcrafted descriptor workflows, scaling is fitted only on the training block. For SOAP workflows, both scaling and PCA are fitted only on the training block. This avoids information leakage from held-out or screening structures.

The known CCSD block is kept separate from the remaining screening structures. Candidate selection is performed after prediction and follows the row-index convention described above.

---

## Suggested usage

For handcrafted descriptor models:

```bash
python3 1_train_handcrafted_delta_ml_models.py
python3 2_select_candidate_structures.py
```

For SOAP descriptor models:

```bash
python3 1_train_soap_delta_ml_models.py
python3 2_select_candidate_structures.py
```

Run these commands from the dataset directory of the desired system and energy-pair setting.

---

## Citation

If you use this repository, please cite the associated manuscript or publication describing the delta-machine-learning workflow for donor-acceptor complexes.
