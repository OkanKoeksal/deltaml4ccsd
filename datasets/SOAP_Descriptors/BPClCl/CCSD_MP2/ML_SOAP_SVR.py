#!/usr/bin/env python3
"""
Tail_Weighted_SOAP_SVR_ONLY.py

SOAP (DScribe) + optional MP2 feature + train-only scaling + optional train-only PCA
SVR-only training (Optuna) with tail-weighted CCSD MAE objective.
Produces:
- soap_features_BPBBr3.npy (cached SOAP descriptors)
- scaler + PCA models (optional)
- final SVR models per random_state
- predictions for unseen structures per random_state
- summary TSV

Assumptions (same as your working pipeline):
- processed_xyz_files_final.txt contains XYZ paths in the SAME order as MP2_ML_Final.txt / CCSD_ML_Final.txt
- MP2_ML_Final.txt has at least 5557 lines
- CCSD_ML_Final.txt has at least 972 lines
"""

import os
import numpy as np
from multiprocessing import Pool

from ase.io import read
from dscribe.descriptors import SOAP

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import joblib
import optuna

# =====================================================
# Controls
# =====================================================
ADD_MP2_AS_FEATURE = True
RANDOM_STATE_BASE = 1
NUM_RANDOM_STATES = 120

# =====================================================
# Dimensionality reduction (recommended for SVR)
# =====================================================
USE_PCA = True
# Either set an int (e.g. 200) or a float in (0,1) meaning variance to keep (e.g. 0.99)
PCA_N_COMPONENTS = 10
PCA_WHITEN = False
PCA_RANDOM_STATE = 0
SAVE_TRANSFORMS = True
SCALER_PATH = "soap_scaler.pkl"
PCA_MODEL_PATH = "soap_pca.pkl"

# =====================================================
# SOAP controls
# =====================================================
XYZ_LIST_FILE = "processed_xyz_files_final.txt"
SOAP_NPY_FILE = "soap_features_BPBBr3.npy"
SOAP_N_PROCESSES = 4

SOAP_SPECIES = ["B", "P", "Cl"]
SOAP_R_CUT = 5.5
SOAP_N_MAX = 8
SOAP_L_MAX = 6
SOAP_SIGMA = 0.4
SOAP_AVERAGE = "inner"  # fixed-size vector per structure

# =====================================================
# Tail-weighting on *CCSD error*
# =====================================================
USE_TAIL_WEIGHTED_CCSD_LOSS = True
TAIL_Q_CCSD = 0.10
TAIL_FACTOR_CCSD = 10.0
USE_SVR_WEIGHTS = True  # SVR supports sample_weight in sklearn

# =====================================================
# Data sizes (keep as in your workflow)
# =====================================================
N_TOTAL = 5714
N_KNOWN = 1127
N_TRAIN = 1027

# =====================================================
# Optuna SVR search
# =====================================================
SVR_N_TRIALS = 42
SVR_N_JOBS = -1  # parallel optuna trials

# =====================================================
# Helpers
# =====================================================
def append_mp2_feature(X, mp2_vec):
    mp2_vec = np.asarray(mp2_vec, dtype=float).reshape(-1, 1)
    return np.hstack([X, mp2_vec])

def weighted_mae(y_true, y_pred, w):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.average(np.abs(y_true - y_pred), weights=w))

def build_tail_weights(y, q=0.10, factor=5.0):
    """
    Tail weights for most negative values in y (CCSD energies).
    Weights normalized to mean 1.
    """
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y, dtype=float)
    cut = np.quantile(y, q)
    w[y <= cut] *= factor
    w *= (len(w) / w.sum())
    return w

def tail_mae(y_true, y_pred, q=0.10):
    """
    MAE restricted to lowest-q fraction of y_true.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cut = np.quantile(y_true, q)
    mask = (y_true <= cut)
    if mask.sum() == 0:
        return np.nan, 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))), int(mask.sum())

# =====================================================
# SOAP feature building / caching
# =====================================================
def load_xyz_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"XYZ list file not found: {path}")
    with open(path, "r") as f:
        files = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not files:
        raise ValueError(f"No XYZ paths found in {path}")
    return files

def build_soap():
    return SOAP(
        species=SOAP_SPECIES,
        r_cut=SOAP_R_CUT,
        n_max=SOAP_N_MAX,
        l_max=SOAP_L_MAX,
        sigma=SOAP_SIGMA,
        periodic=False,
        rbf="gto",
        average=SOAP_AVERAGE,
    )

def soap_one(xyz_path):
    atoms = read(xyz_path, format="xyz")
    soap = build_soap()
    return soap.create(atoms)  # (soap_dim,)

def get_or_build_soap_features(xyz_files, npy_path, nproc):
    if os.path.exists(npy_path):
        X = np.load(npy_path)
        print(f"Loaded SOAP features from {npy_path}: shape={X.shape}")
        return X

    print("Computing SOAP features...")
    with Pool(processes=nproc) as pool:
        feats_list = pool.map(soap_one, xyz_files)

    X = np.asarray(feats_list, dtype=np.float32)
    np.save(npy_path, X)
    print(f"Saved SOAP features to {npy_path}: shape={X.shape}")
    return X

# =====================================================
# Main
# =====================================================
def main():
    # -----------------------------
    # Build/load SOAP features
    # -----------------------------
    xyz_files = load_xyz_list(XYZ_LIST_FILE)
    if len(xyz_files) < N_TOTAL:
        raise ValueError(f"xyz_files has {len(xyz_files)} entries but need at least {N_TOTAL}.")

    features = get_or_build_soap_features(xyz_files[:N_TOTAL], SOAP_NPY_FILE, SOAP_N_PROCESSES)
    print("SOAP features final shape:", features.shape)

    # -----------------------------
    # Load energies
    # -----------------------------
    mp2_energies = np.loadtxt("MP2_ML_Final.txt")[:N_TOTAL]
    ccsd_known = np.loadtxt("CCSD_ML_Final.txt")[:N_KNOWN]

    # -----------------------------
    # Split blocks (same logic as your current workflow)
    # -----------------------------
    X_train_raw     = features[:N_TRAIN]
    X_remaining_raw = features[N_TRAIN:N_KNOWN]
    X_unseen_raw    = features[N_TRAIN:N_TOTAL]

    ccsd_train     = ccsd_known[:N_TRAIN]
    ccsd_remaining = ccsd_known[N_TRAIN:N_KNOWN]

    mp2_train     = mp2_energies[:N_TRAIN]
    mp2_remaining = mp2_energies[N_TRAIN:N_KNOWN]
    mp2_unseen    = mp2_energies[N_TRAIN:N_TOTAL]

    # Target is delta (CCSD - MP2)
    y_train_delta = ccsd_train - mp2_train

    # Tail weights computed on CCSD (train only)
    w_ccsd_train = build_tail_weights(ccsd_train, q=TAIL_Q_CCSD, factor=TAIL_FACTOR_CCSD) \
        if USE_TAIL_WEIGHTED_CCSD_LOSS else np.ones_like(ccsd_train, dtype=float)
    np.savetxt("ccsd_tail_weights_train.txt", w_ccsd_train, fmt="%.8f")

    # Append MP2 as feature (optional)
    if ADD_MP2_AS_FEATURE:
        X_train_raw2     = append_mp2_feature(X_train_raw, mp2_train)
        X_remaining_raw2 = append_mp2_feature(X_remaining_raw, mp2_remaining)
        X_unseen_raw2    = append_mp2_feature(X_unseen_raw, mp2_unseen)
    else:
        X_train_raw2, X_remaining_raw2, X_unseen_raw2 = X_train_raw, X_remaining_raw, X_unseen_raw

    # -----------------------------
    # Fit scaler on TRAIN ONLY
    # -----------------------------
    scaler = StandardScaler().fit(X_train_raw2)
    X_train_scaled = scaler.transform(X_train_raw2)
    X_remaining_scaled = scaler.transform(X_remaining_raw2)
    X_unseen_scaled = scaler.transform(X_unseen_raw2)

    # -----------------------------
    # Fit PCA on TRAIN ONLY (optional)
    # -----------------------------
    if USE_PCA:
        # allow float variance target
        if isinstance(PCA_N_COMPONENTS, (int, np.integer)):
            if PCA_N_COMPONENTS >= X_train_scaled.shape[1]:
                raise ValueError(
                    f"PCA_N_COMPONENTS={PCA_N_COMPONENTS} must be < n_features={X_train_scaled.shape[1]}"
                )
        pca = PCA(
            n_components=PCA_N_COMPONENTS,
            whiten=PCA_WHITEN,
            random_state=PCA_RANDOM_STATE,
        ).fit(X_train_scaled)

        X_train_final = pca.transform(X_train_scaled)
        X_remaining_final = pca.transform(X_remaining_scaled)
        X_unseen_final = pca.transform(X_unseen_scaled)

        print(f"PCA: final dim = {X_train_final.shape[1]}")
        print(f"PCA explained_variance_ratio_ sum = {float(np.sum(pca.explained_variance_ratio_)):.6f}")
        np.savetxt("pca_explained_variance_ratio.txt", pca.explained_variance_ratio_, fmt="%.10e")

        if SAVE_TRANSFORMS:
            joblib.dump(pca, PCA_MODEL_PATH)
            print(f"Saved PCA to {PCA_MODEL_PATH}")
    else:
        X_train_final = X_train_scaled
        X_remaining_final = X_remaining_scaled
        X_unseen_final = X_unseen_scaled

    if SAVE_TRANSFORMS:
        joblib.dump(scaler, SCALER_PATH)
        print(f"Saved scaler to {SCALER_PATH}")

    # -----------------------------
    # Training runs across random states
    # -----------------------------
    results = []

    for i in range(NUM_RANDOM_STATES):
        random_state = RANDOM_STATE_BASE + i
        print(f"\nProcessing random state: {random_state}")

        # Create val/test split from the remaining known block
        idx_block = np.arange(N_TRAIN, N_KNOWN)
        idx_val, idx_test = train_test_split(idx_block, test_size=0.5, random_state=random_state)

        map_val  = idx_val  - N_TRAIN
        map_test = idx_test - N_TRAIN

        X_val = X_remaining_final[map_val]
        X_test = X_remaining_final[map_test]

        ccsd_val = ccsd_remaining[map_val]
        ccsd_test = ccsd_remaining[map_test]

        mp2_val = mp2_remaining[map_val]
        mp2_test = mp2_remaining[map_test]

        # For evaluation only
        y_test_delta = ccsd_test - mp2_test

        # Tail weights on TEST (for reporting)
        w_ccsd_test = build_tail_weights(ccsd_test, q=TAIL_Q_CCSD, factor=TAIL_FACTOR_CCSD) \
            if USE_TAIL_WEIGHTED_CCSD_LOSS else np.ones_like(ccsd_test, dtype=float)

        # -----------------------------
        # Optuna objective (CV on TRAIN block)
        # -----------------------------
        svr_model_path = f"final_svr_model_{random_state}.pkl"
        best_path = f"best_hyperparameters_svr_{random_state}.txt"

        if os.path.exists(svr_model_path):
            svr_model = joblib.load(svr_model_path)
            print(f"[SVR] Loaded model from {svr_model_path}")
        else:
            def objective_svr(trial):
                C = trial.suggest_float("C", 1e-2, 1e3, log=True)
                epsilon = trial.suggest_float("epsilon", 1e-4, 0.5, log=True)
                gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)

                kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
                scores = []

                for tr_idx, va_idx in kf.split(X_train_final):
                    X_tr, X_va = X_train_final[tr_idx], X_train_final[va_idx]
                    y_tr = y_train_delta[tr_idx]

                    ccsd_va_true = ccsd_train[va_idx]
                    mp2_va_fold  = mp2_train[va_idx]

                    w_tr = w_ccsd_train[tr_idx]
                    w_va = w_ccsd_train[va_idx]

                    model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)

                    if USE_SVR_WEIGHTS:
                        model.fit(X_tr, y_tr, sample_weight=w_tr)
                    else:
                        model.fit(X_tr, y_tr)

                    delta_va_pred = model.predict(X_va)
                    ccsd_va_pred = mp2_va_fold + delta_va_pred

                    score = weighted_mae(ccsd_va_true, ccsd_va_pred, w_va) \
                            if USE_TAIL_WEIGHTED_CCSD_LOSS else mean_absolute_error(ccsd_va_true, ccsd_va_pred)
                    scores.append(score)

                return float(np.mean(scores))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective_svr, n_trials=SVR_N_TRIALS, n_jobs=SVR_N_JOBS)

            best = study.best_params
            print(f"[SVR] Best params: {best}")

            with open(best_path, "w") as f:
                f.write(f"C={best['C']}\n")
                f.write(f"epsilon={best['epsilon']}\n")
                f.write(f"gamma={best['gamma']}\n")

            svr_model = SVR(kernel="rbf", C=best["C"], epsilon=best["epsilon"], gamma=best["gamma"])
            if USE_SVR_WEIGHTS:
                svr_model.fit(X_train_final, y_train_delta, sample_weight=w_ccsd_train)
            else:
                svr_model.fit(X_train_final, y_train_delta)

            joblib.dump(svr_model, svr_model_path)
            print(f"[SVR] Saved model to {svr_model_path}")

        # -----------------------------
        # Evaluate on test split
        # -----------------------------
        delta_test_pred = svr_model.predict(X_test)
        ccsd_test_pred = mp2_test + delta_test_pred

        mae_ccsd = mean_absolute_error(ccsd_test, ccsd_test_pred)
        wmae_ccsd = weighted_mae(ccsd_test, ccsd_test_pred, w_ccsd_test)
        tail_mae_ccsd, tail_n = tail_mae(ccsd_test, ccsd_test_pred, q=TAIL_Q_CCSD)

        print(f"[SVR] Test MAE (CCSD): {mae_ccsd:.12f}")
        print(f"[SVR] Test tail-weighted MAE (CCSD): {wmae_ccsd:.12f}")
        print(f"[SVR] Test tail-only MAE (CCSD): {tail_mae_ccsd:.12f} (N_tail={tail_n})")
        print(f"[SVR] Test MAE (delta): {mean_absolute_error(y_test_delta, delta_test_pred):.12f}")

        # -----------------------------
        # Predict unseen and write predictions
        # -----------------------------
        delta_unseen_pred = svr_model.predict(X_unseen_final)
        ccsd_unseen_pred = mp2_unseen + delta_unseen_pred

        out_pred = f"predicted_SOCASSI_SVR_unseen_optuna_{random_state}.txt"
        with open(out_pred, "w") as f:
            for e in ccsd_unseen_pred:
                f.write(f"{e:.8f}\n")
        print(f"[SVR] Unseen predictions saved: {out_pred}")

        results.append({
            "random_state": random_state,
            "SVR_MAE_CCSD": mae_ccsd,
            "SVR_WMAE_CCSD": wmae_ccsd,
            "SVR_TAIL_MAE_CCSD": tail_mae_ccsd,
            "PCA_DIM": int(X_train_final.shape[1]),
        })

    # -----------------------------
    # Save summary
    # -----------------------------
    df = pd.DataFrame(results)
    df.to_csv("svr_only_predictions_summary.tsv", sep="\t", index=False)
    print("\nSaved: svr_only_predictions_summary.tsv")

    print("\nDone.")
    print(f"SOAP cached file: {SOAP_NPY_FILE}")
    print(f"ADD_MP2_AS_FEATURE: {ADD_MP2_AS_FEATURE}")
    print(f"USE_PCA: {USE_PCA}, PCA_N_COMPONENTS: {PCA_N_COMPONENTS}")
    print(f"Final feature dim: {X_train_final.shape[1]}")


if __name__ == "__main__":
    import pandas as pd  # local import to keep header minimal
    main()