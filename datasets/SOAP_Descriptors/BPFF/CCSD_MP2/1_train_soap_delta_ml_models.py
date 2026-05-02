#!/usr/bin/env python3

import os
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import joblib
import optuna

from ase.io import read
from dscribe.descriptors import SOAP

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


# =====================================================
# User controls: BPFF / SOAP / MP2 -> CCSD
# =====================================================

SYSTEM_LABEL = "BPFF"
DESCRIPTOR_LABEL = "SOAP"
LOW_LEVEL_LABEL = "MP2"

XYZ_LIST_FILE = "processed_xyz_files.dat"
LOW_LEVEL_ENERGIES_FILE = "MP2.dat"
CCSD_ENERGIES_FILE = "CCSD.dat"

N_TOTAL = 5497
N_KNOWN = 900
N_TRAIN = 800

RANDOM_STATE = 36

SOAP_NPY_FILE = "soap_features.npy"
SOAP_N_PROCESSES = 4

SOAP_SPECIES = ["B", "P", "F"]
SOAP_R_CUT = 5.5
SOAP_N_MAX = 8
SOAP_L_MAX = 6
SOAP_SIGMA = 0.4
SOAP_AVERAGE = "inner"

ADD_LOW_LEVEL_AS_FEATURE = True

USE_PCA = True
PCA_N_COMPONENTS = 10
PCA_WHITEN = False
PCA_RANDOM_STATE = 0

SCALER_FILE = "soap_scaler.pkl"
PCA_FILE = "soap_pca.pkl"

USE_TAIL_WEIGHTED_CCSD_LOSS = True
TAIL_Q_CCSD = 0.10
TAIL_FACTOR_CCSD = 10.0
USE_SVR_WEIGHTS = True

SVR_TRIALS = 96
SVR_N_JOBS = -1

SVR_MODEL_FILE = "pretrained_svr_soap_delta_ccsd_mp2.pkl"

RESULTS_DIR = "Predictions_SOAP"

PRED_SVR_FILE = os.path.join(
    RESULTS_DIR,
    "predicted_ccsd_from_mp2_soap_svr.txt",
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "summary_metrics_soap_svr.tsv",
)


# =====================================================
# Helpers
# =====================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_xyz_list(path):
    with open(path, "r") as f:
        xyz_files = [line.strip() for line in f if line.strip()]

    if not xyz_files:
        raise ValueError(f"No xyz paths found in {path}")

    return xyz_files


def append_low_level_feature(X, low_level_vector):
    low_level_vector = np.asarray(low_level_vector, dtype=float).reshape(-1, 1)
    return np.hstack([X, low_level_vector])


def weighted_mae(y_true, y_pred, weights):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def build_tail_weights(y, q=0.10, factor=10.0):
    y = np.asarray(y, dtype=float)
    weights = np.ones_like(y, dtype=float)

    cutoff = np.quantile(y, q)
    weights[y <= cutoff] *= factor

    weights *= len(weights) / weights.sum()
    return weights


def tail_mae(y_true, y_pred, q=0.10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    cutoff = np.quantile(y_true, q)
    mask = y_true <= cutoff

    if mask.sum() == 0:
        return np.nan, 0

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))), int(mask.sum())


# =====================================================
# SOAP feature generation
# =====================================================

def build_soap_descriptor():
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
    soap = build_soap_descriptor()
    return soap.create(atoms)


def get_or_build_soap_features(xyz_files, npy_path, n_processes):
    if os.path.exists(npy_path):
        X = np.load(npy_path)
        print(f"Loaded SOAP features from {npy_path} with shape {X.shape}")
        return X

    print("Computing SOAP features from xyz list...")
    print("Number of structures:", len(xyz_files))

    with Pool(processes=n_processes) as pool:
        features_list = pool.map(soap_one, xyz_files)

    X = np.asarray(features_list, dtype=np.float32)
    np.save(npy_path, X)

    print(f"Saved SOAP features to {npy_path} with shape {X.shape}")
    return X


# =====================================================
# Model training/loading
# =====================================================

def load_or_train_svr(
    X_train_final,
    y_train_delta,
    ccsd_train,
    low_train,
    weights_train,
):
    if os.path.exists(SVR_MODEL_FILE):
        print("[SOAP-SVR] Loading pretrained model:", SVR_MODEL_FILE)
        return joblib.load(SVR_MODEL_FILE)

    print("[SOAP-SVR] No pretrained model found. Training new model.")

    def objective_svr(trial):
        C = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-4, 0.5, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for tr_idx, va_idx in kf.split(X_train_final):
            X_tr = X_train_final[tr_idx]
            X_va = X_train_final[va_idx]

            y_tr = y_train_delta[tr_idx]

            ccsd_va_true = ccsd_train[va_idx]
            low_va = low_train[va_idx]

            w_tr = weights_train[tr_idx]
            w_va = weights_train[va_idx]

            model = SVR(
                kernel="rbf",
                C=C,
                epsilon=epsilon,
                gamma=gamma,
            )

            if USE_SVR_WEIGHTS:
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr)

            delta_va_pred = model.predict(X_va)
            ccsd_va_pred = low_va + delta_va_pred

            if USE_TAIL_WEIGHTED_CCSD_LOSS:
                score = weighted_mae(ccsd_va_true, ccsd_va_pred, w_va)
            else:
                score = mean_absolute_error(ccsd_va_true, ccsd_va_pred)

            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_svr, n_trials=SVR_TRIALS, n_jobs=SVR_N_JOBS)

    best = study.best_params
    print("[SOAP-SVR] Best params:", best)

    with open(os.path.join(RESULTS_DIR, "best_hyperparameters_soap_svr.txt"), "w") as f:
        f.write(f"C={best['C']}\n")
        f.write(f"epsilon={best['epsilon']}\n")
        f.write(f"gamma={best['gamma']}\n")

    model = SVR(
        kernel="rbf",
        C=best["C"],
        epsilon=best["epsilon"],
        gamma=best["gamma"],
    )

    if USE_SVR_WEIGHTS:
        model.fit(X_train_final, y_train_delta, sample_weight=weights_train)
    else:
        model.fit(X_train_final, y_train_delta)

    joblib.dump(model, SVR_MODEL_FILE)
    print("[SOAP-SVR] Saved model:", SVR_MODEL_FILE)

    return model


def evaluate_model(model, X_test, ccsd_test, low_test):
    delta_pred = model.predict(X_test)
    ccsd_pred = low_test + delta_pred

    mae_delta = mean_absolute_error(ccsd_test - low_test, delta_pred)
    mae_ccsd = mean_absolute_error(ccsd_test, ccsd_pred)

    weights_test = build_tail_weights(
        ccsd_test,
        q=TAIL_Q_CCSD,
        factor=TAIL_FACTOR_CCSD,
    )

    weighted_mae_ccsd = weighted_mae(ccsd_test, ccsd_pred, weights_test)
    tail_mae_ccsd, n_tail = tail_mae(ccsd_test, ccsd_pred, q=TAIL_Q_CCSD)

    print("[SOAP-SVR] Test MAE delta: {:.12f}".format(mae_delta))
    print("[SOAP-SVR] Test MAE CCSD:  {:.12f}".format(mae_ccsd))
    print("[SOAP-SVR] Test weighted MAE CCSD: {:.12f}".format(weighted_mae_ccsd))
    print("[SOAP-SVR] Test tail MAE CCSD: {:.12f}  N_tail={}".format(tail_mae_ccsd, n_tail))

    return {
        "model": "SOAP-SVR",
        "mae_delta": mae_delta,
        "mae_ccsd": mae_ccsd,
        "weighted_mae_ccsd": weighted_mae_ccsd,
        "tail_mae_ccsd": tail_mae_ccsd,
        "n_tail": n_tail,
    }


# =====================================================
# Main
# =====================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    require_file(XYZ_LIST_FILE)
    require_file(LOW_LEVEL_ENERGIES_FILE)
    require_file(CCSD_ENERGIES_FILE)

    print("Using files:")
    print("  XYZ list:          ", XYZ_LIST_FILE)
    print("  Low-level energies:", LOW_LEVEL_ENERGIES_FILE)
    print("  CCSD energies:     ", CCSD_ENERGIES_FILE)
    print("  SOAP cache:        ", SOAP_NPY_FILE)

    xyz_files = load_xyz_list(XYZ_LIST_FILE)

    if len(xyz_files) < N_TOTAL:
        raise ValueError(
            f"XYZ list has {len(xyz_files)} entries, but N_TOTAL={N_TOTAL}."
        )

    xyz_files = xyz_files[:N_TOTAL]

    features = get_or_build_soap_features(
        xyz_files=xyz_files,
        npy_path=SOAP_NPY_FILE,
        n_processes=SOAP_N_PROCESSES,
    )

    if features.shape[0] < N_TOTAL:
        raise ValueError(
            f"SOAP feature matrix has {features.shape[0]} rows, but N_TOTAL={N_TOTAL}."
        )

    features = features[:N_TOTAL]

    low_level_energies = np.loadtxt(LOW_LEVEL_ENERGIES_FILE)[:N_TOTAL]
    ccsd_known = np.loadtxt(CCSD_ENERGIES_FILE)[:N_KNOWN]

    print("SOAP feature matrix:", features.shape)
    print("Low-level energies:", low_level_energies.shape)
    print("CCSD known energies:", ccsd_known.shape)

    X_train_raw = features[:N_TRAIN]
    X_remaining_raw = features[N_TRAIN:N_KNOWN]
    X_unseen_raw = features[N_TRAIN:N_TOTAL]

    ccsd_train = ccsd_known[:N_TRAIN]
    ccsd_remaining = ccsd_known[N_TRAIN:N_KNOWN]

    low_train = low_level_energies[:N_TRAIN]
    low_remaining = low_level_energies[N_TRAIN:N_KNOWN]
    low_unseen = low_level_energies[N_TRAIN:N_TOTAL]

    y_train_delta = ccsd_train - low_train

    if USE_TAIL_WEIGHTED_CCSD_LOSS:
        weights_train = build_tail_weights(
            ccsd_train,
            q=TAIL_Q_CCSD,
            factor=TAIL_FACTOR_CCSD,
        )
    else:
        weights_train = np.ones_like(ccsd_train, dtype=float)

    np.savetxt(
        os.path.join(RESULTS_DIR, "ccsd_tail_weights_train_soap.txt"),
        weights_train,
        fmt="%.8f",
    )

    if ADD_LOW_LEVEL_AS_FEATURE:
        X_train_raw2 = append_low_level_feature(X_train_raw, low_train)
        X_remaining_raw2 = append_low_level_feature(X_remaining_raw, low_remaining)
        X_unseen_raw2 = append_low_level_feature(X_unseen_raw, low_unseen)
    else:
        X_train_raw2 = X_train_raw
        X_remaining_raw2 = X_remaining_raw
        X_unseen_raw2 = X_unseen_raw

    scaler = StandardScaler().fit(X_train_raw2)

    X_train_scaled = scaler.transform(X_train_raw2)
    X_remaining_scaled = scaler.transform(X_remaining_raw2)
    X_unseen_scaled = scaler.transform(X_unseen_raw2)

    joblib.dump(scaler, SCALER_FILE)
    print("Saved SOAP scaler:", SCALER_FILE)

    if USE_PCA:
        if isinstance(PCA_N_COMPONENTS, int):
            if PCA_N_COMPONENTS >= X_train_scaled.shape[1]:
                raise ValueError(
                    "PCA_N_COMPONENTS={} must be smaller than n_features={}.".format(
                        PCA_N_COMPONENTS,
                        X_train_scaled.shape[1],
                    )
                )

        pca = PCA(
            n_components=PCA_N_COMPONENTS,
            whiten=PCA_WHITEN,
            random_state=PCA_RANDOM_STATE,
        )

        pca.fit(X_train_scaled)

        X_train_final = pca.transform(X_train_scaled)
        X_remaining_final = pca.transform(X_remaining_scaled)
        X_unseen_final = pca.transform(X_unseen_scaled)

        joblib.dump(pca, PCA_FILE)

        np.savetxt(
            os.path.join(RESULTS_DIR, "pca_explained_variance_ratio_soap.txt"),
            pca.explained_variance_ratio_,
            fmt="%.10e",
        )

        print("Saved SOAP PCA:", PCA_FILE)
        print("PCA final dimension:", X_train_final.shape[1])
        print("PCA explained variance sum:", float(np.sum(pca.explained_variance_ratio_)))
    else:
        X_train_final = X_train_scaled
        X_remaining_final = X_remaining_scaled
        X_unseen_final = X_unseen_scaled

    idx_block = np.arange(N_TRAIN, N_KNOWN)

    idx_val, idx_test = train_test_split(
        idx_block,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    map_test = idx_test - N_TRAIN

    X_test = X_remaining_final[map_test]
    ccsd_test = ccsd_remaining[map_test]
    low_test = low_remaining[map_test]

    np.savetxt(os.path.join(RESULTS_DIR, "val_indices_soap.txt"), idx_val, fmt="%d")
    np.savetxt(os.path.join(RESULTS_DIR, "test_indices_soap.txt"), idx_test, fmt="%d")

    svr_model = load_or_train_svr(
        X_train_final=X_train_final,
        y_train_delta=y_train_delta,
        ccsd_train=ccsd_train,
        low_train=low_train,
        weights_train=weights_train,
    )

    metrics = evaluate_model(
        model=svr_model,
        X_test=X_test,
        ccsd_test=ccsd_test,
        low_test=low_test,
    )

    pd.DataFrame([metrics]).to_csv(
        SUMMARY_FILE,
        sep="\t",
        index=False,
    )

    print("Wrote summary:", SUMMARY_FILE)

    delta_unseen_pred = svr_model.predict(X_unseen_final)
    ccsd_unseen_pred = low_unseen + delta_unseen_pred

    np.savetxt(PRED_SVR_FILE, ccsd_unseen_pred, fmt="%.10f")

    print("Wrote predictions:")
    print("  ", PRED_SVR_FILE)

    run_info = {
        "system_label": SYSTEM_LABEL,
        "descriptor_label": DESCRIPTOR_LABEL,
        "low_level_label": LOW_LEVEL_LABEL,
        "xyz_list_file": XYZ_LIST_FILE,
        "low_level_energies_file": LOW_LEVEL_ENERGIES_FILE,
        "ccsd_energies_file": CCSD_ENERGIES_FILE,
        "soap_npy_file": SOAP_NPY_FILE,
        "n_total": N_TOTAL,
        "n_known": N_KNOWN,
        "n_train": N_TRAIN,
        "random_state": RANDOM_STATE,
        "add_low_level_as_feature": ADD_LOW_LEVEL_AS_FEATURE,
        "use_pca": USE_PCA,
        "pca_n_components": PCA_N_COMPONENTS,
        "soap_species": SOAP_SPECIES,
        "soap_r_cut": SOAP_R_CUT,
        "soap_n_max": SOAP_N_MAX,
        "soap_l_max": SOAP_L_MAX,
        "soap_sigma": SOAP_SIGMA,
        "soap_average": SOAP_AVERAGE,
        "tail_weighted_ccsd_loss": USE_TAIL_WEIGHTED_CCSD_LOSS,
        "tail_q_ccsd": TAIL_Q_CCSD,
        "tail_factor_ccsd": TAIL_FACTOR_CCSD,
        "svr_model_file": SVR_MODEL_FILE,
        "scaler_file": SCALER_FILE,
        "pca_file": PCA_FILE,
    }

    with open(os.path.join(RESULTS_DIR, "run_info_soap.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()