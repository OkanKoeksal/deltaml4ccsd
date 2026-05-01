#!/usr/bin/env python3

import os
import json
import joblib
import numpy as np
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


# =====================================================
# User controls: BPFF / DFT -> CCSD
# =====================================================

SYSTEM_LABEL = "BPFF"
LOW_LEVEL_LABEL = "DFT"

FEATURES_FILE = "processed_features.txt"
XYZ_LIST_FILE = "processed_xyz_files.txt"
LOW_LEVEL_ENERGIES_FILE = "DFT.txt"
CCSD_ENERGIES_FILE = "CCSD.txt"

N_TOTAL = 5493
N_CCSD = 900
N_TRAIN = 800

# Accepted BPFF / DFT model random state.
RANDOM_STATE = 34

# BPFF original DFT workflow did not append DFT as an additional descriptor.
ADD_LOW_LEVEL_AS_FEATURE = False

# BPFF original workflow optimized delta MAE directly.
USE_TAIL_WEIGHTED_CCSD_LOSS = False
USE_WEIGHTS_IN_FIT = False
USE_SVR_WEIGHTS = False

# Optuna trials are used only if pretrained models are absent.
GBR_TRIALS = 96
SVR_TRIALS = 96
GPR_TRIALS = 96

# Clean pretrained/publication model names.
GBR_MODEL_FILE = "pretrained_gbr_delta_ccsd_dft.pkl"
SVR_MODEL_FILE = "pretrained_svr_delta_ccsd_dft.pkl"
GPR_MODEL_FILE = "pretrained_gpr_delta_ccsd_dft.pkl"

# Optional legacy names from previous BPFF / DFT runs.
LEGACY_GBR_MODEL_FILE = f"final_gb_model_{RANDOM_STATE}.pkl"
LEGACY_SVR_MODEL_FILE = f"final_svr_pipeline_{RANDOM_STATE}.pkl"
LEGACY_GPR_MODEL_FILE = f"final_gpr_pipeline_{RANDOM_STATE}.pkl"

RESULTS_DIR = "Predictions"

PRED_GBR_FILE = os.path.join(RESULTS_DIR, "predicted_ccsd_from_dft_gbr.txt")
PRED_SVR_FILE = os.path.join(RESULTS_DIR, "predicted_ccsd_from_dft_svr.txt")
PRED_GPR_FILE = os.path.join(RESULTS_DIR, "predicted_ccsd_from_dft_gpr.txt")

SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary_metrics.tsv")


# =====================================================
# Helpers
# =====================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def append_low_level_feature(X, low_level_vector):
    low_level_vector = np.asarray(low_level_vector, dtype=float).reshape(-1, 1)
    return np.hstack([X, low_level_vector])


def load_clean_or_legacy_model(clean_path, legacy_path, label):
    if os.path.exists(clean_path):
        print(f"[{label}] Loading pretrained model: {clean_path}")
        return joblib.load(clean_path)

    if legacy_path and os.path.exists(legacy_path):
        print(f"[{label}] Loading legacy model: {legacy_path}")
        model = joblib.load(legacy_path)
        joblib.dump(model, clean_path)
        print(f"[{label}] Also saved clean pretrained name: {clean_path}")
        return model

    return None


def load_or_train_svr(X_train_raw, y_train_delta):
    model = load_clean_or_legacy_model(
        clean_path=SVR_MODEL_FILE,
        legacy_path=LEGACY_SVR_MODEL_FILE,
        label="SVR",
    )

    if model is not None:
        return model

    print("[SVR] No pretrained model found. Training new model.")

    def objective_svr(trial):
        C = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 0.5, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)),
        ])

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        maes = []

        for tr_idx, va_idx in kf.split(X_train_raw):
            X_tr = X_train_raw[tr_idx]
            X_va = X_train_raw[va_idx]
            y_tr = y_train_delta[tr_idx]
            y_va = y_train_delta[va_idx]

            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)

            maes.append(mean_absolute_error(y_va, pred))

        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_svr, n_trials=SVR_TRIALS, n_jobs=-1)

    best = study.best_params
    print("[SVR] Best params:", best)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=best["C"],
            epsilon=best["epsilon"],
            gamma=best["gamma"],
        )),
    ])

    pipe.fit(X_train_raw, y_train_delta)

    joblib.dump(pipe, SVR_MODEL_FILE)
    print("[SVR] Saved:", SVR_MODEL_FILE)

    return pipe


def load_or_train_gpr(X_train_raw, y_train_delta):
    model = load_clean_or_legacy_model(
        clean_path=GPR_MODEL_FILE,
        legacy_path=LEGACY_GPR_MODEL_FILE,
        label="GPR",
    )

    if model is not None:
        return model

    print("[GPR] No pretrained model found. Training new model.")

    def objective_gpr(trial):
        length_scale = trial.suggest_float("length_scale", 1e-2, 1e3, log=True)
        alpha = trial.suggest_float("alpha", 1e-8, 1e-1, log=True)

        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=alpha)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=False,
                random_state=RANDOM_STATE,
            )),
        ])

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        maes = []

        for tr_idx, va_idx in kf.split(X_train_raw):
            X_tr = X_train_raw[tr_idx]
            X_va = X_train_raw[va_idx]
            y_tr = y_train_delta[tr_idx]
            y_va = y_train_delta[va_idx]

            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)

            maes.append(mean_absolute_error(y_va, pred))

        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_gpr, n_trials=GPR_TRIALS, n_jobs=-1)

    best = study.best_params
    print("[GPR] Best params:", best)

    kernel = RBF(length_scale=best["length_scale"]) + WhiteKernel(
        noise_level=best["alpha"]
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=kernel,
            alpha=best["alpha"],
            normalize_y=False,
            random_state=RANDOM_STATE,
        )),
    ])

    pipe.fit(X_train_raw, y_train_delta)

    joblib.dump(pipe, GPR_MODEL_FILE)
    print("[GPR] Saved:", GPR_MODEL_FILE)

    return pipe


def load_or_train_gbr(X_train_scaled, y_train_delta):
    model = load_clean_or_legacy_model(
        clean_path=GBR_MODEL_FILE,
        legacy_path=LEGACY_GBR_MODEL_FILE,
        label="GBR",
    )

    if model is not None:
        return model

    print("[GBR] No pretrained model found. Training new model.")

    def objective_gbr(trial):
        n_estimators = trial.suggest_int("n_estimators", 1, 10000)
        max_depth = trial.suggest_int("max_depth", 1, 20)

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        maes = []

        for tr_idx, va_idx in kf.split(X_train_scaled):
            X_tr = X_train_scaled[tr_idx]
            X_va = X_train_scaled[va_idx]
            y_tr = y_train_delta[tr_idx]
            y_va = y_train_delta[va_idx]

            gb = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=RANDOM_STATE,
            )

            gb.fit(X_tr, y_tr)
            pred = gb.predict(X_va)

            maes.append(mean_absolute_error(y_va, pred))

        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_gbr, n_trials=GBR_TRIALS, n_jobs=-1)

    best = study.best_params
    print("[GBR] Best params:", best)

    model = GradientBoostingRegressor(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train_delta)

    joblib.dump(model, GBR_MODEL_FILE)
    print("[GBR] Saved:", GBR_MODEL_FILE)

    return model


def evaluate_model(name, model, X_test, ccsd_test, low_test):
    delta_pred = model.predict(X_test)
    ccsd_pred = low_test + delta_pred

    mae_delta = mean_absolute_error(ccsd_test - low_test, delta_pred)
    mae_ccsd = mean_absolute_error(ccsd_test, ccsd_pred)

    print("[{}] Test MAE delta: {:.12f}".format(name, mae_delta))
    print("[{}] Test MAE CCSD:  {:.12f}".format(name, mae_ccsd))

    return {
        "model": name,
        "mae_delta": mae_delta,
        "mae_ccsd": mae_ccsd,
    }


# =====================================================
# Main
# =====================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    require_file(FEATURES_FILE)
    require_file(XYZ_LIST_FILE)
    require_file(LOW_LEVEL_ENERGIES_FILE)
    require_file(CCSD_ENERGIES_FILE)

    print("Using files:")
    print("  Features:          ", FEATURES_FILE)
    print("  XYZ list:          ", XYZ_LIST_FILE)
    print("  Low-level energies:", LOW_LEVEL_ENERGIES_FILE)
    print("  CCSD energies:     ", CCSD_ENERGIES_FILE)

    features = np.loadtxt(FEATURES_FILE)[:N_TOTAL]
    low_level_energies = np.loadtxt(LOW_LEVEL_ENERGIES_FILE)[:N_TOTAL]
    ccsd_known = np.loadtxt(CCSD_ENERGIES_FILE)[:N_CCSD]

    with open(XYZ_LIST_FILE, "r") as f:
        xyz_files = [line.strip() for line in f if line.strip()]

    if len(xyz_files) < N_TOTAL:
        raise ValueError(
            f"XYZ list has {len(xyz_files)} entries, but N_TOTAL={N_TOTAL}."
        )

    print("Feature matrix shape:", features.shape)
    print("Low-level energies:", low_level_energies.shape)
    print("CCSD known energies:", ccsd_known.shape)
    print("XYZ files:", len(xyz_files))

    X_train_raw = features[:N_TRAIN]
    X_remaining_raw = features[N_TRAIN:N_CCSD]
    X_unseen_raw = features[N_TRAIN:N_TOTAL]

    ccsd_train = ccsd_known[:N_TRAIN]
    ccsd_remaining = ccsd_known[N_TRAIN:N_CCSD]

    low_train = low_level_energies[:N_TRAIN]
    low_remaining = low_level_energies[N_TRAIN:N_CCSD]
    low_unseen = low_level_energies[N_TRAIN:N_TOTAL]

    if ADD_LOW_LEVEL_AS_FEATURE:
        X_train_raw2 = append_low_level_feature(X_train_raw, low_train)
        X_remaining_raw2 = append_low_level_feature(X_remaining_raw, low_remaining)
        X_unseen_raw2 = append_low_level_feature(X_unseen_raw, low_unseen)
    else:
        X_train_raw2 = X_train_raw
        X_remaining_raw2 = X_remaining_raw
        X_unseen_raw2 = X_unseen_raw

    # GBR uses externally scaled features, following the original BPFF workflow.
    scaler_for_gbr = StandardScaler().fit(X_train_raw2)
    X_train_scaled = scaler_for_gbr.transform(X_train_raw2)
    X_remaining_scaled = scaler_for_gbr.transform(X_remaining_raw2)
    X_unseen_scaled = scaler_for_gbr.transform(X_unseen_raw2)

    y_train_delta = ccsd_train - low_train

    idx_block = np.arange(N_TRAIN, N_CCSD)

    idx_val, idx_test = train_test_split(
        idx_block,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    map_test = idx_test - N_TRAIN

    X_test_pipe = X_remaining_raw2[map_test]
    X_test_gbr = X_remaining_scaled[map_test]

    ccsd_test = ccsd_remaining[map_test]
    low_test = low_remaining[map_test]

    np.savetxt(os.path.join(RESULTS_DIR, "val_indices.txt"), idx_val, fmt="%d")
    np.savetxt(os.path.join(RESULTS_DIR, "test_indices.txt"), idx_test, fmt="%d")

    svr_model = load_or_train_svr(X_train_raw2, y_train_delta)
    gpr_model = load_or_train_gpr(X_train_raw2, y_train_delta)
    gbr_model = load_or_train_gbr(X_train_scaled, y_train_delta)

    metrics = []

    metrics.append(
        evaluate_model("SVR", svr_model, X_test_pipe, ccsd_test, low_test)
    )

    metrics.append(
        evaluate_model("GPR", gpr_model, X_test_pipe, ccsd_test, low_test)
    )

    metrics.append(
        evaluate_model("GBR", gbr_model, X_test_gbr, ccsd_test, low_test)
    )

    with open(SUMMARY_FILE, "w") as f:
        f.write("model\tmae_delta\tmae_ccsd\n")

        for row in metrics:
            f.write(
                "{model}\t{mae_delta:.10f}\t{mae_ccsd:.10f}\n".format(**row)
            )

    print("Wrote summary:", SUMMARY_FILE)

    pred_svr = low_unseen + svr_model.predict(X_unseen_raw2)
    pred_gpr = low_unseen + gpr_model.predict(X_unseen_raw2)
    pred_gbr = low_unseen + gbr_model.predict(X_unseen_scaled)

    np.savetxt(PRED_SVR_FILE, pred_svr, fmt="%.10f")
    np.savetxt(PRED_GPR_FILE, pred_gpr, fmt="%.10f")
    np.savetxt(PRED_GBR_FILE, pred_gbr, fmt="%.10f")

    print("Wrote predictions:")
    print("  ", PRED_SVR_FILE)
    print("  ", PRED_GPR_FILE)
    print("  ", PRED_GBR_FILE)

    run_info = {
        "system_label": SYSTEM_LABEL,
        "low_level_label": LOW_LEVEL_LABEL,
        "features_file": FEATURES_FILE,
        "xyz_list_file": XYZ_LIST_FILE,
        "low_level_energies_file": LOW_LEVEL_ENERGIES_FILE,
        "ccsd_energies_file": CCSD_ENERGIES_FILE,
        "n_total": N_TOTAL,
        "n_ccsd": N_CCSD,
        "n_train": N_TRAIN,
        "random_state": RANDOM_STATE,
        "add_low_level_as_feature": ADD_LOW_LEVEL_AS_FEATURE,
        "gbr_model_file": GBR_MODEL_FILE,
        "svr_model_file": SVR_MODEL_FILE,
        "gpr_model_file": GPR_MODEL_FILE,
    }

    with open(os.path.join(RESULTS_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()