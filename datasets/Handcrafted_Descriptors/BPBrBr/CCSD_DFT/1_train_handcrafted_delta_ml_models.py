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
# User controls
# =====================================================

LOW_LEVEL_LABEL = "DFT"

FEATURES_FILE = "processed_features.txt"
XYZ_LIST_FILE = "processed_xyz_files.txt"
LOW_LEVEL_ENERGIES_FILE = "DFT.txt"
CCSD_ENERGIES_FILE = "CCSD.txt"

N_TOTAL = 5556
N_CCSD = 972
N_TRAIN = 800

# Accepted-paper model random state for this DFT counterpart.
RANDOM_STATE = 92

ADD_LOW_LEVEL_AS_FEATURE = True

USE_TAIL_WEIGHTED_CCSD_LOSS = True
TAIL_Q_CCSD = 0.10
TAIL_FACTOR_CCSD = 10.0
USE_WEIGHTS_IN_FIT = True
USE_SVR_WEIGHTS = True

# Optuna trials used only if models do not already exist.
GBR_TRIALS = 96
SVR_TRIALS = 96
GPR_TRIALS = 96

# Clean publication model names.
GBR_MODEL_FILE = "final_gbr_delta_ccsd_dft.pkl"
SVR_MODEL_FILE = "final_svr_delta_ccsd_dft.pkl"
GPR_MODEL_FILE = "final_gpr_delta_ccsd_dft.pkl"

# Optional legacy model names from previous runs.
LEGACY_GBR_MODEL_FILE = "pretrained_gbr_delta_ccsd_dft.pkl"
LEGACY_SVR_MODEL_FILE = "pretrained_svr_delta_ccsd_dft.pkl"
LEGACY_GPR_MODEL_FILE = "pretrained_gpr_delta_ccsd_dft.pkl"

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


def load_or_train_gbr(X_train_scaled, y_train_delta, ccsd_train, low_train, weights_train):
    if os.path.exists(GBR_MODEL_FILE):
        print("[GBR] Loading existing model:", GBR_MODEL_FILE)
        return joblib.load(GBR_MODEL_FILE)

    if os.path.exists(LEGACY_GBR_MODEL_FILE):
        print("[GBR] Loading legacy model:", LEGACY_GBR_MODEL_FILE)
        model = joblib.load(LEGACY_GBR_MODEL_FILE)
        joblib.dump(model, GBR_MODEL_FILE)
        print("[GBR] Also saved model name:", GBR_MODEL_FILE)
        return model

    print("[GBR] No existing model found. Training new model.")

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 8000)
        max_depth = trial.suggest_int("max_depth", 1, 10)

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for tr_idx, va_idx in kf.split(X_train_scaled):
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=RANDOM_STATE,
            )

            if USE_WEIGHTS_IN_FIT:
                model.fit(
                    X_train_scaled[tr_idx],
                    y_train_delta[tr_idx],
                    sample_weight=weights_train[tr_idx],
                )
            else:
                model.fit(X_train_scaled[tr_idx], y_train_delta[tr_idx])

            delta_pred = model.predict(X_train_scaled[va_idx])
            ccsd_pred = low_train[va_idx] + delta_pred

            if USE_TAIL_WEIGHTED_CCSD_LOSS:
                score = weighted_mae(
                    ccsd_train[va_idx],
                    ccsd_pred,
                    weights_train[va_idx],
                )
            else:
                score = mean_absolute_error(ccsd_train[va_idx], ccsd_pred)

            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=GBR_TRIALS, n_jobs=-1)

    best = study.best_params
    print("[GBR] Best params:", best)

    model = GradientBoostingRegressor(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        random_state=RANDOM_STATE,
    )

    if USE_WEIGHTS_IN_FIT:
        model.fit(X_train_scaled, y_train_delta, sample_weight=weights_train)
    else:
        model.fit(X_train_scaled, y_train_delta)

    joblib.dump(model, GBR_MODEL_FILE)
    print("[GBR] Saved:", GBR_MODEL_FILE)

    return model


def load_or_train_svr(X_train_raw, y_train_delta, ccsd_train, low_train, weights_train):
    if os.path.exists(SVR_MODEL_FILE):
        print("[SVR] Loading existing model:", SVR_MODEL_FILE)
        return joblib.load(SVR_MODEL_FILE)

    if os.path.exists(LEGACY_SVR_MODEL_FILE):
        print("[SVR] Loading legacy model:", LEGACY_SVR_MODEL_FILE)
        model = joblib.load(LEGACY_SVR_MODEL_FILE)
        joblib.dump(model, SVR_MODEL_FILE)
        print("[SVR] Also saved model name:", SVR_MODEL_FILE)
        return model

    print("[SVR] No existing model found. Training new model.")

    def objective(trial):
        C = trial.suggest_float("C", 1e-2, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-4, 0.5, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)),
        ])

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for tr_idx, va_idx in kf.split(X_train_raw):
            if USE_SVR_WEIGHTS and USE_WEIGHTS_IN_FIT:
                pipe.fit(
                    X_train_raw[tr_idx],
                    y_train_delta[tr_idx],
                    model__sample_weight=weights_train[tr_idx],
                )
            else:
                pipe.fit(X_train_raw[tr_idx], y_train_delta[tr_idx])

            delta_pred = pipe.predict(X_train_raw[va_idx])
            ccsd_pred = low_train[va_idx] + delta_pred

            if USE_TAIL_WEIGHTED_CCSD_LOSS:
                score = weighted_mae(
                    ccsd_train[va_idx],
                    ccsd_pred,
                    weights_train[va_idx],
                )
            else:
                score = mean_absolute_error(ccsd_train[va_idx], ccsd_pred)

            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=SVR_TRIALS, n_jobs=-1)

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

    if USE_SVR_WEIGHTS and USE_WEIGHTS_IN_FIT:
        pipe.fit(X_train_raw, y_train_delta, model__sample_weight=weights_train)
    else:
        pipe.fit(X_train_raw, y_train_delta)

    joblib.dump(pipe, SVR_MODEL_FILE)
    print("[SVR] Saved:", SVR_MODEL_FILE)

    return pipe


def load_or_train_gpr(X_train_raw, y_train_delta, ccsd_train, low_train, weights_train):
    if os.path.exists(GPR_MODEL_FILE):
        print("[GPR] Loading existing model:", GPR_MODEL_FILE)
        return joblib.load(GPR_MODEL_FILE)

    if os.path.exists(LEGACY_GPR_MODEL_FILE):
        print("[GPR] Loading legacy model:", LEGACY_GPR_MODEL_FILE)
        model = joblib.load(LEGACY_GPR_MODEL_FILE)
        joblib.dump(model, GPR_MODEL_FILE)
        print("[GPR] Also saved model name:", GPR_MODEL_FILE)
        return model

    print("[GPR] No existing model found. Training new model.")

    def objective(trial):
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
        scores = []

        for tr_idx, va_idx in kf.split(X_train_raw):
            pipe.fit(X_train_raw[tr_idx], y_train_delta[tr_idx])

            delta_pred = pipe.predict(X_train_raw[va_idx])
            ccsd_pred = low_train[va_idx] + delta_pred

            if USE_TAIL_WEIGHTED_CCSD_LOSS:
                score = weighted_mae(
                    ccsd_train[va_idx],
                    ccsd_pred,
                    weights_train[va_idx],
                )
            else:
                score = mean_absolute_error(ccsd_train[va_idx], ccsd_pred)

            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=GPR_TRIALS, n_jobs=-1)

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


def evaluate_model(name, model, X_test, ccsd_test, low_test):
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

    print("[{}] Test MAE delta: {:.12f}".format(name, mae_delta))
    print("[{}] Test MAE CCSD:  {:.12f}".format(name, mae_ccsd))
    print("[{}] Test weighted MAE CCSD: {:.12f}".format(name, weighted_mae_ccsd))
    print("[{}] Test tail MAE CCSD: {:.12f}  N_tail={}".format(name, tail_mae_ccsd, n_tail))

    return {
        "model": name,
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

    features_file = require_file(FEATURES_FILE)
    xyz_list_file = require_file(XYZ_LIST_FILE)
    low_level_file = require_file(LOW_LEVEL_ENERGIES_FILE)
    ccsd_file = require_file(CCSD_ENERGIES_FILE)

    print("Using files:")
    print("  Features:          ", features_file)
    print("  XYZ list:          ", xyz_list_file)
    print("  Low-level energies:", low_level_file)
    print("  CCSD energies:     ", ccsd_file)

    features = np.loadtxt(features_file)[:N_TOTAL]
    low_level_energies = np.loadtxt(low_level_file)[:N_TOTAL]
    ccsd_known = np.loadtxt(ccsd_file)[:N_CCSD]

    with open(xyz_list_file, "r") as f:
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

    scaler_for_gbr = StandardScaler().fit(X_train_raw2)
    X_train_scaled = scaler_for_gbr.transform(X_train_raw2)
    X_remaining_scaled = scaler_for_gbr.transform(X_remaining_raw2)
    X_unseen_scaled = scaler_for_gbr.transform(X_unseen_raw2)

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
        os.path.join(RESULTS_DIR, "ccsd_tail_weights_train.txt"),
        weights_train,
        fmt="%.8f",
    )

    idx_block = np.arange(N_TRAIN, N_CCSD)

    idx_val, idx_test = train_test_split(
        idx_block,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    map_test = idx_test - N_TRAIN

    X_test_gbr = X_remaining_scaled[map_test]
    X_test_pipe = X_remaining_raw2[map_test]

    ccsd_test = ccsd_remaining[map_test]
    low_test = low_remaining[map_test]

    np.savetxt(os.path.join(RESULTS_DIR, "val_indices.txt"), idx_val, fmt="%d")
    np.savetxt(os.path.join(RESULTS_DIR, "test_indices.txt"), idx_test, fmt="%d")

    gbr_model = load_or_train_gbr(
        X_train_scaled,
        y_train_delta,
        ccsd_train,
        low_train,
        weights_train,
    )

    svr_model = load_or_train_svr(
        X_train_raw2,
        y_train_delta,
        ccsd_train,
        low_train,
        weights_train,
    )

    gpr_model = load_or_train_gpr(
        X_train_raw2,
        y_train_delta,
        ccsd_train,
        low_train,
        weights_train,
    )

    metrics = []

    metrics.append(
        evaluate_model("GBR", gbr_model, X_test_gbr, ccsd_test, low_test)
    )

    metrics.append(
        evaluate_model("SVR", svr_model, X_test_pipe, ccsd_test, low_test)
    )

    metrics.append(
        evaluate_model("GPR", gpr_model, X_test_pipe, ccsd_test, low_test)
    )

    with open(SUMMARY_FILE, "w") as f:
        f.write(
            "model\tmae_delta\tmae_ccsd\tweighted_mae_ccsd\t"
            "tail_mae_ccsd\tn_tail\n"
        )

        for row in metrics:
            f.write(
                "{model}\t{mae_delta:.10f}\t{mae_ccsd:.10f}\t"
                "{weighted_mae_ccsd:.10f}\t{tail_mae_ccsd:.10f}\t"
                "{n_tail}\n".format(**row)
            )

    print("Wrote summary:", SUMMARY_FILE)

    pred_gbr = low_unseen + gbr_model.predict(X_unseen_scaled)
    pred_svr = low_unseen + svr_model.predict(X_unseen_raw2)
    pred_gpr = low_unseen + gpr_model.predict(X_unseen_raw2)

    np.savetxt(PRED_GBR_FILE, pred_gbr, fmt="%.10f")
    np.savetxt(PRED_SVR_FILE, pred_svr, fmt="%.10f")
    np.savetxt(PRED_GPR_FILE, pred_gpr, fmt="%.10f")

    print("Wrote predictions:")
    print("  ", PRED_GBR_FILE)
    print("  ", PRED_SVR_FILE)
    print("  ", PRED_GPR_FILE)

    run_info = {
        "low_level_label": LOW_LEVEL_LABEL,
        "features_file": features_file,
        "xyz_list_file": xyz_list_file,
        "low_level_energies_file": low_level_file,
        "ccsd_energies_file": ccsd_file,
        "n_total": N_TOTAL,
        "n_ccsd": N_CCSD,
        "n_train": N_TRAIN,
        "random_state": RANDOM_STATE,
        "add_low_level_as_feature": ADD_LOW_LEVEL_AS_FEATURE,
        "tail_weighted_ccsd_loss": USE_TAIL_WEIGHTED_CCSD_LOSS,
        "tail_q_ccsd": TAIL_Q_CCSD,
        "tail_factor_ccsd": TAIL_FACTOR_CCSD,
        "gbr_model_file": GBR_MODEL_FILE,
        "svr_model_file": SVR_MODEL_FILE,
        "gpr_model_file": GPR_MODEL_FILE,
    }

    with open(os.path.join(RESULTS_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()