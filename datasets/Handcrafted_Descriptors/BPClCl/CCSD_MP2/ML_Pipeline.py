import os
import numpy as np
from ase.io import read
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib
import optuna
import matplotlib.pyplot as plt

# =====================================================
# Controls
# =====================================================
ADD_MP2_AS_FEATURE = True      # append MP2 as last input feature (recommended)
RANDOM_STATE_BASE = 1
NUM_RANDOM_STATES = 120

# =====================================================
# NEW: Tail-weighting on *CCSD error* (not only delta error)
# =====================================================
USE_TAIL_WEIGHTED_CCSD_LOSS = True
TAIL_Q_CCSD = 0.10            # lowest 10% (most negative CCSD) is emphasized
TAIL_FACTOR_CCSD = 10.0       # weight multiplier for that tail (try 2, 5, 10, 20)
USE_WEIGHTS_IN_FIT = True     # also pass weights into .fit() where supported

# SVR supports sample_weight; sklearn GPR does not.
USE_SVR_WEIGHTS = True

# -----------------------------------------------------
# Optional: rank correlations (Spearman/Kendall)
# -----------------------------------------------------
USE_RANK_CORR = True
try:
    from scipy.stats import spearmanr, kendalltau
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -----------------------------
# Helpers
# -----------------------------
def extract_values_and_save(filename, output_file):
    extracted_values = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Angstrom" in line or "degrees" in line:
                value = float(line.split(":")[-1].strip().split()[0])
                extracted_values.append(value)
                output_file.write(f"{value:.8f}\n")
    return extracted_values

def save_dataset_with_absolute_energies(X, socassi_energies, mp2_energies, dataset_type, random_state):
    np.savetxt(f"{dataset_type}_features_random_state_{random_state}.txt", X, fmt="%.8f")
    np.savetxt(f"{dataset_type}_socassi_energies_random_state_{random_state}.txt", socassi_energies, fmt="%.10f")
    np.savetxt(f"{dataset_type}_mp2_energies_random_state_{random_state}.txt", mp2_energies, fmt="%.10f")
    print(f"{dataset_type.capitalize()} dataset with absolute energies saved for random state {random_state}.")

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
    Tail weights for *most negative* values in y.
    - y: energies (e.g., CCSD) where 'lower = better' (more negative)
    - q: fraction emphasized
    - factor: multiplier in the tail
    Returns weights normalized to mean 1.
    """
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y, dtype=float)
    cut = np.quantile(y, q)
    w[y <= cut] *= factor
    # normalize mean weight to 1
    w *= (len(w) / w.sum())
    return w

def tail_mae(y_true, y_pred, q=0.10):
    """
    MAE restricted to the lowest-q fraction of y_true (most negative CCSD tail).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cut = np.quantile(y_true, q)
    mask = (y_true <= cut)
    if mask.sum() == 0:
        return np.nan, 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))), int(mask.sum())

def rank_metrics_safe(y_true, y_pred, label=""):
    if not (USE_RANK_CORR and SCIPY_OK):
        return

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # constant/near-constant detection
    if np.allclose(y_true, y_true[0]):
        print(f"\nRank correlations for {label}: undefined (y_true constant).")
        return
    if np.allclose(y_pred, y_pred[0]):
        print(f"\nRank correlations for {label}: undefined (y_pred constant).")
        return

    rho, p_rho = spearmanr(y_true, y_pred)
    tau, p_tau = kendalltau(y_true, y_pred)

    print(f"\nRank correlations for {label}:")
    print(f"  Spearman rho = {rho:.6f} (p={p_rho:.3e})")
    print(f"  Kendall  tau = {tau:.6f} (p={p_tau:.3e})")

def quick_stats(name, x):
    x = np.asarray(x, dtype=float)
    uniq = np.unique(np.round(x, 12)).size
    print(f"{name}: min={x.min():.12e} max={x.max():.12e} std={x.std():.12e} unique(~1e-12)={uniq}")

# -----------------------------
# Load or build features
# -----------------------------
processed_features_file = "processed_features_final.txt"
if os.path.exists(processed_features_file):
    features = np.loadtxt(processed_features_file)
    with open("processed_xyz_files_final.txt", "r") as f:
        xyz_files = f.read().splitlines()
    print("Loaded processed features and XYZ file paths.")
else:
    print("Processed features file not found. Starting feature extraction...")
    input_filename = "Input_Geometries.txt"
    output_filename = "extracted_bond_lengths_and_angles.txt"
    directory = "/cygdrive/d/Unikram/Aufnahmen_und_Vorlesungen/Clean_Approach/Fresh/Structures"

    with open(output_filename, 'w') as output_file:
        extract_values_and_save(input_filename, output_file)
        print("Extracted values saved to", output_filename)

    xyz_files = []
    # If you want to rebuild features, put your xyz file enumeration here.
    print("Total files:", len(xyz_files))

    atomic_coordinates_list, bond_lengths_angles_list = [], []
    index = 0
    for xyz_file in xyz_files:
        try:
            atoms = read(xyz_file)
            atomic_coordinates = atoms.get_positions().flatten()
            atomic_coordinates_list.append(atomic_coordinates)

            with open(output_filename, 'r') as file:
                extracted_values = [float(line.strip()) for line in file]

            num_values = 13
            start_index = index * num_values
            end_index = start_index + num_values
            bond_lengths_angles = extracted_values[start_index:end_index]
            bond_lengths_angles_list.append(bond_lengths_angles)

            print(f"Processed: {xyz_file}")
            index += 1
        except Exception as e:
            print(f"Error processing {xyz_file}: {e}")

    features = np.concatenate((np.array(atomic_coordinates_list), np.array(bond_lengths_angles_list)), axis=1)
    np.savetxt(processed_features_file, features)
    print("Shape of features:", features.shape)
    print(f"Processed features saved as {processed_features_file}")

    with open("processed_xyz_files_final.txt", "w") as f:
        for file_path in xyz_files:
            f.write(file_path + "\n")
    print("Processed XYZ file paths saved as processed_xyz_files_final.txt")

# -----------------------------
# Energies
# -----------------------------
mp2_energies = np.loadtxt("MP2_ML_Final.txt")[:5714]
socassi_energies_known = np.loadtxt("CCSD_ML_Final.txt")[:1127]

# -----------------------------
# Define raw blocks (NO scaling yet) - avoids leakage
# -----------------------------
X_train_raw     = features[:1027]
X_remaining_raw = features[1027:1127]     # to be split into val/test randomly
X_unseen_raw    = features[1027:5714]     # for final predictions

socassi_train     = socassi_energies_known[:1027]
socassi_remaining = socassi_energies_known[1027:1127]

mp2_train     = mp2_energies[:1027]
mp2_remaining = mp2_energies[1027:1127]
mp2_unseen    = mp2_energies[1027:5714]

# =====================================================
# Append MP2 as feature column BEFORE scaling (optional)
# =====================================================
if ADD_MP2_AS_FEATURE:
    X_train_raw2     = append_mp2_feature(X_train_raw, mp2_train)
    X_remaining_raw2 = append_mp2_feature(X_remaining_raw, mp2_remaining)
    X_unseen_raw2    = append_mp2_feature(X_unseen_raw, mp2_unseen)
else:
    X_train_raw2, X_remaining_raw2, X_unseen_raw2 = X_train_raw, X_remaining_raw, X_unseen_raw

# -----------------------------
# Fit scaler on TRAIN ONLY; transform others - no leakage
# -----------------------------
scaler = StandardScaler().fit(X_train_raw2)
X_train_exclusive = scaler.transform(X_train_raw2)
X_remaining       = scaler.transform(X_remaining_raw2)
X_unseen_all      = scaler.transform(X_unseen_raw2)

# -----------------------------
# Targets
# -----------------------------
y_train_delta = socassi_train - mp2_train  # model predicts delta
# NEW: weights based on CCSD tail (training only; no leakage)
w_ccsd_train = build_tail_weights(socassi_train, q=TAIL_Q_CCSD, factor=TAIL_FACTOR_CCSD) \
    if USE_TAIL_WEIGHTED_CCSD_LOSS else np.ones_like(socassi_train, dtype=float)

np.savetxt("ccsd_tail_weights_train.txt", w_ccsd_train, fmt="%.8f")
print("Wrote ccsd_tail_weights_train.txt (tail weights for the 1027 training rows)")

# -----------------------------
# Runs
# -----------------------------
all_predictions = []

for i in range(NUM_RANDOM_STATES):
    random_state = RANDOM_STATE_BASE + i
    print(f"\nProcessing random state: {random_state}")

    idx_block = np.arange(1027, 1127)
    idx_val, idx_test = train_test_split(idx_block, test_size=0.5, random_state=random_state)

    map_val  = idx_val  - 1027
    map_test = idx_test - 1027

    X_val  = X_remaining[map_val]
    X_test = X_remaining[map_test]

    socassi_val  = socassi_remaining[map_val]
    socassi_test = socassi_remaining[map_test]

    mp2_val  = mp2_remaining[map_val]
    mp2_test = mp2_remaining[map_test]

    save_dataset_with_absolute_energies(X_train_exclusive, socassi_train, mp2_train, "train", random_state)
    save_dataset_with_absolute_energies(X_val, socassi_val, mp2_val, "val", random_state)
    save_dataset_with_absolute_energies(X_test, socassi_test, mp2_test, "test", random_state)

    np.savetxt(f"val_indices_{random_state}.txt",  idx_val,  fmt="%d")
    np.savetxt(f"test_indices_{random_state}.txt", idx_test, fmt="%d")

    # =====================================================
    # 1) GBR + Optuna
    #    Objective: Tail-weighted MAE on CCSD error (NOT delta error)
    # =====================================================
    model_file_path = f"final_gb_model_{random_state}.pkl"
    if os.path.exists(model_file_path):
        final_gb_model = joblib.load(model_file_path)
        print(f"Trained GBR loaded from {model_file_path}")
    else:
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 100, 8000)
            max_depth    = trial.suggest_int("max_depth", 1, 10)

            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            fold_scores = []

            for tr_idx, va_idx in kf.split(X_train_exclusive):
                X_tr, X_va = X_train_exclusive[tr_idx], X_train_exclusive[va_idx]
                y_tr, y_va_delta = y_train_delta[tr_idx], y_train_delta[va_idx]

                # true CCSD and MP2 for validation fold
                ccsd_va_true = socassi_train[va_idx]
                mp2_va       = mp2_train[va_idx]

                # weights (CCSD-tail) for train/val fold
                w_tr = w_ccsd_train[tr_idx]
                w_va = w_ccsd_train[va_idx]

                gb = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )

                if USE_WEIGHTS_IN_FIT:
                    gb.fit(X_tr, y_tr, sample_weight=w_tr)
                else:
                    gb.fit(X_tr, y_tr)

                delta_va_pred = gb.predict(X_va)
                ccsd_va_pred  = mp2_va + delta_va_pred

                # Tail-weighted loss on CCSD prediction error
                score = weighted_mae(ccsd_va_true, ccsd_va_pred, w_va)
                fold_scores.append(score)

            return float(np.mean(fold_scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2, n_jobs=-1)

        best_params = study.best_params
        with open(f"best_hyperparameters_{random_state}.txt", "w") as params_file:
            params_file.write(f"n_estimators={best_params['n_estimators']}\n")
            params_file.write(f"max_depth={best_params['max_depth']}\n")
        print("Best Hyperparameters (GBR):", best_params)

        final_gb_model = GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=random_state
        )

        if USE_WEIGHTS_IN_FIT:
            final_gb_model.fit(X_train_exclusive, y_train_delta, sample_weight=w_ccsd_train)
        else:
            final_gb_model.fit(X_train_exclusive, y_train_delta)

        joblib.dump(final_gb_model, model_file_path)
        print(f"Trained GBR saved to {model_file_path}")

    # -----------------------------------------------------
    # GBR evaluation on test
    # -----------------------------------------------------
    predicted_delta_test = final_gb_model.predict(X_test)
    y_test_delta = socassi_test - mp2_test
    ccsd_pred_test = mp2_test + predicted_delta_test

    # Standard MAEs
    mae_delta_test = mean_absolute_error(y_test_delta, predicted_delta_test)
    mae_ccsd_test  = mean_absolute_error(socassi_test, ccsd_pred_test)

    # Tail-weighted CCSD MAE on test (for reporting)
    w_ccsd_test = build_tail_weights(socassi_test, q=TAIL_Q_CCSD, factor=TAIL_FACTOR_CCSD) \
        if USE_TAIL_WEIGHTED_CCSD_LOSS else np.ones_like(socassi_test, dtype=float)
    wmae_ccsd_test = weighted_mae(socassi_test, ccsd_pred_test, w_ccsd_test)

    tail_mae_ccsd, tail_n = tail_mae(socassi_test, ccsd_pred_test, q=TAIL_Q_CCSD)

    print(f"[GBR] Test MAE (delta): {mae_delta_test:.12f}")
    print(f"[GBR] Test MAE (CCSD):  {mae_ccsd_test:.12f}")
    print(f"[GBR] Test tail-weighted MAE (CCSD): {wmae_ccsd_test:.12f}  (q={TAIL_Q_CCSD}, factor={TAIL_FACTOR_CCSD})")
    print(f"[GBR] Test tail-only MAE (CCSD):      {tail_mae_ccsd:.12f}  (N_tail={tail_n})")

    # Diagnostics that often explain Spearman/Kendall issues
    quick_stats("Delta_true(test)", y_test_delta)
    quick_stats("Delta_pred(test)", predicted_delta_test)

    # Rank correlations (baseline vs model)
    rank_metrics_safe(socassi_test, mp2_test, label="Baseline: CCSD_true vs MP2 (test)")
    rank_metrics_safe(socassi_test, ccsd_pred_test, label="Model: CCSD_true vs CCSD_pred (test)")
    rank_metrics_safe(y_test_delta, predicted_delta_test, label="Model: Delta_true vs Delta_pred (test)")

    # Write detailed diagnostics (so you can inspect the symptomatic rows)
    out_test = np.column_stack([
        socassi_test,                  # CCSD_true
        mp2_test,                      # MP2
        y_test_delta,                  # Delta_true
        predicted_delta_test,          # Delta_pred
        ccsd_pred_test,                # CCSD_pred
        np.abs(ccsd_pred_test - socassi_test)  # AbsErr_CCSD
    ])
    header = "CCSD_true\tMP2\tDelta_true\tDelta_pred\tCCSD_pred\tAbsErr_CCSD"
    np.savetxt(f"gbr_test_diagnostics_{random_state}.tsv",
               out_test, fmt="%.10f", delimiter="\t", header=header)
    print(f"[GBR] Wrote: gbr_test_diagnostics_{random_state}.tsv")

    # Predict unseen and write predictions
    predicted_delta_unseen = final_gb_model.predict(X_unseen_all)
    socassi_pred_unseen_gbr = mp2_unseen + predicted_delta_unseen
    with open(f"predicted_SOCASSI_GB_unseen_optuna_{random_state}.txt", "w") as output_file:
        for energy in socassi_pred_unseen_gbr:
            output_file.write(f"{energy:.8f}\n")
    print(f"[GBR] Unseen predictions saved for random state {random_state}")

    # =====================================================
    # 2) SVR via Pipeline + Optuna
    #    Objective: tail-weighted MAE on CCSD error
    # =====================================================
    svr_model_path = f"final_svr_pipeline_{random_state}.pkl"
    if os.path.exists(svr_model_path):
        svr_pipe = joblib.load(svr_model_path)
        print(f"[SVR] Loaded pipeline from {svr_model_path}")
    else:
        def objective_svr(trial):
            C = trial.suggest_float("C", 1e-2, 1e3, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-4, 0.5, log=True)
            gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            scores = []

            for tr_idx, va_idx in kf.split(X_train_raw2):
                X_tr, X_va = X_train_raw2[tr_idx], X_train_raw2[va_idx]
                y_tr = y_train_delta[tr_idx]

                ccsd_va_true = socassi_train[va_idx]
                mp2_va       = mp2_train[va_idx]
                w_tr = w_ccsd_train[tr_idx]
                w_va = w_ccsd_train[va_idx]

                if USE_SVR_WEIGHTS and USE_WEIGHTS_IN_FIT:
                    pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
                else:
                    pipe.fit(X_tr, y_tr)

                delta_va_pred = pipe.predict(X_va)
                ccsd_va_pred  = mp2_va + delta_va_pred

                score = weighted_mae(ccsd_va_true, ccsd_va_pred, w_va) \
                        if USE_TAIL_WEIGHTED_CCSD_LOSS else mean_absolute_error(ccsd_va_true, ccsd_va_pred)
                scores.append(score)

            return float(np.mean(scores))

        study_svr = optuna.create_study(direction="minimize")
        study_svr.optimize(objective_svr, n_trials=42, n_jobs=-1)
        best_svr = study_svr.best_params
        print(f"[SVR] Best params: {best_svr}")

        svr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf",
                          C=best_svr["C"], epsilon=best_svr["epsilon"], gamma=best_svr["gamma"]))
        ])

        if USE_SVR_WEIGHTS and USE_WEIGHTS_IN_FIT:
            svr_pipe.fit(X_train_raw2, y_train_delta, model__sample_weight=w_ccsd_train)
        else:
            svr_pipe.fit(X_train_raw2, y_train_delta)

        joblib.dump(svr_pipe, svr_model_path)
        print(f"[SVR] Pipeline saved to {svr_model_path}")

    svr_delta_test = svr_pipe.predict(X_remaining_raw2[map_test])
    svr_ccsd_pred_test = mp2_test + svr_delta_test
    svr_mae_ccsd = mean_absolute_error(socassi_test, svr_ccsd_pred_test)
    svr_wmae_ccsd = weighted_mae(socassi_test, svr_ccsd_pred_test, w_ccsd_test)
    svr_tail_mae, svr_tail_n = tail_mae(socassi_test, svr_ccsd_pred_test, q=TAIL_Q_CCSD)

    print(f"[SVR] Test MAE (CCSD): {svr_mae_ccsd:.12f}")
    print(f"[SVR] Test tail-weighted MAE (CCSD): {svr_wmae_ccsd:.12f}")
    print(f"[SVR] Test tail-only MAE (CCSD):      {svr_tail_mae:.12f}  (N_tail={svr_tail_n})")
    rank_metrics_safe(socassi_test, svr_ccsd_pred_test, label="SVR: CCSD_true vs CCSD_pred (test)")

    svr_pred_delta_unseen = svr_pipe.predict(X_unseen_raw2)
    svr_socassi_pred_unseen = mp2_unseen + svr_pred_delta_unseen
    with open(f"predicted_SOCASSI_SVR_unseen_optuna_{random_state}.txt", "w") as f_out:
        for e in svr_socassi_pred_unseen:
            f_out.write(f"{e:.8f}\n")
    print(f"[SVR] Unseen predictions saved for random state {random_state}")

    # =====================================================
    # 3) GPR via Pipeline + Optuna
    #    Objective: tail-weighted MAE on CCSD error (hyperparam selection only)
    # =====================================================
    gpr_model_path = f"final_gpr_pipeline_{random_state}.pkl"
    if os.path.exists(gpr_model_path):
        gpr_pipe = joblib.load(gpr_model_path)
        print(f"[GPR] Loaded pipeline from {gpr_model_path}")
    else:
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
                    random_state=random_state
                ))
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            scores = []

            for tr_idx, va_idx in kf.split(X_train_raw2):
                X_tr, X_va = X_train_raw2[tr_idx], X_train_raw2[va_idx]
                y_tr = y_train_delta[tr_idx]

                ccsd_va_true = socassi_train[va_idx]
                mp2_va       = mp2_train[va_idx]
                w_va = w_ccsd_train[va_idx]

                pipe.fit(X_tr, y_tr)
                delta_va_pred = pipe.predict(X_va)
                ccsd_va_pred  = mp2_va + delta_va_pred

                score = weighted_mae(ccsd_va_true, ccsd_va_pred, w_va) \
                        if USE_TAIL_WEIGHTED_CCSD_LOSS else mean_absolute_error(ccsd_va_true, ccsd_va_pred)
                scores.append(score)

            return float(np.mean(scores))

        study_gpr = optuna.create_study(direction="minimize")
        study_gpr.optimize(objective_gpr, n_trials=2, n_jobs=-1)  # GPR is expensive; raise if you want
        best_gpr = study_gpr.best_params
        print(f"[GPR] Best params: {best_gpr}")

        kernel_final = RBF(length_scale=best_gpr["length_scale"]) + WhiteKernel(noise_level=best_gpr["alpha"])
        gpr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=kernel_final,
                alpha=best_gpr["alpha"],
                normalize_y=False,
                random_state=random_state
            ))
        ])
        gpr_pipe.fit(X_train_raw2, y_train_delta)
        joblib.dump(gpr_pipe, gpr_model_path)
        print(f"[GPR] Pipeline saved to {gpr_model_path}")

    gpr_delta_test = gpr_pipe.predict(X_remaining_raw2[map_test])
    gpr_ccsd_pred_test = mp2_test + gpr_delta_test
    gpr_mae_ccsd = mean_absolute_error(socassi_test, gpr_ccsd_pred_test)
    gpr_wmae_ccsd = weighted_mae(socassi_test, gpr_ccsd_pred_test, w_ccsd_test)
    gpr_tail_mae, gpr_tail_n = tail_mae(socassi_test, gpr_ccsd_pred_test, q=TAIL_Q_CCSD)

    print(f"[GPR] Test MAE (CCSD): {gpr_mae_ccsd:.12f}")
    print(f"[GPR] Test tail-weighted MAE (CCSD): {gpr_wmae_ccsd:.12f}")
    print(f"[GPR] Test tail-only MAE (CCSD):      {gpr_tail_mae:.12f}  (N_tail={gpr_tail_n})")
    rank_metrics_safe(socassi_test, gpr_ccsd_pred_test, label="GPR: CCSD_true vs CCSD_pred (test)")

    gpr_pred_delta_unseen = gpr_pipe.predict(X_unseen_raw2)
    gpr_socassi_pred_unseen = mp2_unseen + gpr_pred_delta_unseen
    with open(f"predicted_SOCASSI_GPR_unseen_optuna_{random_state}.txt", "w") as f_out:
        for e in gpr_socassi_pred_unseen:
            f_out.write(f"{e:.8f}\n")
    print(f"[GPR] Unseen predictions saved for random state {random_state}")

    all_predictions.append({
        'random_state': random_state,
        'GBR_MAE_CCSD': mae_ccsd_test,
        'GBR_WMAE_CCSD': wmae_ccsd_test,
        'GBR_TAIL_MAE_CCSD': tail_mae_ccsd,
        'SVR_MAE_CCSD': svr_mae_ccsd,
        'SVR_WMAE_CCSD': svr_wmae_ccsd,
        'SVR_TAIL_MAE_CCSD': svr_tail_mae,
        'GPR_MAE_CCSD': gpr_mae_ccsd,
        'GPR_WMAE_CCSD': gpr_wmae_ccsd,
        'GPR_TAIL_MAE_CCSD': gpr_tail_mae
    })

# -----------------------------
# Summary
# -----------------------------
with open("all_predictions_summary.txt", "w") as summary_file:
    for r in all_predictions:
        summary_file.write(
            f"Random State: {r['random_state']}, "
            f"GBR_MAE_CCSD: {r['GBR_MAE_CCSD']:.10f}, GBR_WMAE_CCSD: {r['GBR_WMAE_CCSD']:.10f}, GBR_TAIL_MAE_CCSD: {r['GBR_TAIL_MAE_CCSD']:.10f}, "
            f"SVR_MAE_CCSD: {r['SVR_MAE_CCSD']:.10f}, SVR_WMAE_CCSD: {r['SVR_WMAE_CCSD']:.10f}, SVR_TAIL_MAE_CCSD: {r['SVR_TAIL_MAE_CCSD']:.10f}, "
            f"GPR_MAE_CCSD: {r['GPR_MAE_CCSD']:.10f}, GPR_WMAE_CCSD: {r['GPR_WMAE_CCSD']:.10f}, GPR_TAIL_MAE_CCSD: {r['GPR_TAIL_MAE_CCSD']:.10f}\n"
        )

print("\nAll predictions summary saved to all_predictions_summary.txt")
print(f"MP2 appended as feature column: {ADD_MP2_AS_FEATURE}")
print(f"Tail-weighted CCSD loss active: {USE_TAIL_WEIGHTED_CCSD_LOSS} (q={TAIL_Q_CCSD}, factor={TAIL_FACTOR_CCSD})")
print(f"Weights used in fit where supported: {USE_WEIGHTS_IN_FIT}")
