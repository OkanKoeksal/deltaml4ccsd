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
    np.savetxt(f"{dataset_type}_socassi_energies_random_state_{random_state}.txt", socassi_energies, fmt="%.8f")
    np.savetxt(f"{dataset_type}_mp2_energies_random_state_{random_state}.txt", mp2_energies, fmt="%.8f")
    print(f"{dataset_type.capitalize()} dataset with absolute energies saved for random state {random_state}.")

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
    directory = "/cygdrive/d/Unikram/Aufnahmen_und_Vorlesungen/Forschungsarbeit_Donor_Acceptor_AI/ML_Donor_Acceptor/Fig6/F3B_PF3_8f_NEW_100_CCSD_Energies_R_0_09_901_Values_Leakage_Free_Only_5503_Structs/Structures"

    with open(output_filename, 'w') as output_file:
        extract_values_and_save(input_filename, output_file)
        print("Extracted values saved to", output_filename)

    xyz_files = []
    for i in range(1, 301):
        filepath = os.path.join(directory, f"randomR0001A{i}.xyz")
        if os.path.exists(filepath):
            xyz_files.append(filepath)	

    for i in range(1, 101):
        filepath = os.path.join(directory, f"randomR009A{i}.xyz")
        if os.path.exists(filepath):
            xyz_files.append(filepath)
	
    # Generate the list of file names for output1.xyz to output5003.xyz
    for i in range(0, 5003):
        file_name = f"output{i}.xyz"
        file_path = os.path.join(directory, file_name)
        xyz_files.append(file_path)


    for i in range(1, 101):
        filepath = os.path.join(directory, f"perturbations{i}.xyz")
        if os.path.exists(filepath):
            xyz_files.append(filepath)

    print("Total files:", len(xyz_files))

    atomic_coordinates_list, dipole_moments_list, bond_lengths_angles_list = [], [], []
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

    features = np.concatenate((
        np.array(atomic_coordinates_list),
        np.array(bond_lengths_angles_list)
    ), axis=1)
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
mp2_energies = np.loadtxt("MP2_ML_Final.txt")[:5497]
socassi_energies_known = np.loadtxt("CCSD_ML_Final.txt")[:900]

# -----------------------------
# Define raw blocks (NO scaling yet) - avoids leakage
# -----------------------------
X_train_raw     = features[:800]
X_remaining_raw = features[800:900]     # to be split into val/test randomly
X_unseen_raw    = features[800:5497]    # for final predictions (same slice as remaining + more)

socassi_train     = socassi_energies_known[:800]
socassi_remaining = socassi_energies_known[800:900]

mp2_train     = mp2_energies[:800]
mp2_remaining = mp2_energies[800:900]

# -----------------------------
# Fit scaler on TRAIN ONLY; transform others - no leakage
# -----------------------------
scaler = StandardScaler().fit(X_train_raw)
X_train_exclusive = scaler.transform(X_train_raw)
X_remaining       = scaler.transform(X_remaining_raw)
X_unseen_all      = scaler.transform(X_unseen_raw)

# -----------------------------
# Fixed base seed + 7 deterministic runs
# -----------------------------
num_random_states = 120
base_seed = 1
all_predictions = []

for i in range(num_random_states):
    random_state = base_seed + i
    print(f"Processing random state: {random_state}")

    # Split remaining block (800-900) into val/test (random partition, not contiguous ranges)
    idx_block = np.arange(800, 900)
    idx_val, idx_test = train_test_split(idx_block, test_size=0.5, random_state=random_state)

    # Map indices within 800:900 to 0:300 positions
    map_val  = idx_val  - 800
    map_test = idx_test - 800

    X_val  = X_remaining[map_val]
    X_test = X_remaining[map_test]

    socassi_val  = socassi_remaining[map_val]
    socassi_test = socassi_remaining[map_test]

    mp2_val  = mp2_remaining[map_val]
    mp2_test = mp2_remaining[map_test]

    # Save datasets (features + absolute energies) that match the actual split
    save_dataset_with_absolute_energies(X_train_exclusive, socassi_train, mp2_train, "train", random_state)
    save_dataset_with_absolute_energies(X_val, socassi_val, mp2_val, "val", random_state)
    save_dataset_with_absolute_energies(X_test, socassi_test, mp2_test, "test", random_state)

    # (Optional) save the indices for auditing
    np.savetxt(f"val_indices_{random_state}.txt",  idx_val,  fmt="%d")
    np.savetxt(f"test_indices_{random_state}.txt", idx_test, fmt="%d")

	
    # =====================================================
    # 1) SVR (SVM) via Pipeline + Optuna
    # =====================================================
    svr_model_path = f"final_svr_pipeline_{random_state}.pkl"
    if os.path.exists(svr_model_path):
        svr_pipe = joblib.load(svr_model_path)
        print(f"[SVR] Loaded pipeline from {svr_model_path}")
    else:
        def objective_svr(trial):
            C = trial.suggest_float("C", 1e-2, 1e3, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-3, 0.5, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            maes = []
            y_train_delta = socassi_train - mp2_train

            for tr_idx, va_idx in kf.split(X_train_raw):  # RAW; scaler fits inside pipeline
                X_tr, X_va = X_train_raw[tr_idx], X_train_raw[va_idx]
                y_tr, y_va = y_train_delta[tr_idx], y_train_delta[va_idx]
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_va)
                maes.append(mean_absolute_error(y_va, pred))

            return float(np.mean(maes))

        study_svr = optuna.create_study(direction="minimize")
        study_svr.optimize(objective_svr, n_trials=24, n_jobs=-1)
        best_svr = study_svr.best_params
        print(f"[SVR] Best params: {best_svr}")

        svr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf",
                          C=best_svr["C"], epsilon=best_svr["epsilon"], gamma=best_svr["gamma"]))
        ])
        svr_pipe.fit(X_train_raw, socassi_train - mp2_train)
        joblib.dump(svr_pipe, svr_model_path)
        print(f"[SVR] Pipeline saved to {svr_model_path}")

    # Evaluate SVR on the same held-out test (delta)
    svr_pred_test_delta = svr_pipe.predict(X_remaining_raw[map_test])
    svr_mae_test = mean_absolute_error((socassi_test - mp2_test), svr_pred_test_delta)
    print(f"[SVR] MAE on Test Set (random state {random_state}): {svr_mae_test}")

    # Predict strictly unseen using RAW features; pipeline handles scaling
    svr_pred_delta_unseen = svr_pipe.predict(X_unseen_raw)
    svr_socassi_pred_unseen = mp2_energies[800:5497] + svr_pred_delta_unseen
    with open(f"predicted_SOCASSI_SVR_unseen_optuna_{random_state}.txt", "w") as f_out:
        for e in svr_socassi_pred_unseen:
            f_out.write(f"{e:.8f}\n")
    print(f"[SVR] Unseen predictions saved for random state {random_state}")

    # =====================================================
    # 2) Gaussian Process Regressor via Pipeline + Optuna
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
                    kernel=kernel, alpha=alpha, normalize_y=False, random_state=random_state
                ))
            ])

            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            maes = []
            y_train_delta = socassi_train - mp2_train

            for tr_idx, va_idx in kf.split(X_train_raw):
                X_tr, X_va = X_train_raw[tr_idx], X_train_raw[va_idx]
                y_tr, y_va = y_train_delta[tr_idx], y_train_delta[va_idx]
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_va)
                maes.append(mean_absolute_error(y_va, pred))

            return float(np.mean(maes))

        study_gpr = optuna.create_study(direction="minimize")
        study_gpr.optimize(objective_gpr, n_trials=24, n_jobs=-1)  # GPR heavier
        best_gpr = study_gpr.best_params
        print(f"[GPR] Best params: {best_gpr}")

        kernel_final = RBF(length_scale=best_gpr["length_scale"]) + WhiteKernel(noise_level=best_gpr["alpha"])
        gpr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianProcessRegressor(
                kernel=kernel_final, alpha=best_gpr["alpha"], normalize_y=False, random_state=random_state
            ))
        ])
        gpr_pipe.fit(X_train_raw, socassi_train - mp2_train)
        joblib.dump(gpr_pipe, gpr_model_path)
        print(f"[GPR] Pipeline saved to {gpr_model_path}")

    # Evaluate GPR on held-out test (delta)
    gpr_pred_test_delta = gpr_pipe.predict(X_remaining_raw[map_test])
    gpr_mae_test = mean_absolute_error((socassi_test - mp2_test), gpr_pred_test_delta)
    print(f"[GPR] MAE on Test Set (random state {random_state}): {gpr_mae_test}")

    # Predict strictly unseen
    gpr_pred_delta_unseen = gpr_pipe.predict(X_unseen_raw)
    gpr_socassi_pred_unseen = mp2_energies[800:5497] + gpr_pred_delta_unseen
    with open(f"predicted_SOCASSI_GPR_unseen_optuna_{random_state}.txt", "w") as f_out:
        for e in gpr_socassi_pred_unseen:
            f_out.write(f"{e:.8f}\n")
    print(f"[GPR] Unseen predictions saved for random state {random_state}")

    # =====================================================
    # 3) GradientBoostingRegressor + Optuna (LAST)
    # =====================================================
    model_file_path = f"final_gb_model_{random_state}.pkl"
    if os.path.exists(model_file_path):
        final_gb_model = joblib.load(model_file_path)
        print(f"[GBR] Trained GBR loaded from {model_file_path}")
    else:
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 1, 10000)
            max_depth    = trial.suggest_int("max_depth", 1, 20)

            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            mae_folds = []

            for tr_idx, va_idx in kf.split(X_train_exclusive):
                X_tr, X_va = X_train_exclusive[tr_idx], X_train_exclusive[va_idx]
                y_tr = socassi_train[tr_idx] - mp2_train[tr_idx]
                y_va = socassi_train[va_idx] - mp2_train[va_idx]

                gb = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )
                gb.fit(X_tr, y_tr)
                pred_va = gb.predict(X_va)
                mae_folds.append(mean_absolute_error(y_va, pred_va))

            return float(np.mean(mae_folds))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=24, n_jobs=-1)

        best_params = study.best_params
        with open(f"best_hyperparameters_{random_state}.txt", "w") as params_file:
            params_file.write(f"n_estimators={best_params['n_estimators']}\n")
            params_file.write(f"max_depth={best_params['max_depth']}\n")
        print("[GBR] Best Hyperparameters:", best_params)

        final_gb_model = GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=random_state
        )
        final_gb_model.fit(X_train_exclusive, socassi_train - mp2_train)
        joblib.dump(final_gb_model, model_file_path)
        print(f"[GBR] Trained GBR saved to {model_file_path}")

    # Evaluate GBR on test (? = SOCASSI - MP2)
    predicted_delta_test = final_gb_model.predict(X_test)
    mae_test_gbr = mean_absolute_error(socassi_test - mp2_test, predicted_delta_test)
    print(f"[GBR] MAE on Test Set (random state {random_state}): {mae_test_gbr}")

    # Predict for unseen (800-5497) using the scaler fit on train
    predicted_delta_unseen = final_gb_model.predict(X_unseen_all)
    socassi_pred_unseen_gbr = mp2_energies[800:5497] + predicted_delta_unseen

    with open(f"predicted_SOCASSI_GB_unseen_optuna_{random_state}.txt", "w") as output_file:
        for energy in socassi_pred_unseen_gbr:
            output_file.write(f"{energy:.8f}\n")
    print(f"[GBR] Unseen predictions saved for random state {random_state}")

    # -----------------------------
    # Collect summary per seed
    # -----------------------------
    all_predictions.append({
        'random_state': random_state,
        'GBR_MAE': mae_test_gbr,
        'SVR_MAE': svr_mae_test,
        'GPR_MAE': gpr_mae_test
    })

# -----------------------------
# Summary
# -----------------------------
with open("all_predictions_summary.txt", "w") as summary_file:
    for r in all_predictions:
        summary_file.write(
            f"Random State: {r['random_state']}, "
            f"GBR_MAE: {r['GBR_MAE']:.8f}, "
            f"SVR_MAE: {r['SVR_MAE']:.8f}, "
            f"GPR_MAE: {r['GPR_MAE']:.8f}\n"
        )
print("All predictions across 7 fixed, deterministic seeds saved (GBR + SVR + GPR).")
