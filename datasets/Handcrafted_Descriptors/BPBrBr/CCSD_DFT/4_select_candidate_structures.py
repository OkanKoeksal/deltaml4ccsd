import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =====================================================
# User controls
# =====================================================

LOW_LEVEL_LABEL = "DFT"

# Prefer reordered files if they exist.
VALIDATION_FILE = "CCSD_ML_Final_reordered.txt"
FALLBACK_VALIDATION_FILE = "CCSD_ML_Final.txt"

PROCESSED_XYZ_LIST = "processed_xyz_files_reordered.txt"
FALLBACK_PROCESSED_XYZ_LIST = "processed_xyz_files_final.txt"

# Prediction line 1 corresponds to dataset row 800 because the ML script predicts
# rows N_TRAIN:N_TOTAL.
LINE_OFFSET = 800

# Compare first 172 prediction rows against CCSD rows 800:972.
N_COMPARE = 172

# Predictions produced by the clean training script.
PREDICTION_FILES = {
    "GB":  "results_clean/predicted_ccsd_from_dft_gbr.txt",
    "GPR": "results_clean/predicted_ccsd_from_dft_gpr.txt",
    "SVR": "results_clean/predicted_ccsd_from_dft_svr.txt",
}

# Optional fallback patterns for older output names.
FALLBACK_PRED_PATTERNS = [
    "predicted_CCSD_GB_unseen_optuna_*.txt",
    "predicted_CCSD_GPR_unseen_optuna_*.txt",
    "predicted_CCSD_SVR_unseen_optuna_*.txt",
    "predicted_SOCASSI_GB_unseen_optuna_*.txt",
    "predicted_SOCASSI_GPR_unseen_optuna_*.txt",
    "predicted_SOCASSI_SVR_unseen_optuna_*.txt",
]

# If None, measured MAE from the 172-point comparison is used as the energy window.
MANUAL_MAE_FOR_WINDOW = None

# Set to None to always plot/scan.
# Example: 1.0e-3 skips files whose max abs diff exceeds 1e-3 Hartree.
MAX_ABS_DIFF_GATE = None

OUTPUT_PLOTS_DIR = "output_combined_plots_clean"
FOUND_DIR = "Found_Within_Range_clean"
METRICS_SUMMARY = "all_regression_metrics_clean.txt"

# If too many structures fall in the minimum-energy window, skip copying.
HITS_COPY_THRESHOLD = 170


# =====================================================
# Helpers
# =====================================================

def choose_existing_file(primary, fallback):
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Neither file exists: {primary} nor {fallback}")


def load_first_column(path):
    arr = np.loadtxt(path, dtype=float)
    arr = np.atleast_1d(arr)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr


def extract_number(filename):
    m = re.search(r"(\d+)(?=\D*$)", filename)
    return int(m.group(1)) if m else 10**18


def infer_model_tag(path):
    base = os.path.basename(path)

    lowered = base.lower()

    if "gpr" in lowered:
        return "GPR"
    if "svr" in lowered:
        return "SVR"
    if "gb" in lowered or "gbr" in lowered:
        return "GB"

    return "X"


def collect_prediction_files():
    files = []

    for model_tag, path in PREDICTION_FILES.items():
        if os.path.exists(path):
            files.append((model_tag, path))

    if files:
        return files

    # Fallback for older names.
    import glob
    fallback_files = []
    for pattern in FALLBACK_PRED_PATTERNS:
        fallback_files.extend(glob.glob(pattern))

    fallback_files = sorted(set(fallback_files), key=extract_number)

    for path in fallback_files:
        files.append((infer_model_tag(path), path))

    if not files:
        raise FileNotFoundError(
            "No prediction files found. Expected either results_clean/predicted_ccsd_from_dft_*.txt "
            "or older predicted_CCSD_* / predicted_SOCASSI_* files."
        )

    return files


def safe_copy_structure(xyz_path, target_dir):
    """
    Copy xyz_path to target_dir.

    Works with:
      - new relative paths from processed_xyz_files_reordered.txt
      - old absolute paths from processed_xyz_files_final.txt
    """
    xyz_path = xyz_path.strip()
    if not xyz_path:
        return False

    source = xyz_path

    if not os.path.exists(source):
        # If list contains stale absolute paths, try basename in current directory tree.
        basename = os.path.basename(source)

        candidate_dirs = [
            "Structures_reordered",
            "Structures_renamed",
            "Structures",
            "Structures_ML",
        ]

        found = None
        for d in candidate_dirs:
            candidate = os.path.join(d, basename)
            if os.path.exists(candidate):
                found = candidate
                break

        if found is None:
            print(f"[WARN] Could not find xyz file: {xyz_path}")
            return False

        source = found

    shutil.copy2(source, os.path.join(target_dir, os.path.basename(source)))
    return True


def main():
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    os.makedirs(FOUND_DIR, exist_ok=True)

    validation_file = choose_existing_file(
        VALIDATION_FILE,
        FALLBACK_VALIDATION_FILE,
    )

    processed_xyz_list = choose_existing_file(
        PROCESSED_XYZ_LIST,
        FALLBACK_PROCESSED_XYZ_LIST,
    )

    print("Using files:")
    print(f"  Validation CCSD file: {validation_file}")
    print(f"  XYZ list:             {processed_xyz_list}")

    validation_all = load_first_column(validation_file)

    if validation_all.size < LINE_OFFSET + N_COMPARE:
        raise ValueError(
            f"{validation_file} has {validation_all.size} values, but need at least "
            f"{LINE_OFFSET + N_COMPARE} for rows {LINE_OFFSET}:{LINE_OFFSET + N_COMPARE}."
        )

    # Compare prediction rows 1:172 with known CCSD rows 800:972.
    validation_data = validation_all[LINE_OFFSET:LINE_OFFSET + N_COMPARE]

    with open(processed_xyz_list, "r") as f:
        processed_lines = f.readlines()

    prediction_files = collect_prediction_files()

    print("\nPrediction files:")
    for model_tag, pred_path in prediction_files:
        print(f"  {model_tag}: {pred_path}")

    with open(METRICS_SUMMARY, "w") as metrics_file:
        metrics_file.write(
            "File_Name\tModel\tR2\tMSE\tMAE_measured\tStdErr\tMAE_window_used\t"
            "Most_negative\tLower_window\tUpper_window\tHits\n"
        )

        for model_tag, pred_path in prediction_files:
            base = os.path.basename(pred_path)
            name_wo_ext = os.path.splitext(base)[0]

            try:
                pred_all = load_first_column(pred_path)
            except Exception as exc:
                print(f"[SKIP] Could not load {pred_path}: {exc}")
                continue

            if pred_all.size < N_COMPARE:
                print(f"[SKIP] {base}: only {pred_all.size} predictions, need {N_COMPARE}.")
                continue

            predicted_data = pred_all[:N_COMPARE].copy()
            vdata = validation_data.copy()

            coeffs = np.polyfit(vdata, predicted_data, 1)
            regression_line = np.polyval(coeffs, vdata)

            r2 = r2_score(vdata, predicted_data)
            mse = mean_squared_error(vdata, predicted_data)
            mae_measured = mean_absolute_error(vdata, predicted_data)

            residuals = predicted_data - regression_line
            stderr = np.sqrt(np.sum(residuals ** 2) / max(1, len(vdata) - 2))

            mae_window = (
                float(MANUAL_MAE_FOR_WINDOW)
                if MANUAL_MAE_FOR_WINDOW is not None
                else float(mae_measured)
            )

            abs_diff = np.abs(predicted_data - vdata)
            max_abs_diff = float(np.max(abs_diff))

            if MAX_ABS_DIFF_GATE is not None and max_abs_diff > MAX_ABS_DIFF_GATE:
                print(
                    f"[SKIP] {base}: max abs diff {max_abs_diff:.3e} "
                    f"> gate {MAX_ABS_DIFF_GATE:.3e}. Skipping plot and range scan."
                )
                continue

            # -----------------------------
            # Plot regression + error histogram
            # -----------------------------
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].scatter(
                vdata,
                predicted_data,
                s=100,
                marker="x",
                label=f"Predicted CCSD ({model_tag})",
            )
            axs[0].scatter(
                vdata,
                vdata,
                s=100,
                marker="+",
                label="Calculated CCSD",
            )
            axs[0].plot(
                vdata,
                regression_line,
                linestyle="dashed",
                linewidth=2,
                label="Regression line",
            )

            axs[0].set_xlabel("Calculated CCSD energy [Hartree]")
            axs[0].set_ylabel("Predicted CCSD energy [Hartree]")
            axs[0].legend()
            axs[0].set_title(f"Regression - {name_wo_ext}")

            textstr = (
                f"R$^2$ = {r2:.2f}\n"
                f"MSE = {mse:.2e}\n"
                f"MAE = {mae_measured:.2e}\n"
                f"MAE window = {mae_window:.2e}\n"
                f"Std Err = {stderr:.2e}\n"
                f"Max |diff| = {max_abs_diff:.2e}"
            )

            axs[0].text(
                0.025,
                0.75,
                textstr,
                transform=axs[0].transAxes,
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            axs[1].hist(
                abs_diff,
                bins=20,
                alpha=0.7,
                label="|Predicted - Calculated|",
            )
            axs[1].axvline(
                x=mae_window,
                linestyle="--",
                label=f"MAE window = {mae_window:.2e}",
            )
            axs[1].set_title(f"Absolute error histogram - {name_wo_ext}")
            axs[1].set_xlabel("Absolute error [Hartree]")
            axs[1].set_ylabel("Frequency")
            axs[1].grid(True)
            axs[1].legend()

            plt.tight_layout()

            plot_path = os.path.join(
                OUTPUT_PLOTS_DIR,
                f"combined_plot_{name_wo_ext}.png",
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"[PLOT] Saved: {plot_path}")

            # -----------------------------
            # Range scan near minimum predicted CCSD energy
            # -----------------------------
            most_negative_value = float(np.min(pred_all))
            lower = most_negative_value - mae_window
            upper = most_negative_value + mae_window

            values_within = []
            dataset_row_indices = []

            for pred_line_index, value in enumerate(pred_all, start=1):
                if lower <= value <= upper:
                    dataset_row = pred_line_index + LINE_OFFSET - 1
                    values_within.append(float(value))
                    dataset_row_indices.append(dataset_row)

            hits_count = len(values_within)

            print(
                f"[{name_wo_ext}] Most negative: {most_negative_value:.10f} | "
                f"Window: [{lower:.10f}, {upper:.10f}] | Hits: {hits_count}"
            )

            metrics_file.write(
                f"{name_wo_ext}\t{model_tag}\t{r2:.6f}\t{mse:.10e}\t"
                f"{mae_measured:.10e}\t{stderr:.10e}\t{mae_window:.10e}\t"
                f"{most_negative_value:.10f}\t{lower:.10f}\t{upper:.10f}\t{hits_count}\n"
            )
            metrics_file.flush()

            if hits_count >= HITS_COPY_THRESHOLD:
                print(
                    f"[SKIP COPY] Hits {hits_count} >= {HITS_COPY_THRESHOLD}. "
                    f"No xyz files copied for {name_wo_ext}."
                )
                continue

            values_out = f"Values_Within_Range_{model_tag}_{LOW_LEVEL_LABEL}.txt"

            with open(values_out, "w") as outf:
                outf.write("dataset_row\tpredicted_ccsd\txyz_path\n")
                for dataset_row, value in zip(dataset_row_indices, values_within):
                    if 0 <= dataset_row < len(processed_lines):
                        xyz_path = processed_lines[dataset_row].strip()
                    else:
                        xyz_path = ""
                    outf.write(f"{dataset_row}\t{value:.10f}\t{xyz_path}\n")

            target_dir = os.path.join(FOUND_DIR, f"{model_tag}_{LOW_LEVEL_LABEL}")
            os.makedirs(target_dir, exist_ok=True)

            copied = 0
            for dataset_row in dataset_row_indices:
                if 0 <= dataset_row < len(processed_lines):
                    xyz_path = processed_lines[dataset_row].strip()
                    if safe_copy_structure(xyz_path, target_dir):
                        copied += 1

            print(f"[COPY] {copied} files copied to {target_dir} for {name_wo_ext}")

    print("\nDone.")
    print(f"Metrics written to: {METRICS_SUMMARY}")
    print(f"Plots written to:   {OUTPUT_PLOTS_DIR}")
    print(f"Structures copied to: {FOUND_DIR}")


if __name__ == "__main__":
    main()