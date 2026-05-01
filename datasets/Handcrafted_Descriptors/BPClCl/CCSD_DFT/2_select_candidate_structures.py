#!/usr/bin/env python3

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =====================================================
# User controls
# =====================================================

LOW_LEVEL_LABEL = "DFT"

VALIDATION_FILE = "CCSD.txt"
PROCESSED_XYZ_LIST = "processed_xyz_files.txt"

# Prediction line 1 corresponds to dataset row N_TRAIN.
# For this dataset:
#   N_TRAIN = 1027
#   N_CCSD = 1127
#   prediction line 1   -> dataset row 1027
#   prediction line 100 -> dataset row 1126
LINE_OFFSET = 1027

# Compare first 100 prediction rows against CCSD rows 1027:1127.
N_COMPARE = 100

PREDICTION_FILES = {
    "GB":  "Predictions/predicted_ccsd_from_dft_gbr.txt",
    "GPR": "Predictions/predicted_ccsd_from_dft_gpr.txt",
    "SVR": "Predictions/predicted_ccsd_from_dft_svr.txt",
}

# If None, measured MAE from the comparison block is used as the energy window.
MANUAL_MAE_FOR_WINDOW = None

# Set to None to always plot/scan.
# Example: 1.0e-3 skips files whose max abs diff exceeds 1e-3 Hartree.
MAX_ABS_DIFF_GATE = None

OUTPUT_PLOTS_DIR = "Plots_HeldOut_Set"
FOUND_DIR = "Found_Within_MAE_Range"
METRICS_SUMMARY = "all_regression_metrics.txt"

# If too many structures fall in the minimum-energy window, skip copying.
HITS_COPY_THRESHOLD = 200


# =====================================================
# Helpers
# =====================================================

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_first_column(path):
    arr = np.loadtxt(path, dtype=float)
    arr = np.atleast_1d(arr)

    if arr.ndim == 2:
        arr = arr[:, 0]

    return arr


def safe_copy_structure(xyz_path, target_dir):
    """
    Copy xyz_path to target_dir.

    The script first tries the path exactly as listed in processed_xyz_files.txt.
    If that path does not exist, it tries Structures/<basename>.
    """
    xyz_path = xyz_path.strip()

    if not xyz_path:
        return False

    source = xyz_path

    if not os.path.exists(source):
        basename = os.path.basename(source)
        candidate = os.path.join("Structures", basename)

        if os.path.exists(candidate):
            source = candidate
        else:
            print(f"[WARN] Could not find xyz file: {xyz_path}")
            return False

    shutil.copy2(source, os.path.join(target_dir, os.path.basename(source)))
    return True


def collect_prediction_files():
    files = []

    for model_tag, path in PREDICTION_FILES.items():
        require_file(path)
        files.append((model_tag, path))

    return files


# =====================================================
# Main
# =====================================================

def main():
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    os.makedirs(FOUND_DIR, exist_ok=True)

    require_file(VALIDATION_FILE)
    require_file(PROCESSED_XYZ_LIST)

    prediction_files = collect_prediction_files()

    print("Using files:")
    print(f"  Validation CCSD file: {VALIDATION_FILE}")
    print(f"  XYZ list:             {PROCESSED_XYZ_LIST}")

    print("\nPrediction files:")
    for model_tag, pred_path in prediction_files:
        print(f"  {model_tag}: {pred_path}")

    validation_all = load_first_column(VALIDATION_FILE)

    if validation_all.size < LINE_OFFSET + N_COMPARE:
        raise ValueError(
            f"{VALIDATION_FILE} has {validation_all.size} values, but need at least "
            f"{LINE_OFFSET + N_COMPARE} values for rows "
            f"{LINE_OFFSET}:{LINE_OFFSET + N_COMPARE}."
        )

    validation_data = validation_all[LINE_OFFSET:LINE_OFFSET + N_COMPARE]

    with open(PROCESSED_XYZ_LIST, "r") as f:
        processed_lines = f.readlines()

    if len(processed_lines) < LINE_OFFSET + 1:
        raise ValueError(
            f"{PROCESSED_XYZ_LIST} has only {len(processed_lines)} rows, "
            f"but LINE_OFFSET={LINE_OFFSET}."
        )

    with open(METRICS_SUMMARY, "w") as metrics_file:
        metrics_file.write(
            "File_Name\tModel\tR2\tMSE\tMAE_measured\tStdErr\t"
            "MAE_window_used\tMost_negative\tLower_window\tUpper_window\tHits\n"
        )

        for model_tag, pred_path in prediction_files:
            base = os.path.basename(pred_path)
            name_wo_ext = os.path.splitext(base)[0]

            pred_all = load_first_column(pred_path)

            if pred_all.size < N_COMPARE:
                print(
                    f"[SKIP] {base}: only {pred_all.size} predictions, "
                    f"need {N_COMPARE}."
                )
                continue

            if len(processed_lines) < LINE_OFFSET + pred_all.size:
                raise ValueError(
                    f"{PROCESSED_XYZ_LIST} has only {len(processed_lines)} rows, "
                    f"but predictions in {base} require rows up to "
                    f"{LINE_OFFSET + pred_all.size - 1}."
                )

            predicted_data = pred_all[:N_COMPARE].copy()
            vdata = validation_data.copy()

            coeffs = np.polyfit(vdata, predicted_data, 1)
            regression_line = np.polyval(coeffs, vdata)

            r2 = r2_score(vdata, predicted_data)
            mse = mean_squared_error(vdata, predicted_data)
            mae_measured = mean_absolute_error(vdata, predicted_data)

            residuals = predicted_data - regression_line
            stderr = np.sqrt(np.sum(residuals ** 2) / max(1, len(vdata) - 2))

            if MANUAL_MAE_FOR_WINDOW is not None:
                mae_window = float(MANUAL_MAE_FOR_WINDOW)
            else:
                mae_window = float(mae_measured)

            abs_diff = np.abs(predicted_data - vdata)
            max_abs_diff = float(np.max(abs_diff))

            if MAX_ABS_DIFF_GATE is not None and max_abs_diff > MAX_ABS_DIFF_GATE:
                print(
                    f"[SKIP] {base}: max abs diff {max_abs_diff:.3e} "
                    f"> gate {MAX_ABS_DIFF_GATE:.3e}. "
                    f"Skipping plot and range scan."
                )
                continue

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
                f"{name_wo_ext}\t{model_tag}\t{r2:.6f}\t"
                f"{mse:.10e}\t{mae_measured:.10e}\t{stderr:.10e}\t"
                f"{mae_window:.10e}\t{most_negative_value:.10f}\t"
                f"{lower:.10f}\t{upper:.10f}\t{hits_count}\n"
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
                    xyz_path = processed_lines[dataset_row].strip()
                    outf.write(f"{dataset_row}\t{value:.10f}\t{xyz_path}\n")

            target_dir = os.path.join(FOUND_DIR, f"{model_tag}_{LOW_LEVEL_LABEL}")
            os.makedirs(target_dir, exist_ok=True)

            copied = 0

            for dataset_row in dataset_row_indices:
                xyz_path = processed_lines[dataset_row].strip()

                if safe_copy_structure(xyz_path, target_dir):
                    copied += 1

            print(f"[COPY] {copied} files copied to {target_dir} for {name_wo_ext}")

    print("\nDone.")
    print(f"Metrics written to:     {METRICS_SUMMARY}")
    print(f"Plots written to:       {OUTPUT_PLOTS_DIR}")
    print(f"Structures copied to:   {FOUND_DIR}")


if __name__ == "__main__":
    main()