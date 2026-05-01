import os
import re
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------
# User-configurable parameters
# ---------------------------------------
VALIDATION_FILE = "CCSD_ML_Final.txt"  # ground truth file
N_COMPARE = 100               # compare first N predictions vs last N validation values

PRED_PATTERNS = [
    "predicted_SOCASSI_GB_unseen_optuna_*.txt",
    "predicted_SOCASSI_GPR_unseen_optuna_*.txt",
    "predicted_SOCASSI_SVR_unseen_optuna_*.txt",
]

PROCESSED_XYZ_LIST = "processed_xyz_files_final.txt"  # one absolute xyz path per line

LINE_OFFSET = 800  # offset added to predicted-file line index to map into processed_xyz_files_final.txt

# Enter MAE manually here (like your second script). If None, the script will use the measured MAE.
MANUAL_MAE_FOR_WINDOW = None  # set to None to use measured MAE from regression

# Optional: gate plotting + range-scan by max abs diff in the N_COMPARE regression slice.
# Set to None to always plot/scan. If set (e.g. 1e-3), files exceeding it will be skipped.
MAX_ABS_DIFF_GATE = None  # e.g. 1.0e-3

OUTPUT_PLOTS_DIR = "output_combined_plots"
GENERIC_FOUND_DIR = "Found_Within_Range"
METRICS_SUMMARY = "all_regression_metrics.txt"

# Hits threshold for copying: if hits_count >= this, skip copying
HITS_COPY_THRESHOLD = 250
# ---------------------------------------

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
os.makedirs(GENERIC_FOUND_DIR, exist_ok=True)

def extract_number(filename: str) -> int:
    """Extract trailing integer for numeric sorting; non-matching files sort last."""
    m = re.search(r"(\d+)(?=\D*$)", filename)
    return int(m.group(1)) if m else 10**18

def load_first_column(path: str) -> np.ndarray:
    """Load numeric data; if multiple columns exist, take the first column."""
    arr = np.loadtxt(path, dtype=float)
    arr = np.atleast_1d(arr)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr

# Load validation data (last N_COMPARE)
validation_all = load_first_column(VALIDATION_FILE)
if validation_all.size < N_COMPARE:
    raise ValueError(f"{VALIDATION_FILE} has only {validation_all.size} values, need {N_COMPARE}.")
validation_data = validation_all[-N_COMPARE:]

# Load processed xyz file list once
with open(PROCESSED_XYZ_LIST, "r") as f:
    processed_lines = f.readlines()

# Collect predicted files
predicted_files = []
for patt in PRED_PATTERNS:
    predicted_files.extend(glob.glob(patt))
predicted_files = sorted(predicted_files, key=extract_number)

# Write metrics header once
with open(METRICS_SUMMARY, "w") as metrics_file:
    metrics_file.write("File Name\tModel\tR^2\tMSE\tMAE_measured\tStdErr\tMAE_window_used\n")
    metrics_file.write("=" * 100 + "\n")

    for pred_path in predicted_files:
        base = os.path.basename(pred_path)
        name_wo_ext = os.path.splitext(base)[0]

        # Extract model tag and suffix
        model_tag, suffix = "X", "X"
        m = re.search(r"predicted_SOCASSI_([A-Za-z0-9]+)_unseen_optuna_(\d+)", base)
        if m:
            model_tag, suffix = m.group(1), m.group(2)

        # Load predictions
        try:
            pred_all = load_first_column(pred_path)
        except Exception as e:
            print(f"[SKIP] Could not load {pred_path}: {e}")
            continue

        if pred_all.size < N_COMPARE:
            print(f"[SKIP] {base}: only {pred_all.size} predictions, need {N_COMPARE}.")
            continue

        predicted_data = pred_all[:N_COMPARE].copy()
        vdata = validation_data.copy()

        # Regression fit (Pred vs Calc)
        if len(vdata) < 2:
            print(f"[SKIP] {base}: not enough points for regression.")
            continue

        coeffs = np.polyfit(vdata, predicted_data, 1)
        regression_line = np.polyval(coeffs, vdata)

        # Metrics
        r2 = r2_score(vdata, predicted_data)
        mse = mean_squared_error(vdata, predicted_data)
        mae_measured = mean_absolute_error(vdata, predicted_data)

        residuals = predicted_data - regression_line
        se = np.sqrt(np.sum(residuals ** 2) / max(1, (len(vdata) - 2)))

        # Use manual MAE if provided, else use measured MAE
        mae_window = float(MANUAL_MAE_FOR_WINDOW) if MANUAL_MAE_FOR_WINDOW is not None else float(mae_measured)

        # Save metrics row
        metrics_file.write(
            f"{name_wo_ext}\t{model_tag}\t{r2:.4f}\t{mse:.6e}\t{mae_measured:.6e}\t{se:.6e}\t{mae_window:.6e}\n"
        )
        metrics_file.flush()

        # Abs differences on the regression slice
        abs_diff = np.abs(predicted_data - vdata)
        max_abs_diff = float(np.max(abs_diff))

        if MAX_ABS_DIFF_GATE is not None and max_abs_diff > MAX_ABS_DIFF_GATE:
            print(
                f"[SKIP] {base}: max abs diff {max_abs_diff:.3e} > gate {MAX_ABS_DIFF_GATE:.3e}. "
                f"Skipping plot and range-scan."
            )
            continue

        # ---- Plot combined figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].scatter(vdata, predicted_data, s=100, marker="x",
                       label=f"Predicted ({model_tag})")
        axs[0].scatter(vdata, vdata, s=100, marker="+",
                       label="Calculated (identity)")
        axs[0].plot(vdata, regression_line, linestyle="dashed", linewidth=2,
                    label="Regression line")

        axs[0].set_xlabel("Calculated SOCASSI energy [Hartree/a.u.]")
        axs[0].set_ylabel("Predicted SOCASSI energy [Hartree/a.u.]")
        axs[0].legend()
        axs[0].set_title(f"Regression - {name_wo_ext}")

        textstr = (
            f"R^2 = {r2:.2f}\n"
            f"MSE = {mse:.2e}\n"
            f"MAE(measured) = {mae_measured:.2e}\n"
            f"MAE(window) = {mae_window:.2e}\n"
            f"Std Err = {se:.2e}\n"
            f"Max |diff| = {max_abs_diff:.2e}"
        )
        axs[0].text(
            0.025, 0.75, textstr, transform=axs[0].transAxes,
            fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        axs[1].hist(abs_diff, bins=20, alpha=0.7, label="|Pred - Calc| (slice)")
        if MAX_ABS_DIFF_GATE is not None:
            axs[1].axvline(x=MAX_ABS_DIFF_GATE, linestyle="--",
                           label=f"Gate = {MAX_ABS_DIFF_GATE}")
        axs[1].axvline(x=mae_window, linestyle="--",
                       label=f"MAE(window) = {mae_window:.2e}")
        axs[1].set_title(f"Abs diff histogram - {name_wo_ext}")
        axs[1].set_xlabel("Absolute Difference [Hartree/a.u.]")
        axs[1].set_ylabel("Frequency")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_PLOTS_DIR, f"combined_plot_{name_wo_ext}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[PLOT] Saved: {plot_path}")

        # ---- Range-scan on the whole prediction file using MAE window
        most_negative_value = float(np.min(pred_all))
        lower = most_negative_value - mae_window
        upper = most_negative_value + mae_window

        values_within = []
        line_numbers = []

        with open(pred_path, "r") as f:
            for i, line in enumerate(f, start=1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    v = float(parts[0])
                except ValueError:
                    continue
                if lower <= v <= upper:
                    values_within.append(v)
                    line_numbers.append(i + LINE_OFFSET)

        hits_count = len(values_within)
        print(
            f"[{name_wo_ext}] Most negative: {most_negative_value:.8f} | "
            f"Window: [{lower:.8f}, {upper:.8f}] | Hits: {hits_count}"
        )

        if hits_count >= HITS_COPY_THRESHOLD:
            print(f"[SKIP COPY] Hits {hits_count} >= {HITS_COPY_THRESHOLD}. No files copied for {name_wo_ext}.")
            continue

        values_out = f"Values_Within_Range_{model_tag}_unseen_{suffix}.txt"
        with open(values_out, "w") as outf:
            for ln, v in zip(line_numbers, values_within):
                path_line = processed_lines[ln - 1] if 1 <= ln <= len(processed_lines) else "\n"
                outf.write(f"{ln}\t{v}\t{path_line}")

        target_dir = os.path.join(GENERIC_FOUND_DIR, f"{model_tag}_{suffix}")
        os.makedirs(target_dir, exist_ok=True)

        copied = 0
        for ln in line_numbers:
            if 1 <= ln <= len(processed_lines):
                xyz_path = processed_lines[ln - 1].strip()
                if xyz_path:
                    try:
                        shutil.copy2(xyz_path, os.path.join(target_dir, os.path.basename(xyz_path)))
                        copied += 1
                    except Exception as e:
                        print(f"[WARN] Copy failed for {xyz_path}: {e}")

        print(f"[COPY] {copied} files copied to {target_dir} for {name_wo_ext}")
