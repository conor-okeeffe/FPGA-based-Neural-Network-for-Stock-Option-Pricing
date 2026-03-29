"""
fpga_compatible_accuracy.py
===========================
Accuracy comparison on the FPGA-compatible subset of the test data.

The hls4ml kernel uses ap_fixed<16,6> for all inputs, which represents values
in the range [-32, 32).  Two features in the original test set — Stock Price
and Dividends — have 37 samples that exceed this range.  These cause silent
fixed-point overflow on the FPGA, producing catastrophically wrong outputs.

This script:
  1. Loads the full test set and identifies the 37 overflow samples.
  2. Filters to the 29,564 FPGA-compatible samples (all features < 32).
  3. Runs the Keras float32 model on the filtered set.
  4. Loads the FPGA predictions from the last fpga_accuracy.py run and
     restricts them to the same filtered set.
  5. Reports and plots three-way metrics:
       - Keras float32  (software reference)
       - FPGA ap_fixed  (measured on hardware)
       - FPGA vs Keras  (quantisation error: what hls4ml adds vs float32)

Outputs:
  results/compatible_accuracy/compatible_metrics.csv
  results/compatible_accuracy/predictions_detail.csv
  results/compatible_accuracy/predictions_vs_truth.png
  results/compatible_accuracy/error_distribution.png

Usage:
    python fpga_compatible_accuracy.py
    python fpga_compatible_accuracy.py --no_fpga   # Keras only, no FPGA preds
"""

import os, sys, types, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Python 3.10 ships without distutils; TF 2.14 tries to import distutils.spawn
# inside tensorflow.lite which is loaded at import time.  Stub it out early.
if "distutils.spawn" not in sys.modules:
    sys.modules["distutils.spawn"] = types.ModuleType("distutils.spawn")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL  = os.path.join(_HERE, "my_vitis_project_RF10/keras_model.keras")
TEST_CSV     = os.path.join(_HERE,
               "Mathys_Pradel_Work/RB3_Option_pricing_mathys_Pradels/scripts_v4/Data/testing_set.csv")
FPGA_PREDS   = os.path.join(_HERE, "results/accuracy/predictions_detail.csv")
OUTPUT_DIR   = os.path.join(_HERE, "results/compatible_accuracy")

# ap_fixed<16,6>: signed 16-bit, 6 integer bits → range [-32, 32)
AP_FIXED_MAX = 32.0


def compute_metrics(y_true, y_pred, label):
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))
    print(f"\n  ── {label} ──")
    print(f"     MAE  : {mae:.6f}")
    print(f"     MSE  : {mse:.6f}")
    print(f"     RMSE : {rmse:.6f}")
    print(f"     R²   : {r2:.6f}")
    return {"Label": label, "N": len(y_true),
            "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    parser = argparse.ArgumentParser(
        description="Accuracy comparison on FPGA-compatible (in-range) samples")
    parser.add_argument("--no_fpga", action="store_true",
                        help="Skip FPGA predictions (Keras only)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load test data ────────────────────────────────────────────────────────
    print("[COMPAT]  Loading test data...")
    df   = pd.read_csv(TEST_CSV)
    feat_cols = [c for c in df.columns if c != "Call Price"]
    X    = df[feat_cols].values.astype(np.float32)
    Y    = df["Call Price"].values.astype(np.float32)
    N_total = len(Y)

    # ── Identify FPGA-compatible samples ─────────────────────────────────────
    # ap_fixed<16,6> range: [-32, 32).  Any feature with |x| >= 32 will overflow.
    in_range = np.all(np.abs(X) < AP_FIXED_MAX, axis=1)
    N_compat = int(in_range.sum())
    N_drop   = N_total - N_compat

    print(f"[COMPAT]  Total test samples   : {N_total}")
    print(f"[COMPAT]  FPGA-compatible (<32) : {N_compat}  ({100*N_compat/N_total:.2f}%)")
    print(f"[COMPAT]  Dropped (overflow)    : {N_drop}  ({100*N_drop/N_total:.3f}%)")
    print(f"\n[COMPAT]  Overflow breakdown by feature:")
    for i, col in enumerate(feat_cols):
        n = int((np.abs(X[:, i]) >= AP_FIXED_MAX).sum())
        if n > 0:
            print(f"           {col:20s}: {n} samples  "
                  f"(max={X[:,i].max():.2f}, limit=±{AP_FIXED_MAX:.0f})")

    X_c = X[in_range]
    Y_c = Y[in_range]
    compat_idx = np.where(in_range)[0]

    # ── Run Keras float32 model ───────────────────────────────────────────────
    for path, name in [(KERAS_MODEL, "Keras model"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    print(f"\n[COMPAT]  Loading Keras model: {KERAS_MODEL}")
    import tensorflow as tf
    model = tf.keras.models.load_model(KERAS_MODEL)
    model.summary(print_fn=lambda s: print(f"[COMPAT]  {s}"))

    print(f"\n[COMPAT]  Running Keras float32 inference on {N_compat} compatible samples...")
    t0    = time.perf_counter()
    Y_sw  = model.predict(X_c, batch_size=1024, verbose=0).flatten().astype(np.float32)
    t_sw  = time.perf_counter() - t0
    print(f"[COMPAT]  Keras inference: {t_sw:.3f} s  "
          f"({t_sw/N_compat*1e6:.2f} µs/sample  =  {N_compat/t_sw/1e3:.1f} ksamples/s)")

    # ── Load FPGA predictions ─────────────────────────────────────────────────
    Y_fpga = None
    if not args.no_fpga:
        if os.path.exists(FPGA_PREDS):
            print(f"\n[COMPAT]  Loading FPGA predictions: {FPGA_PREDS}")
            fpga_df = pd.read_csv(FPGA_PREDS)
            # predictions_detail.csv has 29601 rows aligned with test set
            Y_fpga_all = fpga_df["y_fpga"].values.astype(np.float32)
            Y_fpga     = Y_fpga_all[in_range]
            print(f"[COMPAT]  Loaded {len(Y_fpga)} FPGA predictions for compatible samples")
        else:
            print(f"[COMPAT]  Warning: FPGA predictions not found at {FPGA_PREDS}")
            print(f"[COMPAT]  Run fpga_accuracy.py first, or use --no_fpga")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f" ACCURACY — FPGA-compatible subset  (N={N_compat} / {N_total} samples)")
    print(f" Filter: all features in [-32, 32)  (ap_fixed<16,6> representable)")
    print("="*65)

    rows = [compute_metrics(Y_c, Y_sw,
                            f"Keras float32 software  (on {N_compat} compatible samples)")]
    if Y_fpga is not None:
        rows.append(compute_metrics(Y_c, Y_fpga,
                                    f"FPGA ap_fixed<16,6>  (Alveo U280 @ 181 MHz)"))
        rows.append(compute_metrics(Y_sw, Y_fpga,
                                    "FPGA vs Keras  (fixed-point quantisation error only)"))

    # ── Also report full-set Keras metrics for context ────────────────────────
    print(f"\n[COMPAT]  ── Full test set Keras metrics (all {N_total} samples, for context) ──")
    Y_sw_all = model.predict(X, batch_size=1024, verbose=0).flatten().astype(np.float32)
    rows.append(compute_metrics(Y, Y_sw_all,
                                f"Keras float32 software  (ALL {N_total} samples)"))

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "compatible_metrics.csv"), index=False)

    pred_cols = {
        "sample_index": compat_idx,
        "y_true":       Y_c,
        "y_keras":      Y_sw,
    }
    if Y_fpga is not None:
        pred_cols["y_fpga"] = Y_fpga
        pred_cols["err_keras_vs_fpga"] = Y_sw - Y_fpga
    pd.DataFrame(pred_cols).to_csv(
        os.path.join(OUTPUT_DIR, "predictions_detail.csv"), index=False)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    pairs = [("y_keras", Y_sw, "Keras float32", "steelblue")]
    if Y_fpga is not None:
        pairs.append(("y_fpga", Y_fpga, "FPGA ap_fixed<16,6>", "darkorange"))

    ncols = len(pairs)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, (_, y_pred, label, col) in zip(axes, pairs):
        mae = float(mean_absolute_error(Y_c, y_pred))
        r2  = float(r2_score(Y_c, y_pred))
        ax.scatter(Y_c, y_pred, alpha=0.25, s=2, color=col)
        mn, mx = Y_c.min(), Y_c.max()
        ax.plot([mn, mx], [mn, mx], "r--", lw=1, label="Ideal")
        ax.set_xlabel("True Call Price")
        ax.set_ylabel(f"{label} — Predicted")
        ax.set_title(f"{label}\nN={N_compat}  |  MAE={mae:.4f}  |  R²={r2:.4f}")
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Predictions vs Ground Truth — FPGA-compatible subset (N={N_compat})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "predictions_vs_truth.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # ── Error distribution plot ───────────────────────────────────────────────
    if Y_fpga is not None:
        err_keras = Y_sw  - Y_c
        err_fpga  = Y_fpga - Y_c
        err_quant = Y_fpga - Y_sw

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        datasets = [
            (err_keras, "Keras error (pred − true)",  "steelblue"),
            (err_fpga,  "FPGA error (pred − true)",   "darkorange"),
            (err_quant, "Quantisation error (FPGA − Keras)", "green"),
        ]
        for ax, (err, lbl, col) in zip(axes, datasets):
            clip = float(np.percentile(np.abs(err), 99))
            ax.hist(err[np.abs(err) <= clip], bins=100, color=col, alpha=0.75, edgecolor="none")
            ax.axvline(0, color="black", lw=0.8)
            ax.axvline(float(np.median(err)), color="red",   lw=1.2, linestyle="--",
                       label=f"Median: {np.median(err):.4f}")
            ax.axvline(float(err.mean()),     color="green", lw=1.2, linestyle="--",
                       label=f"Mean: {err.mean():.4f}")
            ax.set_xlabel("Error")
            ax.set_ylabel("Count")
            ax.set_title(lbl)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Error Distributions — FPGA-compatible subset (N={N_compat})",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

    print(f"\n[COMPAT]  Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
