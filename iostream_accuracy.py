"""
iostream_accuracy.py
====================
Accuracy verification for myproject_aximm on Alveo U280.

This kernel uses AXI Master (m_axi) ports and HBM buffers. The entire test
set is batched into a single kernel invocation — far more efficient than the
one-sample-at-a-time AXI-Lite approach in fpga_accuracy.py.

Kernel signature:
  myproject_aximm(ap_fixed<16,6>* input_mem,   // M_AXI_GMEM0 → HBM[0]
                  ap_fixed<16,6>* output_mem,  // M_AXI_GMEM1 → HBM[0]
                  unsigned int    n_samples)

Fixed-point encoding:
  ap_fixed<16,6>: 16-bit signed, 6 integer bits, 10 fractional bits
  float → int16:  round(f * 1024)
  int16 → float:  i / 1024.0

Usage:
    python iostream_accuracy.py
    python iostream_accuracy.py --samples 5000
    python iostream_accuracy.py --no_sw
"""

import os, sys, argparse, time
sys.path.insert(0, "/opt/xilinx/xrt/python")
sys.path.insert(0, "/home/conor/Proj_New_515/Sem2Linux_lean/.venv/lib/python3.10/site-packages")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import pyxrt
except ImportError:
    raise ImportError("XRT Python bindings not found. Run: source /opt/xilinx/xrt/setup.sh")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
XCLBIN_PATH = os.path.join(_HERE,
    "my_vitis_project_RF10_IOSTREAM_PYTHON/myproject_aximm.xclbin")
KERAS_MODEL = os.path.join(_HERE, "my_vitis_project_RF10/keras_model.keras")
TEST_CSV    = os.path.join(_HERE,
    "Mathys_Pradel_Work/RB3_Option_pricing_mathys_Pradels/scripts_v4/Data/testing_set.csv")
OUTPUT_DIR  = os.path.join(_HERE, "results/accuracy")

KERNEL_NAME = "myproject_aximm"
N_FEATURES  = 6
SCALE       = 1024   # 2^10  (ap_fixed<16,6> has 10 fractional bits)

# ── Fixed-point helpers ────────────────────────────────────────────────────────
def encode(arr: np.ndarray) -> np.ndarray:
    """Float32 array → ap_fixed<16,6> as int16."""
    q = np.round(arr.astype(np.float64) * SCALE).astype(np.int32)
    np.clip(q, -(1 << 15), (1 << 15) - 1, out=q)
    return q.astype(np.int16)

def decode(arr: np.ndarray) -> np.ndarray:
    """ap_fixed<16,6> int16 array → float32."""
    return arr.astype(np.float32) / SCALE

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label):
    mse  = float(mean_squared_error(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))
    print(f"\n[ACCURACY]  ── {label} ──")
    print(f"[ACCURACY]  MAE  : {mae:.6f}")
    print(f"[ACCURACY]  MSE  : {mse:.6f}")
    print(f"[ACCURACY]  RMSE : {rmse:.6f}")
    print(f"[ACCURACY]  R²   : {r2:.6f}")
    return {"Label": label, "MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--no_sw",   action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path, name in [(XCLBIN_PATH, "xclbin"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # ── Load test data ─────────────────────────────────────────────────────────
    print("[ACCURACY]  Loading test data...")
    df = pd.read_csv(TEST_CSV)
    X  = df.drop("Call Price", axis=1).values.astype(np.float32)
    Y  = df["Call Price"].values.astype(np.float32)
    if args.samples:
        X, Y = X[:args.samples], Y[:args.samples]
    N = len(X)
    feature_cols = list(df.drop("Call Price", axis=1).columns)
    print(f"[ACCURACY]  Samples  : {N:,}")
    print(f"[ACCURACY]  Features : {feature_cols}")
    print(f"[ACCURACY]  Input range: [{X.min():.4f}, {X.max():.4f}]")

    # ── Fixed-point encode ─────────────────────────────────────────────────────
    # Flatten to (N*6,) row-major so kernel reads input_mem[s*6 + i]
    X_q = encode(X).flatten()   # shape (N*6,), dtype int16

    # ── Initialise FPGA ────────────────────────────────────────────────────────
    print(f"\n[ACCURACY]  Loading xclbin...")
    device = pyxrt.device(args.device)
    xclbin = pyxrt.xclbin(XCLBIN_PATH)
    device.register_xclbin(xclbin)
    kernel = pyxrt.kernel(device, xclbin.get_uuid(), KERNEL_NAME)
    print(f"[ACCURACY]  Kernel  : {KERNEL_NAME}  |  Clock: 100 MHz")

    # ── Allocate HBM buffers ───────────────────────────────────────────────────
    in_bytes  = int(N * N_FEATURES * 2)   # int16 = 2 bytes
    out_bytes = int(N * 1 * 2)
    in_bo  = pyxrt.bo(device, in_bytes,  pyxrt.bo.flags.normal, kernel.group_id(0))
    out_bo = pyxrt.bo(device, out_bytes, pyxrt.bo.flags.normal, kernel.group_id(1))

    # ── Transfer input to HBM ──────────────────────────────────────────────────
    in_bo.write(X_q.tobytes(), 0)
    in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # ── Warm-up run ────────────────────────────────────────────────────────────
    print("[ACCURACY]  Warm-up run (small batch)...")
    wn = min(100, N)
    run = kernel(in_bo, out_bo, wn)
    run.wait()

    # ── Timed inference ────────────────────────────────────────────────────────
    print(f"[ACCURACY]  Running full inference ({N:,} samples)...")
    t_start = time.perf_counter()
    run = kernel(in_bo, out_bo, N)
    run.wait()
    t_kernel = time.perf_counter() - t_start

    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    t_total = time.perf_counter() - t_start

    raw = out_bo.read(out_bytes, 0)
    Y_fpga = decode(np.frombuffer(raw, dtype=np.int16))

    print(f"[ACCURACY]  Kernel time  : {t_kernel*1e3:.2f} ms")
    print(f"[ACCURACY]  Total (+ DMA): {t_total*1e3:.2f} ms  "
          f"({t_total/N*1e6:.3f} µs/sample)")

    # ── Keras software baseline ────────────────────────────────────────────────
    Y_sw = None
    if not args.no_sw and os.path.exists(KERAS_MODEL):
        print("\n[ACCURACY]  Running Keras float32 baseline...")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        model = tf.keras.models.load_model(KERAS_MODEL)
        Y_sw  = model.predict(X, batch_size=1024, verbose=0).flatten().astype(np.float32)
    elif not args.no_sw:
        print(f"[ACCURACY]  Keras model not found at {KERAS_MODEL} — skipping.")

    # ── Metrics (full dataset) ─────────────────────────────────────────────────
    rows = [compute_metrics(Y, Y_fpga,
                            "FPGA — myproject_aximm @ 100 MHz (ap_fixed<16,6>)")]
    if Y_sw is not None:
        rows.append(compute_metrics(Y, Y_sw, "Software — Keras float32"))
        rows.append(compute_metrics(Y_sw, Y_fpga, "FPGA vs SW (quantisation error)"))

    # ── Filtered metrics (exclude overflow samples) ────────────────────────────
    # ap_fixed<16,6> represents values in ~[-32, 32).  The 37 samples with
    # Stock Price / Call Price > 31 overflow the fixed-point representation and
    # produce large errors that dominate the global MSE.  Report metrics on the
    # representable (in-range) subset for a fair characterisation.
    AP_MAX = 31.0
    in_range = (np.abs(X).max(axis=1) <= AP_MAX) & (np.abs(Y) <= AP_MAX)
    n_overflow = int((~in_range).sum())
    if n_overflow > 0:
        rows_filt = [compute_metrics(Y[in_range], Y_fpga[in_range],
                                     f"FPGA — in-range samples only ({in_range.sum():,}/{N:,}, "
                                     f"{n_overflow} overflow excluded)")]
        if Y_sw is not None:
            rows_filt.append(compute_metrics(Y[in_range], Y_sw[in_range],
                                             "Software — in-range samples only"))
        print(f"\n[ACCURACY]  Note: {n_overflow} samples ({100*n_overflow/N:.2f}%) have features or "
              f"labels > {AP_MAX} (overflow ap_fixed<16,6>). "
              f"In-range R² is the representative metric.")
    else:
        rows_filt = []

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    pd.DataFrame(rows + rows_filt).to_csv(
        os.path.join(OUTPUT_DIR, "iostream_accuracy_metrics.csv"), index=False)

    detail = {"y_true": Y, "y_fpga": Y_fpga}
    if Y_sw is not None:
        detail["y_software"] = Y_sw
    pd.DataFrame(detail).to_csv(
        os.path.join(OUTPUT_DIR, "iostream_predictions_detail.csv"), index=False)

    best_row = rows_filt[0] if rows_filt else rows[0]
    pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R²", "MAE_inrange", "R2_inrange",
                   "N_overflow", "Kernel_ms", "Total_ms", "Avg_us_per_sample", "N_samples"],
        "Value":  [rows[0]["MAE"], rows[0]["MSE"], rows[0]["RMSE"], rows[0]["R²"],
                   best_row["MAE"], best_row["R²"],
                   n_overflow if n_overflow > 0 else 0,
                   t_kernel*1e3, t_total*1e3, t_total/N*1e6, N]
    }).to_csv(os.path.join(OUTPUT_DIR, "iostream_FPGA_metrics.csv"), index=False)

    # ── Plot (in-range samples for clarity) ────────────────────────────────────
    ncols = 2 if Y_sw is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    Yplot  = Y[in_range]  if n_overflow > 0 else Y
    Yfplot = Y_fpga[in_range] if n_overflow > 0 else Y_fpga
    Ysplot = Y_sw[in_range]   if (Y_sw is not None and n_overflow > 0) else Y_sw
    suffix = f" (in-range, N={in_range.sum():,})" if n_overflow > 0 else ""
    for ax, y_pred, label, colour in zip(
            axes,
            [Yfplot] + ([Ysplot] if Y_sw is not None else []),
            [f"FPGA (myproject_aximm @ 100 MHz){suffix}"] +
            ([f"Software (Keras){suffix}"] if Y_sw is not None else []),
            ["steelblue", "darkorange"]):
        ax.scatter(Yplot, y_pred, alpha=0.3, s=2, color=colour)
        lims = [min(Yplot.min(), y_pred.min()), max(Yplot.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="Ideal")
        ax.set_xlabel("Ground Truth — Call Price")
        ax.set_ylabel(f"{label}")
        ax.set_title(f"Predictions vs Ground Truth\n{label}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "iostream_prediction_vs_truth.png"), dpi=150)
    plt.close()
    print(f"\n[ACCURACY]  Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
