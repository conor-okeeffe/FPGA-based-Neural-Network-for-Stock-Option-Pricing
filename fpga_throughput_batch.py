"""
fpga_throughput_batch.py
========================
Batch throughput benchmarking for the hls4ml neural network on Alveo U280.

Tests batch sizes 1, 100, 1000, 10000 — identical to those used in Mathys
Pradel's CPU_INFERENCE.py and NPU_INFERENCE.py scripts — enabling direct
performance comparison between the Alveo U280 FPGA and the Qualcomm RB3 Gen2.

Since this kernel processes one sample per invocation (no batch hardware),
batch throughput is achieved by running N sequential inferences.
The pipeline initiation interval (II=10 cycles @ 181 MHz) sets the theoretical
maximum: 1 / (10 × 5.525 ns) ≈ 18.1 M samples/s if communication overhead
were zero.

Per-batch outputs (matching Mathys's format exactly):
  results/throughput/FPGA_{N}/{N}_metrics.csv
  results/throughput/FPGA_{N}/predictions_detail.csv
Summary:
  results/throughput/throughput_summary.csv
  results/throughput/throughput_scaling.png

Usage:
    python fpga_throughput_batch.py
    python fpga_throughput_batch.py --batch_sizes 1 100 1000 10000 --repeats 5
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import ctypes
import uuid as _uuidmod

sys.path.insert(0, "/opt/xilinx/xrt/python")
try:
    import pyxrt
    from xrt_binding import (xclOpen, xclClose, xclLoadXclBin,
                               xclIPName2Index, xclOpenContext,
                               xclRegRead, xclRegWrite, xclVerbosityLevel)
except ImportError:
    raise ImportError(
        "XRT Python bindings not found.\n"
        "Run: source /opt/xilinx/xrt/setup.sh"
    )

_h  = None
_ci = None

def _wr(off, val): xclRegWrite(_h, _ci, off, val)
def _rr(off):
    d = ctypes.c_uint(0); xclRegRead(_h, _ci, off, ctypes.byref(d)); return d.value

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
XCLBIN_PATH = os.path.join(_HERE, "my_vitis_project_RF10/myproject_prj/solution2/export.xclbin")
TEST_CSV    = os.path.join(_HERE,
              "Mathys_Pradel_Work/RB3_Option_pricing_mathys_Pradels/scripts_v4/Data/testing_set.csv")
OUTPUT_DIR  = os.path.join(_HERE, "results/throughput")

# ── Kernel / CU identity ──────────────────────────────────────────────────────
CU_NAME = b"myproject:myproject_1"

# ── Hardware reference ────────────────────────────────────────────────────────
KERNEL_FREQ_MHZ    = 181.0
CYCLE_NS           = 1e3 / KERNEL_FREQ_MHZ         # 5.525 ns
PIPELINE_II        = 10                             # initiation interval (cycles)
LATENCY_WC_CYCLES  = 51                             # worst-case HLS latency
THEORETICAL_MAX_TP = 1e9 / (PIPELINE_II * CYCLE_NS) # samples/s, compute-only

# ── AXI-Lite Register Map ─────────────────────────────────────────────────────
REG_CTRL       = 0x00
REG_INPUT_0    = 0x10
REG_INPUT_1    = 0x14
REG_INPUT_2    = 0x18
REG_INPUT_VLD  = 0x1C
REG_OUTPUT_LO  = 0x20
REG_OUTPUT_HI  = 0x24
REG_OUTPUT_VLD = 0x28

AP_START = 0x1
AP_DONE  = 0x2

# ── Fixed-point helpers ───────────────────────────────────────────────────────
FRAC_IN  = 10
FRAC_OUT = 20

def float_to_ap16_6(v: float) -> int:
    s = int(round(v * (1 << FRAC_IN)))
    s = max(-(1 << 15), min((1 << 15) - 1, s))
    return s & 0xFFFF

def pack_inputs(x) -> tuple:
    e = [float_to_ap16_6(float(v)) for v in x]
    return (e[1] << 16) | e[0], (e[3] << 16) | e[2], (e[5] << 16) | e[4]

def unpack_output(lo: int, hi: int) -> float:
    int40 = (int(hi & 0xFF) << 32) | int(lo & 0xFFFFFFFF)
    if int40 >= (1 << 39):
        int40 -= (1 << 40)
    return int40 / (1 << FRAC_OUT)

# ── Batch runner ──────────────────────────────────────────────────────────────
def run_batch(X_batch: np.ndarray) -> np.ndarray:
    """
    Run a batch of N samples sequentially through the FPGA kernel.
    Returns predicted call prices as float32 array of length N.

    Kernel is in AUTO_RESTART mode.  6 AXI-Lite PCIe transactions per sample
    (4 writes + 2 reads, ~3 µs each) are the throughput bottleneck.
    """
    N     = len(X_batch)
    preds = np.empty(N, dtype=np.float32)
    for i in range(N):
        r0, r1, r2 = pack_inputs(X_batch[i])
        _wr(REG_INPUT_0,   r0)
        _wr(REG_INPUT_1,   r1)
        _wr(REG_INPUT_2,   r2)
        _wr(REG_INPUT_VLD, 1)
        preds[i] = unpack_output(_rr(REG_OUTPUT_LO), _rr(REG_OUTPUT_HI))
    return preds

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch throughput benchmark on Alveo U280")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=[1, 100, 1000, 10000],
                        help="Batch sizes to test (default: 1 100 1000 10000)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeat runs per batch size for stable timing (default: 3)")
    parser.add_argument("--device",  type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path, name in [(XCLBIN_PATH, "xclbin"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # ── Load test data ────────────────────────────────────────────────────────
    print("[THROUGHPUT]  Loading test data...")
    df    = pd.read_csv(TEST_CSV)
    X_all = df.drop("Call Price", axis=1).values.astype(np.float32)
    Y_all = df["Call Price"].values.astype(np.float32)
    nb_total = len(X_all)
    print(f"[THROUGHPUT]  Available samples : {nb_total}")

    # ── Initialise FPGA ───────────────────────────────────────────────────────
    global _h, _ci
    print(f"\n[THROUGHPUT]  Loading bitstream: {XCLBIN_PATH}")
    pdev  = pyxrt.device(args.device)
    pxcl  = pyxrt.xclbin(XCLBIN_PATH)
    puuid = pdev.register_xclbin(pxcl)
    uid   = _uuidmod.UUID(puuid.to_string())

    _h = xclOpen(args.device, None, xclVerbosityLevel.XCL_INFO)
    with open(XCLBIN_PATH, "rb") as f:
        buf = ctypes.create_string_buffer(f.read())
    xclLoadXclBin(_h, buf); del buf

    _ci = xclIPName2Index(_h, CU_NAME)
    xclOpenContext(_h, uid, _ci, False)
    _wr(REG_CTRL, 0x81)   # AUTO_RESTART
    print("[THROUGHPUT]  FPGA initialised.")
    print(f"[THROUGHPUT]  CU: {CU_NAME.decode()}  |  Clock: {KERNEL_FREQ_MHZ} MHz  "
          f"|  II: {PIPELINE_II} cycles  "
          f"|  Theoretical peak (compute only): {THEORETICAL_MAX_TP/1e6:.2f} Msamples/s")

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print("\n[THROUGHPUT]  Warm-up (100 samples)...")
    run_batch(X_all[:100])
    print("[THROUGHPUT]  Warm-up complete.")

    # ── Benchmark each batch size ─────────────────────────────────────────────
    summary_rows = []

    for batch_size in args.batch_sizes:
        n_samples = min(batch_size, nb_total)
        X_batch   = X_all[:n_samples]
        Y_batch   = Y_all[:n_samples]

        print(f"\n[THROUGHPUT]  ── Batch size {batch_size:>6}  ({n_samples} samples) ──")

        wall_times = []
        preds_last = None

        for rep in range(args.repeats):
            t_start    = time.perf_counter()
            preds      = run_batch(X_batch)
            t_end      = time.perf_counter()
            elapsed    = t_end - t_start
            wall_times.append(elapsed)
            preds_last = preds
            tp         = n_samples / elapsed
            print(f"[THROUGHPUT]    Rep {rep+1}/{args.repeats}: "
                  f"{elapsed*1e3:>8.2f} ms  |  {tp:>9.1f} samples/s  "
                  f"({tp/1e3:.3f} ksamples/s)")

        # Use the median repeat time for reporting (avoids outlier first run)
        wall_sec   = float(np.median(wall_times))
        throughput = n_samples / wall_sec
        avg_us     = wall_sec / n_samples * 1e6
        avg_ns     = wall_sec / n_samples * 1e9

        mse = float(mean_squared_error(Y_batch, preds_last))
        mae = float(mean_absolute_error(Y_batch, preds_last))
        r2  = float(r2_score(Y_batch, preds_last))

        print(f"[THROUGHPUT]  Wall-time (median): {wall_sec*1e3:.2f} ms")
        print(f"[THROUGHPUT]  Throughput        : {throughput:.1f} samples/s  "
              f"= {throughput/1e3:.3f} ksamples/s")
        print(f"[THROUGHPUT]  Avg/sample        : {avg_us:.3f} µs  ({avg_ns:.1f} ns)")
        print(f"[THROUGHPUT]  MAE: {mae:.6f}  |  MSE: {mse:.8f}  |  R²: {r2:.6f}")

        # Per-batch result directory (mirrors Mathys's resultat_testing structure)
        batch_dir = os.path.join(OUTPUT_DIR, f"FPGA_{batch_size}")
        os.makedirs(batch_dir, exist_ok=True)

        # Mathys-compatible metrics CSV
        pd.DataFrame({
            "Metric": ["MAE", "MSE", "R²", "Wall-time (s)", "Avg / sample (s)"],
            "Value":  [mae, mse, r2, wall_sec, wall_sec / n_samples],
        }).to_csv(os.path.join(batch_dir, f"{batch_size}_metrics.csv"), index=False)

        # Mathys-compatible predictions CSV
        pd.DataFrame({
            "y_reel":   Y_batch,
            "y_prédit": preds_last,
        }).to_csv(os.path.join(batch_dir, "predictions_detail.csv"), index=False)

        # Scatter plot (one per batch size)
        plt.figure(figsize=(6, 5))
        plt.scatter(Y_batch, preds_last, alpha=0.4, s=2, color="steelblue")
        plt.plot([Y_batch.min(), Y_batch.max()],
                 [Y_batch.min(), Y_batch.max()], "r--", linewidth=1)
        plt.xlabel("True Call Price")
        plt.ylabel("FPGA Predicted Call Price")
        plt.title(f"FPGA Predictions vs Ground Truth\n"
                  f"Batch size = {batch_size}  |  R² = {r2:.4f}  |  MAE = {mae:.4f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, "prediction_vs_reel.png"), dpi=150)
        plt.close()

        summary_rows.append({
            "Batch Size":            batch_size,
            "N Samples":             n_samples,
            "Wall-time (s)":         wall_sec,
            "Wall-time (ms)":        wall_sec * 1e3,
            "Throughput (samp/s)":   throughput,
            "Throughput (ksamp/s)":  throughput / 1e3,
            "Avg/sample (µs)":       avg_us,
            "Avg/sample (ns)":       avg_ns,
            "MAE":                   mae,
            "MSE":                   mse,
            "R²":                    r2,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "throughput_summary.csv"), index=False)

    print("\n" + "="*75)
    print(" THROUGHPUT SUMMARY — Alveo U280 @ 181 MHz")
    print("="*75)
    print(summary_df[["Batch Size", "Wall-time (ms)",
                       "Throughput (ksamp/s)", "Avg/sample (µs)", "R²"]].to_string(index=False))
    print(f"\n  Theoretical max throughput (pipeline-limited, compute only):")
    print(f"    II = {PIPELINE_II} cycles @ {KERNEL_FREQ_MHZ} MHz → "
          f"{THEORETICAL_MAX_TP/1e3:.1f} ksamples/s  "
          f"({THEORETICAL_MAX_TP/1e6:.2f} Msamples/s)")

    # ── Throughput scaling plot ───────────────────────────────────────────────
    bs     = [r["Batch Size"]           for r in summary_rows]
    tp_k   = [r["Throughput (ksamp/s)"] for r in summary_rows]
    avg_us = [r["Avg/sample (µs)"]      for r in summary_rows]
    labels = [str(b) for b in bs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars0 = axes[0].bar(labels, tp_k, color="steelblue", edgecolor="black", linewidth=0.6)
    axes[0].set_xlabel("Batch Size (number of samples)", fontsize=11)
    axes[0].set_ylabel("Throughput (ksamples / s)", fontsize=11)
    axes[0].set_title(
        f"FPGA Batch Throughput — Alveo U280 @ {KERNEL_FREQ_MHZ} MHz\n"
        f"(Pipeline II = {PIPELINE_II} cycles,  theoretical peak ≈ "
        f"{THEORETICAL_MAX_TP/1e3:.0f} ksamples/s)", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.4)
    for bar, v in zip(bars0, tp_k):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9)

    bars1 = axes[1].bar(labels, avg_us, color="darkorange", edgecolor="black", linewidth=0.6)
    axes[1].set_xlabel("Batch Size (number of samples)", fontsize=11)
    axes[1].set_ylabel("Average Latency per Sample (µs)", fontsize=11)
    axes[1].set_title(
        f"Average Per-Sample Latency — Alveo U280 @ {KERNEL_FREQ_MHZ} MHz\n"
        f"(Includes sequential PCIe overhead per sample)", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.4)
    for bar, v in zip(bars1, avg_us):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "throughput_scaling.png"), dpi=150)
    plt.close()

    print(f"\n[THROUGHPUT]  All results saved to: {OUTPUT_DIR}/")
    xclClose(_h)


if __name__ == "__main__":
    main()
