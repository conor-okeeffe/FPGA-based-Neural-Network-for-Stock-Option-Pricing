"""
iostream_throughput.py
======================
Batched throughput characterisation for myproject_aximm on Alveo U280.

Sweeps over batch sizes and reports:
  - Transfer time  (host → HBM)
  - Kernel time    (pure execution)
  - Readback time  (HBM → host)
  - Total throughput (samples/s)

Theoretical kernel throughput:
  Sequential loop, 1 sample per ~24 cycles at 100 MHz → ~4.17 MSamples/s
  (inner myproject dataflow core has II=13–14 cycles, but outer loop is
   sequential so effective rate = 1 / latency_per_sample, not 1 / II)

Outputs:
  results/throughput/iostream_throughput_sweep.csv
  results/throughput/iostream_throughput_plot.png

Usage:
    python iostream_throughput.py
    python iostream_throughput.py --repeats 5
"""

import os, sys, argparse, time
sys.path.insert(0, "/opt/xilinx/xrt/python")
sys.path.insert(0, "/home/conor/Proj_New_515/Sem2Linux_lean/.venv/lib/python3.10/site-packages")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyxrt
except ImportError:
    raise ImportError("XRT Python bindings not found. Run: source /opt/xilinx/xrt/setup.sh")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
XCLBIN_PATH = os.path.join(_HERE,
    "my_vitis_project_RF10_IOSTREAM_PYTHON/myproject_aximm.xclbin")
TEST_CSV    = os.path.join(_HERE,
    "Mathys_Pradel_Work/RB3_Option_pricing_mathys_Pradels/scripts_v4/Data/testing_set.csv")
OUTPUT_DIR  = os.path.join(_HERE, "results/throughput")

KERNEL_NAME        = "myproject_aximm"
N_FEATURES         = 6
SCALE              = 1024
KERNEL_FREQ_MHZ    = 100.0
CYCLE_NS           = 1e3 / KERNEL_FREQ_MHZ
HLS_LATENCY_CYCLES = 24
THEORETICAL_MSPS   = KERNEL_FREQ_MHZ / HLS_LATENCY_CYCLES  # ~4.17 MSamples/s

BATCH_SIZES = [1, 10, 100, 1_000, 5_000, 10_000, 50_000, 100_000]

# ── Fixed-point helpers ────────────────────────────────────────────────────────
def encode(arr: np.ndarray) -> np.ndarray:
    q = np.round(arr.astype(np.float64) * SCALE).astype(np.int32)
    np.clip(q, -(1 << 15), (1 << 15) - 1, out=q)
    return q.astype(np.int16)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per batch size (default: 3)")
    parser.add_argument("--device",  type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path, name in [(XCLBIN_PATH, "xclbin"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # ── Load test data (use up to max batch size, tile if needed) ─────────────
    df   = pd.read_csv(TEST_CSV)
    X    = df.drop("Call Price", axis=1).values.astype(np.float32)
    Nmax = max(BATCH_SIZES)
    if len(X) < Nmax:
        reps = -(-Nmax // len(X))   # ceiling div
        X    = np.tile(X, (reps, 1))[:Nmax]
    X_q_all = encode(X[:Nmax]).flatten()  # (Nmax*6,) int16

    print(f"[THROUGHPUT]  Theoretical kernel throughput: "
          f"{THEORETICAL_MSPS:.2f} MSamples/s  "
          f"({HLS_LATENCY_CYCLES} cycles/sample @ {KERNEL_FREQ_MHZ} MHz)")

    # ── Init FPGA ──────────────────────────────────────────────────────────────
    print(f"\n[THROUGHPUT]  Loading xclbin...")
    device = pyxrt.device(args.device)
    xclbin = pyxrt.xclbin(XCLBIN_PATH)
    device.register_xclbin(xclbin)
    kernel = pyxrt.kernel(device, xclbin.get_uuid(), KERNEL_NAME)
    print("[THROUGHPUT]  FPGA initialised.")

    # Allocate max-size buffers once; reuse across all batch sizes
    in_bytes_max  = int(Nmax * N_FEATURES * 2)
    out_bytes_max = int(Nmax * 2)
    in_bo  = pyxrt.bo(device, in_bytes_max,  pyxrt.bo.flags.normal, kernel.group_id(0))
    out_bo = pyxrt.bo(device, out_bytes_max, pyxrt.bo.flags.normal, kernel.group_id(1))

    # ── Warm-up ────────────────────────────────────────────────────────────────
    print("[THROUGHPUT]  Warm-up (1000 samples)...")
    wn = min(1000, Nmax)
    in_bo.write(X_q_all[:wn * N_FEATURES].tobytes(), 0)
    in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run = kernel(in_bo, out_bo, wn)
    run.wait()

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"\n[THROUGHPUT]  Sweeping batch sizes (repeats={args.repeats})...")
    print(f"  {'Batch':>10}  {'Xfer→(ms)':>10}  {'Kernel(ms)':>11}  "
          f"{'←Xfer(ms)':>10}  {'Total(ms)':>10}  {'MSamples/s':>11}")
    print("  " + "-" * 72)

    results = []
    for n in BATCH_SIZES:
        in_bytes  = int(n * N_FEATURES * 2)
        out_bytes = int(n * 2)
        chunk     = X_q_all[:n * N_FEATURES]

        xfer_times    = []
        kernel_times  = []
        readback_times = []
        total_times   = []

        for _ in range(args.repeats):
            t0 = time.perf_counter()

            # 1) Transfer input
            in_bo.write(chunk.tobytes(), 0)
            in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            t1 = time.perf_counter()

            # 2) Kernel execution
            run = kernel(in_bo, out_bo, n)
            run.wait()
            t2 = time.perf_counter()

            # 3) Readback output
            out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            _ = out_bo.read(out_bytes, 0)
            t3 = time.perf_counter()

            xfer_times.append((t1 - t0) * 1e3)
            kernel_times.append((t2 - t1) * 1e3)
            readback_times.append((t3 - t2) * 1e3)
            total_times.append((t3 - t0) * 1e3)

        xfer_ms    = np.median(xfer_times)
        kernel_ms  = np.median(kernel_times)
        read_ms    = np.median(readback_times)
        total_ms   = np.median(total_times)
        msps       = n / (total_ms * 1e-3) / 1e6   # MSamples/s end-to-end
        msps_k     = n / (kernel_ms * 1e-3) / 1e6  # MSamples/s kernel-only

        print(f"  {n:>10,}  {xfer_ms:>10.3f}  {kernel_ms:>11.3f}  "
              f"{read_ms:>10.3f}  {total_ms:>10.3f}  {msps:>10.3f}M  "
              f"(kernel: {msps_k:.3f}M)")

        results.append({
            "n_samples":        n,
            "xfer_to_dev_ms":   xfer_ms,
            "kernel_exec_ms":   kernel_ms,
            "readback_ms":      read_ms,
            "total_ms":         total_ms,
            "throughput_msps":  msps,
            "kernel_msps":      msps_k,
        })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    df_res.to_csv(
        os.path.join(OUTPUT_DIR, "iostream_throughput_sweep.csv"), index=False)

    best_total  = df_res["throughput_msps"].max()
    best_kernel = df_res["kernel_msps"].max()
    print(f"\n[THROUGHPUT]  Peak end-to-end : {best_total:.3f} MSamples/s")
    print(f"[THROUGHPUT]  Peak kernel-only: {best_kernel:.3f} MSamples/s")
    print(f"[THROUGHPUT]  Theoretical max : {THEORETICAL_MSPS:.3f} MSamples/s")
    print(f"[THROUGHPUT]  Efficiency      : {best_kernel/THEORETICAL_MSPS*100:.1f}%")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: throughput vs batch size
    ax = axes[0]
    ax.semilogx(df_res["n_samples"], df_res["throughput_msps"],
                "o-", color="steelblue", label="End-to-end")
    ax.semilogx(df_res["n_samples"], df_res["kernel_msps"],
                "s--", color="darkorange", label="Kernel-only")
    ax.axhline(THEORETICAL_MSPS, color="red", linestyle=":", linewidth=1.5,
               label=f"Theoretical ({THEORETICAL_MSPS:.2f} MSamples/s)")
    ax.set_xlabel("Batch size (samples)")
    ax.set_ylabel("Throughput (MSamples/s)")
    ax.set_title(f"Throughput vs Batch Size\nAlveo U280 @ {KERNEL_FREQ_MHZ} MHz")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)

    # Right: time breakdown for largest batch
    ax = axes[1]
    row_large = df_res[df_res["n_samples"] == df_res["n_samples"].max()].iloc[0]
    labels  = ["Host→HBM\n(xfer)", "Kernel\nexec", "HBM→Host\n(readback)"]
    times   = [row_large["xfer_to_dev_ms"],
               row_large["kernel_exec_ms"],
               row_large["readback_ms"]]
    colours = ["#4878d0", "#ee854a", "#6acc65"]
    bars = ax.bar(labels, times, color=colours, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{t:.1f} ms", ha="center", fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Time Breakdown — Batch {int(row_large['n_samples']):,} samples")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "iostream_throughput_plot.png"), dpi=150)
    plt.close()
    print(f"\n[THROUGHPUT]  Results saved to: {OUTPUT_DIR}/")

    # ── Save summary for report ────────────────────────────────────────────────
    pd.DataFrame({
        "Metric": ["Peak_end_to_end_MSamples_s", "Peak_kernel_MSamples_s",
                   "Theoretical_MSamples_s", "Efficiency_pct"],
        "Value":  [best_total, best_kernel, THEORETICAL_MSPS,
                   best_kernel / THEORETICAL_MSPS * 100]
    }).to_csv(os.path.join(OUTPUT_DIR, "iostream_throughput_summary.csv"), index=False)


if __name__ == "__main__":
    main()
