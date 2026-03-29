"""
iostream_latency.py
===================
Single-sample inference latency measurement for myproject_aximm on Alveo U280.

Unlike the AXI-Lite version (fpga_latency_single.py), this kernel moves data
via HBM buffers.  Two latency components are measured separately:

  1. End-to-end:  host→HBM transfer + kernel execution + HBM→host transfer
  2. Compute-only: kernel execution alone (data pre-loaded to HBM once)

Hardware context (100 MHz kernel clock):
  HLS latency (inner myproject): 23–24 cycles  = 230–240 ns
  HLS pipeline II:               13–14 cycles  = 130–140 ns
  Outer sequential loop / sample: ~24 cycles   = ~240 ns

Outputs:
  results/latency/iostream_latency_summary.csv
  results/latency/iostream_latency_raw.csv
  results/latency/iostream_latency_histogram.png

Usage:
    python iostream_latency.py
    python iostream_latency.py --runs 5000 --warmup 200
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
OUTPUT_DIR  = os.path.join(_HERE, "results/latency")

KERNEL_NAME        = "myproject_aximm"
N_FEATURES         = 6
SCALE              = 1024
KERNEL_FREQ_MHZ    = 100.0
CYCLE_NS           = 1e3 / KERNEL_FREQ_MHZ          # 10.0 ns
HLS_LATENCY_CYCLES = 24                              # worst-case (inner myproject)
HLS_II_CYCLES      = 14                              # pipeline II (inner myproject)
THEORETICAL_NS     = HLS_LATENCY_CYCLES * CYCLE_NS   # 240 ns for compute core

# ── Fixed-point helpers ────────────────────────────────────────────────────────
def encode_sample(x) -> bytes:
    q = np.round(np.asarray(x, dtype=np.float64) * SCALE).astype(np.int32)
    q = np.clip(q, -(1 << 15), (1 << 15) - 1).astype(np.int16)
    return q.tobytes()

def decode_result(b: bytes) -> float:
    return int(np.frombuffer(b, dtype=np.int16)[0]) / SCALE

# ── Stats reporter ─────────────────────────────────────────────────────────────
def report_stats(label: str, data_ns: np.ndarray) -> dict:
    mean_ns = float(data_ns.mean())
    med_ns  = float(np.median(data_ns))
    std_ns  = float(data_ns.std())
    min_ns  = float(data_ns.min())
    max_ns  = float(data_ns.max())
    p95_ns  = float(np.percentile(data_ns, 95))
    p99_ns  = float(np.percentile(data_ns, 99))
    print(f"\n  [{label}]")
    print(f"    Mean   : {mean_ns:>10.0f} ns  ({mean_ns/1e3:>8.3f} µs)")
    print(f"    Median : {med_ns:>10.0f} ns  ({med_ns/1e3:>8.3f} µs)")
    print(f"    Std    : {std_ns:>10.0f} ns")
    print(f"    Min    : {min_ns:>10.0f} ns")
    print(f"    Max    : {max_ns:>10.0f} ns")
    print(f"    P95    : {p95_ns:>10.0f} ns  ({p95_ns/1e3:>8.3f} µs)")
    print(f"    P99    : {p99_ns:>10.0f} ns  ({p99_ns/1e3:>8.3f} µs)")
    return {"Measurement": label, "Mean_ns": mean_ns, "Median_ns": med_ns,
            "Std_ns": std_ns, "Min_ns": min_ns, "Max_ns": max_ns,
            "P95_ns": p95_ns, "P99_ns": p99_ns,
            "Mean_us": mean_ns/1e3, "Median_us": med_ns/1e3}

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",   type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--sample", type=int, default=0,
                        help="Row index in testing_set.csv to use")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path, name in [(XCLBIN_PATH, "xclbin"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # ── Load one test sample ───────────────────────────────────────────────────
    df     = pd.read_csv(TEST_CSV)
    X      = df.drop("Call Price", axis=1).values.astype(np.float32)
    Y      = df["Call Price"].values.astype(np.float32)
    sample = X[args.sample]
    y_true = float(Y[args.sample])
    sample_bytes = encode_sample(sample)

    print(f"[LATENCY]  Test sample  : index={args.sample}")
    print(f"[LATENCY]  Input        : {sample}")
    print(f"[LATENCY]  True price   : {y_true:.6f}")
    print(f"\n[LATENCY]  Hardware reference")
    print(f"           Clock        : {KERNEL_FREQ_MHZ} MHz  "
          f"(period = {CYCLE_NS:.1f} ns)")
    print(f"           HLS latency  : {HLS_LATENCY_CYCLES} cycles  "
          f"= {THEORETICAL_NS:.0f} ns  (inner compute core)")
    print(f"           HLS II       : {HLS_II_CYCLES} cycles  "
          f"= {HLS_II_CYCLES * CYCLE_NS:.0f} ns")

    # ── Init FPGA ──────────────────────────────────────────────────────────────
    print(f"\n[LATENCY]  Loading xclbin...")
    device = pyxrt.device(args.device)
    xclbin = pyxrt.xclbin(XCLBIN_PATH)
    device.register_xclbin(xclbin)
    kernel = pyxrt.kernel(device, xclbin.get_uuid(), KERNEL_NAME)
    print("[LATENCY]  FPGA initialised.")

    in_bytes  = N_FEATURES * 2
    out_bytes = 1 * 2
    in_bo  = pyxrt.bo(device, in_bytes,  pyxrt.bo.flags.normal, kernel.group_id(0))
    out_bo = pyxrt.bo(device, out_bytes, pyxrt.bo.flags.normal, kernel.group_id(1))

    # ── Warm-up ────────────────────────────────────────────────────────────────
    print(f"\n[LATENCY]  Warm-up ({args.warmup} runs)...")
    in_bo.write(sample_bytes, 0)
    in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for _ in range(args.warmup):
        run = kernel(in_bo, out_bo, 1)
        run.wait()
    print("[LATENCY]  Warm-up complete.")

    # ── Verify correctness ─────────────────────────────────────────────────────
    out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    fpga_pred = decode_result(out_bo.read(out_bytes, 0))
    abs_err   = abs(fpga_pred - y_true)
    print(f"\n[LATENCY]  FPGA output  : {fpga_pred:.6f}")
    print(f"[LATENCY]  Ground truth : {y_true:.6f}  |  Abs error : {abs_err:.6f}")

    # ── Measurement A: compute-only (data pre-loaded in HBM) ──────────────────
    # Input buffer already synced to device — only measure kernel execution time.
    print(f"\n[LATENCY]  Measuring compute-only latency ({args.runs} runs)...")
    compute_ns = np.zeros(args.runs, dtype=np.float64)
    for i in range(args.runs):
        t0 = time.perf_counter_ns()
        run = kernel(in_bo, out_bo, 1)
        run.wait()
        compute_ns[i] = time.perf_counter_ns() - t0

    # ── Measurement B: end-to-end (includes host↔HBM DMA each iteration) ──────
    print(f"\n[LATENCY]  Measuring end-to-end latency ({args.runs} runs)...")
    e2e_ns = np.zeros(args.runs, dtype=np.float64)
    for i in range(args.runs):
        t0 = time.perf_counter_ns()
        in_bo.write(sample_bytes, 0)
        in_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = kernel(in_bo, out_bo, 1)
        run.wait()
        out_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        _ = out_bo.read(out_bytes, 0)
        e2e_ns[i] = time.perf_counter_ns() - t0

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f" LATENCY RESULTS — Alveo U280 @ {KERNEL_FREQ_MHZ} MHz  ({args.runs:,} runs)")
    print("=" * 65)
    row_compute = report_stats("Compute-only  (kernel launch + execute)", compute_ns)
    row_e2e     = report_stats("End-to-end    (+ host↔HBM DMA per call)", e2e_ns)

    dma_overhead = e2e_ns.mean() - compute_ns.mean()
    print(f"\n[LATENCY]  Theoretical HLS compute core   : {THEORETICAL_NS:.0f} ns")
    print(f"[LATENCY]  Measured compute median        : {np.median(compute_ns):.0f} ns  "
          f"({np.median(compute_ns)/1e3:.3f} µs)")
    print(f"[LATENCY]  XRT kernel launch overhead     : "
          f"~{compute_ns.mean() - THEORETICAL_NS:.0f} ns  "
          f"({(compute_ns.mean() - THEORETICAL_NS)/1e3:.2f} µs)")
    print(f"[LATENCY]  Host↔HBM DMA overhead (avg)   : "
          f"~{dma_overhead:.0f} ns  ({dma_overhead/1e3:.2f} µs)")
    print(f"[LATENCY]  Implied throughput (compute)   : "
          f"{1e9/np.median(compute_ns):,.0f} samples/s")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    rows = [row_compute, row_e2e]
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "iostream_latency_summary.csv"), index=False)
    pd.DataFrame({"compute_ns": compute_ns, "e2e_ns": e2e_ns}).to_csv(
        os.path.join(OUTPUT_DIR, "iostream_latency_raw.csv"), index=False)

    # ── Histogram ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, label, colour in zip(
            axes,
            [compute_ns / 1e3, e2e_ns / 1e3],
            ["Compute-only (µs)", "End-to-end (µs)"],
            ["steelblue", "darkorange"]):
        clip = float(np.percentile(data, 99.5))
        ax.hist(data[data <= clip], bins=100, color=colour, edgecolor="none", alpha=0.8)
        med  = float(np.median(data))
        mean = float(data.mean())
        ax.axvline(med,  color="red",   linestyle="--", linewidth=1.2,
                   label=f"Median: {med:.2f} µs")
        ax.axvline(mean, color="green", linestyle="--", linewidth=1.2,
                   label=f"Mean:   {mean:.2f} µs")
        ax.set_xlabel("Latency (µs)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\nAlveo U280 @ {KERNEL_FREQ_MHZ} MHz — {args.runs:,} runs")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "iostream_latency_histogram.png"), dpi=150)
    plt.close()
    print(f"\n[LATENCY]  Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
