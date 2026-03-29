"""
fpga_latency_single.py
======================
Single-sample inference latency measurement for the hls4ml neural network
deployed on the Alveo U280 FPGA.

Timing uses time.perf_counter_ns() (nanosecond resolution) and measures the
end-to-end round-trip: fixed-point encoding, 4 AXI-Lite register writes
(3 inputs + IN_VLD), 2 AXI-Lite register reads (output), and float decoding.

The kernel is held in AUTO_RESTART mode — it runs continuously and accepts
new inputs whenever IN_VLD is asserted.  Each PCIe transaction takes ~3 µs,
so the 6 transactions dominate over the 281 ns FPGA compute time.

Reports: mean, median, min, max, std, P95, P99.

Outputs:
  results/latency/latency_summary.csv
  results/latency/latency_raw.csv
  results/latency/latency_histogram.png

Usage:
    python fpga_latency_single.py
    python fpga_latency_single.py --runs 50000 --warmup 500
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
OUTPUT_DIR  = os.path.join(_HERE, "results/latency")

# ── Kernel / CU identity ──────────────────────────────────────────────────────
CU_NAME = b"myproject:myproject_1"

# ── Hardware timing reference ─────────────────────────────────────────────────
KERNEL_FREQ_MHZ        = 181.0
CYCLE_NS               = 1e3 / KERNEL_FREQ_MHZ          # 5.525 ns
HLS_LATENCY_CYCLES_WC  = 51                              # worst-case from HLS report
HLS_LATENCY_CYCLES_BC  = 48                              # best-case from HLS report
PIPELINE_II_CYCLES     = 10                              # initiation interval
THEORETICAL_LATENCY_NS = HLS_LATENCY_CYCLES_WC * CYCLE_NS  # 281.8 ns

# ── AXI-Lite Register Map ─────────────────────────────────────────────────────
REG_CTRL        = 0x00
REG_INPUT_0     = 0x10
REG_INPUT_1     = 0x14
REG_INPUT_2     = 0x18
REG_INPUT_VLD   = 0x1C
REG_OUTPUT_LO   = 0x20
REG_OUTPUT_HI   = 0x24
REG_OUTPUT_VLD  = 0x28

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

# ── Timed inference ───────────────────────────────────────────────────────────
def timed_infer(x) -> tuple:
    """
    Run one inference in AUTO_RESTART mode, returning (e2e_ns, output_value).

    Timing covers 6 AXI-Lite PCIe register transactions:
      4 writes: input_1[0..2] + IN_VLD
      2 reads:  output_lo + output_hi

    FPGA compute (281 ns) is invisible — it completes within the first write
    transaction's PCIe round-trip (~3 µs).
    """
    r0, r1, r2 = pack_inputs(x)
    t0 = time.perf_counter_ns()
    _wr(REG_INPUT_0,   r0)
    _wr(REG_INPUT_1,   r1)
    _wr(REG_INPUT_2,   r2)
    _wr(REG_INPUT_VLD, 1)
    lo  = _rr(REG_OUTPUT_LO)
    hi  = _rr(REG_OUTPUT_HI)
    t1  = time.perf_counter_ns()
    return (t1 - t0), unpack_output(lo, hi)

# ── Statistics reporter ───────────────────────────────────────────────────────
def stats(label: str, data_ns: np.ndarray, unit: str = "ns") -> dict:
    mean_ns   = data_ns.mean()
    median_ns = float(np.median(data_ns))
    std_ns    = data_ns.std()
    min_ns    = data_ns.min()
    max_ns    = data_ns.max()
    p95_ns    = float(np.percentile(data_ns, 95))
    p99_ns    = float(np.percentile(data_ns, 99))

    print(f"\n  [{label}]")
    print(f"    Mean    : {mean_ns:>10.1f} ns  ({mean_ns/1e3:>8.3f} µs)")
    print(f"    Median  : {median_ns:>10.1f} ns  ({median_ns/1e3:>8.3f} µs)")
    print(f"    Std     : {std_ns:>10.1f} ns")
    print(f"    Min     : {min_ns:>10.1f} ns")
    print(f"    Max     : {max_ns:>10.1f} ns")
    print(f"    P95     : {p95_ns:>10.1f} ns  ({p95_ns/1e3:>8.3f} µs)")
    print(f"    P99     : {p99_ns:>10.1f} ns  ({p99_ns/1e3:>8.3f} µs)")

    return {
        "Measurement":  label,
        "Mean_ns":      mean_ns,
        "Median_ns":    median_ns,
        "Std_ns":       std_ns,
        "Min_ns":       min_ns,
        "Max_ns":       max_ns,
        "P95_ns":       p95_ns,
        "P99_ns":       p99_ns,
        "Mean_us":      mean_ns / 1e3,
        "Median_us":    median_ns / 1e3,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Single-sample latency measurement on Alveo U280")
    parser.add_argument("--runs",   type=int, default=10000,
                        help="Number of timed inference runs (default: 10000)")
    parser.add_argument("--warmup", type=int, default=200,
                        help="Warm-up runs before timing (default: 200)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Row index in testing_set.csv to use as test sample (default: 0)")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path, name in [(XCLBIN_PATH, "xclbin"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # ── Load one test sample ──────────────────────────────────────────────────
    df     = pd.read_csv(TEST_CSV)
    X      = df.drop("Call Price", axis=1).values.astype(np.float32)
    Y      = df["Call Price"].values.astype(np.float32)
    sample = X[args.sample]
    y_true = Y[args.sample]

    print(f"[LATENCY]  Test sample index : {args.sample}")
    print(f"[LATENCY]  Input features    : {sample}")
    print(f"[LATENCY]  True call price   : {y_true:.6f}")

    print(f"\n[LATENCY]  Hardware reference")
    print(f"           Kernel clock     : {KERNEL_FREQ_MHZ} MHz  (period = {CYCLE_NS:.3f} ns)")
    print(f"           HLS latency      : {HLS_LATENCY_CYCLES_BC}–{HLS_LATENCY_CYCLES_WC} cycles")
    print(f"           Theoretical min  : {HLS_LATENCY_CYCLES_BC * CYCLE_NS:.1f} ns  "
          f"(best-case)  →  {HLS_LATENCY_CYCLES_WC * CYCLE_NS:.1f} ns  (worst-case)")
    print(f"           Pipeline II      : {PIPELINE_II_CYCLES} cycles  "
          f"= {PIPELINE_II_CYCLES * CYCLE_NS:.1f} ns")

    # ── Initialise FPGA ───────────────────────────────────────────────────────
    global _h, _ci
    print(f"\n[LATENCY]  Loading bitstream: {XCLBIN_PATH}")
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
    _wr(REG_CTRL, 0x81)   # AUTO_RESTART: kernel runs continuously
    print("[LATENCY]  FPGA initialised.")

    # ── Warm-up (ensures PCIe link, caches, and kernel state are stable) ──────
    print(f"\n[LATENCY]  Warm-up ({args.warmup} runs)...")
    for _ in range(args.warmup):
        timed_infer(sample)
    print("[LATENCY]  Warm-up complete.")

    # ── Timed measurement ─────────────────────────────────────────────────────
    print(f"[LATENCY]  Measuring latency: {args.runs} runs...")
    e2e_ns  = np.zeros(args.runs, dtype=np.float64)
    outputs = np.zeros(args.runs, dtype=np.float32)

    for i in range(args.runs):
        e2e_ns[i], outputs[i] = timed_infer(sample)

    # ── Verify correctness ────────────────────────────────────────────────────
    fpga_pred = float(np.median(outputs))
    abs_err   = abs(fpga_pred - y_true)
    print(f"\n[LATENCY]  FPGA output (median over {args.runs} runs): {fpga_pred:.6f}")
    print(f"[LATENCY]  Ground truth: {y_true:.6f}  |  Absolute error: {abs_err:.6f}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(" LATENCY RESULTS — Alveo U280 @ 181 MHz")
    print(f" Runs: {args.runs:,}  |  Sample index: {args.sample}")
    print("="*65)

    row_e2e = stats("End-to-end  (6 AXI-Lite PCIe register transactions)", e2e_ns)

    print(f"\n[LATENCY]  ── Context ──")
    print(f"           Theoretical FPGA compute (worst-case): {THEORETICAL_LATENCY_NS:.1f} ns")
    print(f"           Measured end-to-end mean:  {e2e_ns.mean():.1f} ns  ({e2e_ns.mean()/1e3:.2f} µs)")
    overhead = e2e_ns.mean() - THEORETICAL_LATENCY_NS
    print(f"           PCIe overhead (est.): {overhead:.1f} ns  ({overhead/1e3:.2f} µs)")
    print(f"           6 PCIe transactions × ~{e2e_ns.mean()/6/1e3:.2f} µs each")
    print(f"           Implied max throughput: {1e9/float(np.median(e2e_ns)):,.0f} samples/s  "
          f"= {1e6/float(np.median(e2e_ns)):.2f} ksamples/s")

    # ── Save results ──────────────────────────────────────────────────────────
    pd.DataFrame([row_e2e]).to_csv(
        os.path.join(OUTPUT_DIR, "latency_summary.csv"), index=False)
    pd.DataFrame({"e2e_ns": e2e_ns, "output": outputs}).to_csv(
        os.path.join(OUTPUT_DIR, "latency_raw.csv"), index=False)

    # ── Histogram plot ────────────────────────────────────────────────────────
    e2e_us   = e2e_ns / 1e3
    clip_p99 = float(np.percentile(e2e_us, 99.5))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(e2e_us[e2e_us <= clip_p99], bins=120,
            color="steelblue", edgecolor="none", alpha=0.8)
    med  = float(np.median(e2e_us))
    mean = float(e2e_us.mean())
    ax.axvline(med,  color="red",   linestyle="--", linewidth=1.2,
               label=f"Median: {med:.3f} µs")
    ax.axvline(mean, color="green", linestyle="--", linewidth=1.2,
               label=f"Mean:   {mean:.3f} µs")
    ax.set_xlabel("Latency (µs)")
    ax.set_ylabel("Count")
    ax.set_title(f"End-to-End Inference Latency — Alveo U280 @ {KERNEL_FREQ_MHZ} MHz\n"
                 f"(6 AXI-Lite PCIe register transactions, {args.runs:,} runs)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "latency_histogram.png"), dpi=150)
    plt.close()
    print(f"\n[LATENCY]  Results saved to: {OUTPUT_DIR}/")
    xclClose(_h)


if __name__ == "__main__":
    main()
