"""
fpga_accuracy.py
================
Model accuracy verification for the hls4ml neural network deployed on Alveo U280.
Runs the full test set (or --samples N samples) through the FPGA .xclbin and reports
accuracy metrics.  Optionally compares against the Keras float32 software baseline.

Metrics reported (matching Mathys Pradel's CPU_INFERENCE.py format):
  MAE, MSE, RMSE, R²
Outputs:
  results/accuracy/accuracy_metrics.csv
  results/accuracy/predictions_detail.csv
  results/accuracy/prediction_vs_truth.png

Usage:
    python fpga_accuracy.py
    python fpga_accuracy.py --samples 5000
    python fpga_accuracy.py --no_sw        # skip Keras baseline
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── XRT Python bindings ───────────────────────────────────────────────────────
# pyxrt is used only to load the xclbin and obtain the UUID.
# xrt_binding provides xclRegRead / xclRegWrite for AXI-Lite register access,
# which pyxrt.kernel does NOT expose in XRT 2.16.
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
        "Ensure XRT is installed and 'source /opt/xilinx/xrt/setup.sh' has been run."
    )

# Module-level handle/CU index set during FPGA init
_h  = None   # xclOpen handle
_ci = None   # compute-unit index

def _wr(off, val): xclRegWrite(_h, _ci, off, val)
def _rr(off):
    d = ctypes.c_uint(0); xclRegRead(_h, _ci, off, ctypes.byref(d)); return d.value

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
XCLBIN_PATH = os.path.join(_HERE, "my_vitis_project_RF10/myproject_prj/solution2/export.xclbin")
KERAS_MODEL = os.path.join(_HERE, "my_vitis_project_RF10/keras_model.keras")
TEST_CSV    = os.path.join(_HERE,
              "Mathys_Pradel_Work/RB3_Option_pricing_mathys_Pradels/scripts_v4/Data/testing_set.csv")
OUTPUT_DIR  = os.path.join(_HERE, "results/accuracy")

# ── Kernel / CU identity ──────────────────────────────────────────────────────
CU_NAME = b"myproject:myproject_1"   # full CU name used by xclIPName2Index

# ── AXI-Lite Register Map ─────────────────────────────────────────────────────
# Source: solution2/myproject_prj/solution2/solution2_data.json
# Kernel: myproject(input_t input_1[6], result_t layer10_out[1])
#   input_1   → ap_fixed<16,6> × 6  = 96 bits → 3 × 32-bit registers
#   layer10_out → ap_fixed<40,20> × 1 = 40 bits → 2 registers (32 + 8 bits)
REG_CTRL        = 0x00   # bits: AP_START[0] AP_DONE[1] AP_IDLE[2] AP_READY[3]
REG_INPUT_0     = 0x10   # input_1[1:0]  bits [31:0]
REG_INPUT_1     = 0x14   # input_1[3:2]  bits [63:32]
REG_INPUT_2     = 0x18   # input_1[5:4]  bits [95:64]
REG_INPUT_VLD   = 0x1C   # input_1_ap_vld
REG_OUTPUT_LO   = 0x20   # layer10_out[0] bits [31:0]
REG_OUTPUT_HI   = 0x24   # layer10_out[0] bits [39:32]  (lower 8 bits used)
REG_OUTPUT_VLD  = 0x28   # layer10_out_ap_vld

AP_START = 0x1
AP_DONE  = 0x2
AP_IDLE  = 0x4

# ── Fixed-point conversion helpers ───────────────────────────────────────────
# ap_fixed<16,6>: 16-bit signed, 6 integer bits, 10 fractional bits
# ap_fixed<40,20>: 40-bit signed, 20 integer bits, 20 fractional bits
FRAC_IN  = 10   # fractional bits for input
FRAC_OUT = 20   # fractional bits for output

def float_to_ap16_6(v: float) -> int:
    """Encode a float to ap_fixed<16,6> unsigned representation."""
    s = int(round(v * (1 << FRAC_IN)))
    s = max(-(1 << 15), min((1 << 15) - 1, s))
    return s & 0xFFFF  # two's complement as unsigned 16-bit

def pack_inputs(x) -> tuple:
    """
    Pack 6 ap_fixed<16,6> values into 3 × 32-bit AXI-Lite registers.
    HLS packs element 0 at LSB of the first register.
      reg0 = x[0] | (x[1] << 16)
      reg1 = x[2] | (x[3] << 16)
      reg2 = x[4] | (x[5] << 16)
    """
    e = [float_to_ap16_6(float(v)) for v in x]
    return (e[1] << 16) | e[0], (e[3] << 16) | e[2], (e[5] << 16) | e[4]

def unpack_output(lo: int, hi: int) -> float:
    """Decode ap_fixed<40,20> from two AXI-Lite registers."""
    int40 = (int(hi & 0xFF) << 32) | int(lo & 0xFFFFFFFF)
    if int40 >= (1 << 39):        # sign-extend 40-bit two's complement
        int40 -= (1 << 40)
    return int40 / (1 << FRAC_OUT)

# ── Single FPGA inference ─────────────────────────────────────────────────────
def fpga_infer(x) -> float:
    """Write inputs and read output.

    Kernel is held in AUTO_RESTART mode (AP_CTRL bit7 = 1, set once at init).
    Each PCIe AXI-Lite write takes ~3 µs; the four writes total ~12 µs, which
    is >> the 281 ns FPGA compute time, so the output is valid by the time
    the two read transactions complete.  No AP_DONE polling required.
    """
    r0, r1, r2 = pack_inputs(x)
    _wr(REG_INPUT_0,   r0)
    _wr(REG_INPUT_1,   r1)
    _wr(REG_INPUT_2,   r2)
    _wr(REG_INPUT_VLD, 1)
    return unpack_output(_rr(REG_OUTPUT_LO), _rr(REG_OUTPUT_HI))

# ── Metrics helper ────────────────────────────────────────────────────────────
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

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FPGA accuracy verification on Alveo U280")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of test samples (default: all)")
    parser.add_argument("--device",  type=int, default=0,
                        help="XRT device index (default: 0)")
    parser.add_argument("--no_sw",   action="store_true",
                        help="Skip Keras software baseline comparison")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Validate paths ────────────────────────────────────────────────────────
    for path, name in [(XCLBIN_PATH, "xclbin"), (TEST_CSV, "test CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # ── Load test data ────────────────────────────────────────────────────────
    print("[ACCURACY]  Loading test data...")
    df = pd.read_csv(TEST_CSV)
    X  = df.drop("Call Price", axis=1).values.astype(np.float32)
    Y  = df["Call Price"].values.astype(np.float32)
    if args.samples:
        X, Y = X[:args.samples], Y[:args.samples]
    N = len(X)
    print(f"[ACCURACY]  Samples: {N}")
    print(f"[ACCURACY]  Features: {list(df.drop('Call Price', axis=1).columns)}")
    print(f"[ACCURACY]  Input range: min={X.min():.4f}  max={X.max():.4f}")

    # ── Initialise FPGA ───────────────────────────────────────────────────────
    global _h, _ci
    print(f"\n[ACCURACY]  Loading bitstream: {XCLBIN_PATH}")
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
    # AUTO_RESTART (bit7): kernel runs continuously; host writes inputs and
    # reads output without AP_START / AP_DONE handshaking.
    _wr(REG_CTRL, 0x81)
    print("[ACCURACY]  FPGA initialised.")
    print(f"[ACCURACY]  CU: {CU_NAME.decode()}  |  Device: Alveo U280  |  Clock: 181 MHz")

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print("[ACCURACY]  Warm-up (50 samples)...")
    for i in range(min(50, N)):
        fpga_infer(X[i])

    # ── FPGA inference ────────────────────────────────────────────────────────
    print(f"[ACCURACY]  Running FPGA inference on {N} samples...")
    Y_fpga  = np.zeros(N, dtype=np.float32)
    t_start = time.perf_counter()
    for i in range(N):
        Y_fpga[i] = fpga_infer(X[i])
        if (i + 1) % 1000 == 0 or (i + 1) == N:
            elapsed = time.perf_counter() - t_start
            rate    = (i + 1) / elapsed
            print(f"[ACCURACY]  Progress: {i+1}/{N}  "
                  f"({100*(i+1)/N:.1f}%)  |  {rate:.0f} samples/s")
    t_total = time.perf_counter() - t_start
    print(f"[ACCURACY]  Inference complete: {t_total:.2f} s  "
          f"({t_total/N*1e6:.1f} µs/sample avg)")

    # ── Keras software baseline ───────────────────────────────────────────────
    Y_sw = None
    if not args.no_sw and os.path.exists(KERAS_MODEL):
        print("\n[ACCURACY]  Running Keras float32 software baseline...")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        model = tf.keras.models.load_model(KERAS_MODEL)
        Y_sw  = model.predict(X, batch_size=1024, verbose=0).flatten().astype(np.float32)
    elif not args.no_sw:
        print(f"[ACCURACY]  Warning: Keras model not found at {KERAS_MODEL}, skipping SW baseline.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    rows = [compute_metrics(Y, Y_fpga, "FPGA — Alveo U280 @ 181 MHz (ap_fixed<16,6> input)")]
    if Y_sw is not None:
        rows.append(compute_metrics(Y, Y_sw, "Software — Keras float32"))
        rows.append(compute_metrics(Y_sw, Y_fpga, "FPGA vs SW (fixed-point quantisation error)"))

    # ── Save CSV results ──────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "accuracy_metrics.csv"), index=False)
    pred_cols = {"y_true": Y, "y_fpga": Y_fpga}
    if Y_sw is not None:
        pred_cols["y_software"] = Y_sw
    pd.DataFrame(pred_cols).to_csv(
        os.path.join(OUTPUT_DIR, "predictions_detail.csv"), index=False)

    # ── Save timing summary (Mathys-format) ──────────────────────────────────
    pd.DataFrame({
        "Metric": ["MAE", "MSE", "R²", "Wall-time (s)", "Avg / sample (s)"],
        "Value":  [rows[0]["MAE"], rows[0]["MSE"], rows[0]["R²"],
                   t_total, t_total / N]
    }).to_csv(os.path.join(OUTPUT_DIR, "FPGA_metrics.csv"), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    ncols = 2 if Y_sw is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, y_pred, label, colour in zip(
            axes,
            [Y_fpga] + ([Y_sw] if Y_sw is not None else []),
            ["FPGA (Alveo U280)"] + (["Software (Keras)"] if Y_sw is not None else []),
            ["steelblue", "darkorange"]):
        ax.scatter(Y, y_pred, alpha=0.3, s=2, color=colour)
        ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], "r--", linewidth=1, label="Ideal")
        ax.set_xlabel("Ground Truth — Call Price")
        ax.set_ylabel(f"{label} — Predicted Call Price")
        ax.set_title(f"Predictions vs Ground Truth\n{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_vs_truth.png"), dpi=150)
    plt.close()
    print(f"\n[ACCURACY]  Results saved to: {OUTPUT_DIR}/")
    xclClose(_h)


if __name__ == "__main__":
    main()
