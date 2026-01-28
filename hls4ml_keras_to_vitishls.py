# hls4ml_keras_to_vitishls.py
# Requires: Python 3.10, TF 2.14.0, hls4ml 1.1.0, sympy, NumPy < v2 (1.26.4)
#           pydot, graphviz ( for hls4ml.utils.plot_model() )

# Run the next line in a PyCharm CMD Prompt Terminal to make visible
# C:\Xilinx\Vitis_HLS\2023.1\settings64.bat

from pathlib import Path
import tensorflow as tf
import hls4ml
import os, shutil

root = os.environ.get('VITIS_HLS') or os.environ.get('XILINX_VITIS')
if not root:
    raise RuntimeError("Vitis HLS not configured. Run settings64.bat, or set VITIS_HLS/XILINX_VITIS in runtime config env variables.")

os.environ['PATH'] = os.path.join(root, 'bin') + os.pathsep + os.environ['PATH']

# Optional sanity check
if not shutil.which('vitis_hls'):
    raise RuntimeError("vitis_hls not found on PATH after update.")


# --- Paths ---
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.h5"        # use your original .h5
OUT_DIR    = ROOT / "hls4ml_prj"

CLEAN = True  # set False to preserve previous build

# --- safe cleanup ---
if CLEAN and OUT_DIR.exists():
    # extra guard: ensure we're deleting exactly our build dir under project root
    if OUT_DIR.is_dir() and OUT_DIR.parent == ROOT and OUT_DIR.name == "hls4ml_prj":
        shutil.rmtree(OUT_DIR)
        print(f"Deleted files in output directory {OUT_DIR}")
    else:
        raise RuntimeError(f"Refusing to delete suspicious path: {OUT_DIR}")

# --- Load model (inference only) ---
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(model.summary())

# --- Base HLS config ---
cfg = hls4ml.utils.config_from_keras_model(model, granularity="name")
cfg["Model"]["Precision"] = "ap_fixed<16,6>"     # global fixed-point type

# --- Per-layer ReuseFactor (edit as needed) ---
RF = {
    "dense":    160,   # 6 -> 100
    "dense_1":  160,   # 100 -> 100
    "dense_2":  160,   # 100 -> 100
    "dense_3":  160,   # 100 -> 100
    "output_1":  32,   # 100 -> 1
}
for lname, lcfg in cfg["LayerName"].items():
    if lname in RF:
        lcfg["ReuseFactor"] = RF[lname]

# --- Convert to HLS for PYNQ-Z2 (Zynq-7020) ---
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    output_dir=str(OUT_DIR),
    part="xc7z020clg400-1",
    io_type="io_stream",
    backend="Vitis" # Use Vitis for HLS support, as Vivado HLS is legacy
)

# hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

# --- Build and export bitstream ---
hls_model.build(csim=False, synth=True, cosim=False, export=True)

print("\nBitstream:", OUT_DIR / "firmware" / "myproject.bit")
print("HWH     :", OUT_DIR / "firmware" / "myproject.hwh")
