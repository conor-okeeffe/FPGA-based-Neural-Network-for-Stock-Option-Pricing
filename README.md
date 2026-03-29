# FPGA-Based Neural Network for European Stock Option Pricing

This project deploys a ~31,000 parameter deep neural network onto a Xilinx Alveo U280 FPGA card using [hls4ml](https://fastmachinelearning.org/hls4ml/), targeting real-time inference for European option pricing. The network takes six financial inputs (stock price, strike price, time to maturity, risk-free rate, volatility, dividends) and outputs a predicted option price.

The implementation uses **Reuse Factor = 10** for the hidden layers (100×100 dense, ×3), with RF = 1 kept for the smaller input and output layers. This was arrived at after earlier RF = 4 attempts hit Vivado routing congestion — the DSPs were at 100% utilisation and congestion levels were hitting 7, particularly around the DSP columns. Bumping to RF = 10 with `Strategy = Resource` got it through routing cleanly.

---

## Main Script

**`vitis_hls_alveo_u280.py`** is the entry point for the whole flow. It loads the trained Keras model, sets up the hls4ml config (precision, reuse factors, strategy), and kicks off HLS synthesis and Vitis implementation targeting the U280 part (`xcu280-fsvh2892-2l-e`). The output directory from this script is what gets linked and built into the `.xclbin` bitstream. If you're looking to reproduce or modify the implementation, start here.

The fixed-point precision used throughout is `ap_fixed<16,6>` — 16-bit signed with 6 integer bits. The output layer is forced to `ap_fixed<32,16>` so the DMA buffer maps cleanly to `np.int32` on the host side.

---

## Benchmarking Scripts

There are two sets of benchmarking scripts corresponding to two different kernel interface implementations that were tested:

**AXI-Lite interface (original approach)**

- `fpga_accuracy.py` — runs the full test set through the FPGA and reports MAE, MSE, RMSE and R², with an optional Keras float32 software baseline for comparison
- `fpga_compatible_accuracy.py` — same accuracy comparison but filtered to the 29,564 samples that don't overflow `ap_fixed<16,6>`. There are 37 test samples where stock price or dividends exceed the representable range, which cause silent overflow on hardware
- `fpga_latency_single.py` — measures end-to-end single-sample latency via AXI-Lite register access. Each inference requires 6 PCIe transactions (~3 µs each), so host communication completely dominates over the actual ~280 ns FPGA compute time
- `fpga_throughput_batch.py` — sweeps batch sizes 1/100/1000/10000 and reports throughput, matching the same format used in the Qualcomm RB3 Gen2 CPU/NPU comparison scripts

**io_stream / AXI Master interface (later approach)**

- `iostream_accuracy.py` — accuracy verification for the streaming kernel, which batches the entire test set into a single kernel call via HBM buffers rather than individual AXI-Lite transactions
- `iostream_latency.py` — measures both end-to-end latency (including HBM transfers) and compute-only latency separately. The inner myproject core runs at 100 MHz with ~24 cycle latency per sample
- `iostream_throughput.py` — sweeps batch sizes and breaks down transfer time vs kernel execution time, reporting total throughput in samples/s

---

## Utility Scripts

- `model_details_view.py` — quick sanity check, prints the Keras model summary and lists all layers
- `post_export_views.py` — reads and prints the Vivado HLS synthesis report for a given project directory using `hls4ml.report`

---

## Notes

The project was built and tested with Vitis 2023.2 and XRT on Ubuntu. The `.xclbin` file is not included in this repo due to size. The Keras model should be placed at `Models/model.h5` relative to the working directory before running any of the scripts.
