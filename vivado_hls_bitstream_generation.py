# For quantization, look at part 4 of tutorial, and then first
# 2 steps of part 7a -- omitting these for now (26/1)
import shutil
import os
from tensorflow.keras.models import load_model
# 0. Putting vivado on path - Linux approach
vivado_2020_1_path = '/user/masters/OkeeffeCJ/tools/Xilinx/Vivado/2020.1'
# Update environ variable to 2020.1 Vivado
os.environ['XILINX_VIVADO'] = vivado_2020_1_path
# Add Vivado 2020.1 to PATH
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
# Sanity Check of correct Vivado version
print("Targeting Vivado path:", shutil.which("vivado_hls"))

import hls4ml
model = load_model("Models/model.h5")

# 1. Create config
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# 2. Set config options
# The Part number for PYNQ-Z2 is xc7z020clg400-1
config['Model']['Strategy'] = 'Resource' # Required for high reuse factors
config['Model']['ReuseFactor'] = 200     # Global Default RF     150 DSPs / 220, room for optimization
config['Model']['Precision'] = 'ap_fixed<16,6>' # Standard fixed point starting place

# 3. Per layer reuse factor setting for future tweaking
# --- Layer: dense (6 inputs -> 100 outputs) ---
# 600 mults. RF =60 = 10 DSPs
config['LayerName']['dense']['ReuseFactor'] = 60

# --- Layer: dense_1 (100 -> 100) ---
# 10,000 mults. RF=200 = 50 DSPs.
config['LayerName']['dense_1']['ReuseFactor'] = 200

# --- Layer: dense_2 (100 -> 100) ---
# 10,000 mults. RF=200 = 50 DSPs.
config['LayerName']['dense_2']['ReuseFactor'] = 200

# --- Layer: dense_3 (100 -> 100) ---
# 10,000 mults. RF=200 = 50 DSPs.
config['LayerName']['dense_3']['ReuseFactor'] = 200

# --- Layer: output_1 (100 -> 1) ---
# 100 mults. RF =10 = 10 DSPs
config['LayerName']['output_1']['ReuseFactor'] = 10

# Total DSP use for RF = [60, 200, ... , 10] = 170 / 220

# 4. Print model config
from pprint import pprint # PrettyPrint for config printing - replaces old plotting.print_dict
print("-----------------------------------")
pprint(config)
print("-----------------------------------")

# 5. Generate HLS for model
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='my_pynq_project',
    backend='VivadoAccelerator',  # Target Vivado backend instead of Vitis
    board='pynq-z2'
)

# 6. Compile HLS model code
hls_model.compile()
