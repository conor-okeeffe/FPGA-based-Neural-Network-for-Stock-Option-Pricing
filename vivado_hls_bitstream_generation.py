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
# 1. Input Layer (Small)
# 6 inputs -> 100 outputs.
# Keep RF=60. (Uses ~10 DSPs).
# Note: This technically uses the 'risky' template because 60 > 6,
# but 10 DSPs is small enough that Vivado won't crash.
config['LayerName']['dense']['ReuseFactor'] = 60

# 2. Dense_1: RF=100 (Safe Template)
# 10,000 ops / 100 = 100 Multipliers.
# Matches Input dimension (100) -> Uses stable template.
config['LayerName']['dense_1']['ReuseFactor'] = 100

# 3. Dense_2: RF=100 (Safe Template)
# 10,000 ops / 100 = 100 Multipliers.
config['LayerName']['dense_2']['ReuseFactor'] = 100

# 4. Dense_3: RF=100 (Safe Template)
# 10,000 ops / 100 = 100 Multipliers.
config['LayerName']['dense_3']['ReuseFactor'] = 100

# 5. Output Layer
config['LayerName']['output_1']['ReuseFactor'] = 10

# TOTAL RESOURCE REQUEST:
# ~310 Multipliers.
# Board Capacity: 220 DSPs.
# Outcome: 220 DSPs used (100%), ~90 Multipliers implemented in LUTs.

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
