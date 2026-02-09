import shutil
import os
from tensorflow.keras.models import load_model

# 0. Putting Vitis on path - Linux approach
vitis_hls_2023_2_path = '/user/masters/OkeeffeCJ/tools/Xilinx/Vitis_HLS/2023.2'
# Update environ variable to 2023.2 Vitis
os.environ['XILINX_VITIS'] = vitis_hls_2023_2_path
# Add Vitis 2023.2 to PATH
os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
# Sanity Check of correct Vivado version
print("Targeting Vitis path:", shutil.which("vitis_hls"))

import hls4ml
model = load_model("Models/model.h5")

# 1. Create config
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# 2. Set config options
# The Part number for Alveo U280 is xcu280-fsvh2892-2l-e
config['Model']['Strategy'] = 'Latency' # Instead of Resource strategy as before, due to large FPGA fabric available now
config['Model']['ReuseFactor'] = 1     # Global Default RF
config['Model']['Precision'] = 'ap_fixed<16,6>' # Standard fixed point starting place
config['Backend']['ClockPeriod'] = 3.33  # 3.33ns corresponds to 300MHz - tweak this if timing fails

# 3. Per layer reuse factor setting for future tweaking
# Input Layer (Small)
# 6 inputs -> 100 outputs.
config['LayerName']['dense']['ReuseFactor'] = 1

#   ------- "Hidden" Layers: 100x100, 10,000 MUL operations per layer -------------
#  Dense_1:
config['LayerName']['dense_1']['ReuseFactor'] = 1

# Dense_2:
config['LayerName']['dense_2']['ReuseFactor'] = 1

# Dense_3:
config['LayerName']['dense_3']['ReuseFactor'] = 1
#   -------                                                          -------------

# Output Layer:
# 100 inputs -> 1 output
config['LayerName']['output_1']['ReuseFactor'] = 1

# 4. Print model config
from pprint import pprint # PrettyPrint for config printing - replaces old plotting.print_dict
print("-----------------------------------")
pprint(config)
print("-----------------------------------")

# 5. Generate HLS for model
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='my_vitis_project', # new dir for Vitis / Alveo approach
    backend='Vitis',  # Target Vitis backend instead of Vivado
    part='xcu280-fsvh2892-2l-e'
)

# 6. Compile HLS model code
hls_model.compile()

# 7. Build the Hardware (Synthesis and Bitstream)
# Note: This step can take 1-4 hours for an Alveo U280
# NB: Can comment this out and cd my_vitis_project and then run $ make all
hls_model.build(
    csim=False,       # Skip C-simulation (already done in .compile)
    synth=True,      # Run Vivado HLS Synthesis
    vsynth=True,     # Run Logic Synthesis
    export=True,     # Export as a Vitis Kernel (.xo)
    # Change bitstream = True for actual implementation on board.
    # For minor iterations, bitstream = False will return the Vivado report in minutes rather than hours
    # but will not actually deploy model to card.
    bitstream=False   # Link the kernel to the U280 shell to create .xclbin
)

# 8. View the Reports
# This will show you exactly how many BRAMs/DSPs you used and the latency
hls4ml.report.read_vivado_report('my_vitis_project')