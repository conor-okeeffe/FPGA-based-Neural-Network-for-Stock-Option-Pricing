import hls4ml
import os

USER_PATH = os.getcwd()
PROJECT_PATH = USER_PATH + "/my_vitis_project_RF4/"
MODEL_PATH = USER_PATH + "/Models/"
PLOTS_PATH = USER_PATH + "/Plots/"

hls4ml.report.read_vivado_report(PROJECT_PATH)

