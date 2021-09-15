# Set up UNSAFE_EMD_DATA_PATH: global variable
from sys import path

BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"
path.append(f"{BASE_PATH}Scripts/")

from Scripts.EMD.pspSoloEMD import *

# Create new orbits plot (plot_top_down) for different data
