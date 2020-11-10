# TODO: Think up of some easy tests.
# Example. Try to initialise a signal with data gaps
# Try to add multiple Gaussians
# Try to stretch / Contract and check that maintained
# ???

from numpy.core.fromnumeric import size
from helpers import Signal, EMDFunctionality
import array
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from PyEMD import EMD

emd = EMD()


def test_signal():

    window_displacement = 12
    noise = 0.05  # Percentage of noise
    gauss_std = 10

    short_sig = Signal(
        save_folder="/mnt/sda5/Python/InsituEMDCorrelation/Scripts/Signal_Analysis/tests/short/",
        duration=1000,
        cadence=12,
        mean=30,
        name="Short",
    )

    long_sig = Signal(
        save_folder="/mnt/sda5/Python/InsituEMDCorrelation/Scripts/Signal_Analysis/tests/long/",
        duration=5000,
        cadence=12,
        mean=10,
        name="Long",
    )

    # Create the gaussian and add it into short
    short_sig.create_gaussian_add(
        where=500,
        duration=300,
        std=gauss_std,
    )

    # Include short signal into long
    long_sig.inc_custom_signal(short_sig.gaussian_curve, where=2000)

    # Add the noise
    short_sig.add_noise(noise)
    long_sig.add_noise(noise)

    # Normalise between 0 and 1
    short_sig.data = Signal.normalize_signal(short_sig.data)
    long_sig.data = Signal.normalize_signal(long_sig.data)

    # Add EMD functionality to either signal
    short = EMDFunctionality(short_sig, filter_imfs=False)
    long = EMDFunctionality(long_sig, filter_imfs=False)

    arr, periods = long.generate_windows(short)

    return arr, periods


# Rows are for short IMFs, columns for LONG
arr, periods = test_signal()
print(arr, periods)

# print(arr, periods, sep="\n")
