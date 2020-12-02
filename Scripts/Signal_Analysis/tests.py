# TODO: Think up of some easy tests.
# Example. Try to initialise a signal with data gaps
# Try to add multiple Gaussians
# Try to stretch / Contract and check that maintained
# ???

from os import makedirs
from numpy.core.fromnumeric import size
from helpers import Signal, EMDFunctionality
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from PyEMD import EMD

emd = EMD()

import ray
import psutil

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus, ignore_reinit_error=True)


def test_signal():

    noise = 0.05  # Percentage of noise
    gauss_std = 10
    corr_thr_list = np.round(np.arange(0.5, 1.0, 0.05), 2)

    short_sig = Signal(
        duration=1000,
        cadence=12,
        mean=30,
        name="Short",
    )

    long_sig = Signal(
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
    short = EMDFunctionality(short_sig)
    long = EMDFunctionality(long_sig)

    long.generate_windows(short)
    long.plot_all_results(
        other=short,
        corr_thr_list=corr_thr_list,
    )


# Rows are for short IMFs, columns for LONG
test_signal()

# print(arr, periods, sep="\n")
