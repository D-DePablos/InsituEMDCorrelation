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

<<<<<<< HEAD
    long.generate_windows(short)
    long.plot_all_results(
        other=short,
        corr_thr_list=corr_thr_list,
    )


# Rows are for short IMFs, columns for LONG
test_signal()
=======
    def gen_wind(short=short, long=long):
        """
        Generates the windows and stores in array
        """
        # Create pointers to relevant windows? -> DONE
        # Do I need to store IMF actual values? Maybe not?
        # Assume we will use window disp. Of 1 dt
        # TODO: Could maybe use a generator (yield keyword) to save memory?
        window_length = len(short.s)
        short_imfs = emd.emd(S=short.s, T=short.t)[:-1]  # Ignore residual here as well
        short_periods = EMDFunctionality.determine_periodicities(
            time=short.t, imfs=short_imfs
        )

        arr = []
        periods = []

        def create_period_matrix(pA, pB, pfilter=(5, 20)):
            """
            Based on two lists of periodicities, make matrix with information of which periods are good
            """

            period_m = np.zeros(shape=(len(pA), len(pB)))

            # Probably the fastest possible solution :(
            for i, p_a in enumerate(pA):
                if pfilter[0] < p_a < pfilter[1]:
                    for j, p_b in enumerate(pB):
                        if pfilter[0] < p_b < pfilter[1]:
                            period_m[i, j] = 1

            return period_m

        for i in range(len(long.s) - window_length + 1):  # Range of motion known here
            # Selects the time and data for relevant window
            _long_time = long.t[i : window_length + i]
            _long_data = long.s[i : window_length + i]

            # Can reference time with a single digit = tstart of ts
            _long_imfs = emd.emd(S=_long_data, T=_long_time)[:-1]  # Ignore trend here
            _long_periods = EMDFunctionality.determine_periodicities(
                time=_long_time, imfs=_long_imfs
            )

            # Find correlation and periodicity matrices
            corr_results = EMDFunctionality.corr_coeff_2d(short_imfs, _long_imfs)
            period_matrix = create_period_matrix(short_periods, _long_periods)

            # print(f"{short_periods} to {_long_periods}")
            # print(corr_results)
            # print(period_matrix)
            # print("-------------------------------------------------------------")

            # corr_results[0] = Correlations for first short IMF, all long
            arr.append(corr_results)  # Kept as a list may be good? -> Differing size
            periods.append(period_matrix)

        return arr, periods

    return gen_wind()


# Rows are for short IMFs, columns for LONG
arr, periods = test_signal()
>>>>>>> parent of fb035f9 (Added Ray Multiprocessing)

# print(arr)
