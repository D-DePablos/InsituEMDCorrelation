# TODO: Think up of some easy tests.
# Example. Try to initialise a signal with data gaps
# Try to add multiple Gaussians
# Try to stretch / Contract and check that maintained
# ???

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
        name="Long"
    )

    # Create the gaussian and add it into short
    short_sig.create_gaussian_add(
        where=500, duration = 300, std=gauss_std,
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


    def cross_correlate(imfs_short, imfs_long):
        """
        This function finds cross correlation at different positions
        Could possibly be useful to remember implied periodicities
        To enable filtering after the fact

        """
        # TODO: Simply cross-correlate the two IMF sets
        # TODO: Find a memory efficient method if possible.
        # Check if there are any guides online to do this type of thing

        pass

    def corr_coeff_2d(A, B):
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

    def find_period(timeseries):
        """
        Determine number of maxima and minima of a timeseries
        """

    def determine_periodicities(time, imfs):
        """
        Input a two-dimensional array and return an array of periodicities
        :param imfs: Numpy array with IMF information on rows
        """

        # no_extrema = list(map(emd.find_extrema,{"T":time, "S":imfs[:,None]}) )  # All rows
        no_extrema = [len(emd.find_extrema(time, imf)) for imf in imfs]
        #TODO: Implement to calculate periodicity
        pass


    def gen_wind(short=short, long=long):
        """
        Generates the windows and stores in array
        """
        # Create pointers to relevant windows? -> DONE
        # Do I need to store IMF actual values? Maybe not?
        # Assume we will use window disp. Of 1 dt
        # TODO: Could maybe use a generator (yield keyword) to save memory?
        window_length = len(short.s)
        short_imfs = emd.emd(S=short.s, T=short.t)
        determine_periodicities(time=short.t, imfs=short_imfs)

        arr = []

        for i in range(len(long.s) - window_length + 1): # Range of motion known here
            # Selects the time and data for relevant window
            _long_time = long.t[i: window_length + i]
            _long_data = long.s[i: window_length + i]

            # Can reference time with a single digit = tstart of ts
            long_imfs = emd.emd(S=_long_data, T =_long_time)
            corr_results = corr_coeff_2d(short_imfs, long_imfs)

            # corr_results[0] = Correlations for first short, all long
            arr.append(corr_results)  # Kept as a list may be good?

            pass

        return arr

    return gen_wind()

arr = test_signal()

print(arr)
