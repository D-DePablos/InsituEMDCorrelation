from os import makedirs, getcwd
from os.path import isfile
from sys import path, modules
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import signal as scipy_sig
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from glob import glob
import matplotlib
from PyEMD import EMD, Visualisation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
from datetime import timedelta

# Quick fix for cross-package import
path.append(f"/mnt/sda5/Python/Proj_1/Scripts/")
from Imports.Data_analysis import Tools

emd = EMD()
vis = Visualisation()

# Multiprocessing
import ray
import psutil

num_cpus = psutil.cpu_count(logical=False)

from matplotlib import rc

font = {"family": "DejaVu Sans", "weight": "normal", "size": 20}
rc("font", **font)


@ray.remote
def find_corr_periods(
    long_data, short_time, short_imfs, short_periods, window_length, pfilter, idx_list
):

    corr_all = {}
    period_all = {}

    for idx in idx_list:
        _long_data = long_data[idx : window_length + idx]

        # Can reference time with a single digit = tstart of ts
        _long_imfs = emd.emd(S=_long_data, T=short_time)[:-1]  # Ignore trend here
        _long_periods = EMDFunctionality.determine_periodicities(
            time=short_time, imfs=_long_imfs
        )

        # Find correlation and periodicity matrices
        corr_results = EMDFunctionality.corr_coeff_2d(short_imfs, _long_imfs)
        period_matrix = EMDFunctionality.create_period_matrix(
            short_periods,
            _long_periods,
            pfilter=pfilter,
        )

        corr_all[idx] = corr_results
        period_all[idx] = period_matrix

    return corr_all, period_all


class Heatmap:
    @staticmethod
    def heatmap(
        data: np.array,
        row_labels,
        col_labels,
        valid_data=None,
        ax=None,
        cbar_kw={},
        cbarlabel="",
        vmin_vmax=(-1, 1),
        **kwargs,
    ):
        """
        Create a heatmap from a numpy areay and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        valid_data
            Data which needs to be highlighted
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        vmin_vmax
        |   Minimum and maximum values for image (Hence colourbar)
        **kwargs
            All other arguments are forwarded to `imshow`.
        """
        vmin, vmax = vmin_vmax

        if not ax:
            ax = plt.gca()

        # Plot the heatmap

        im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)
        if valid_data is not None:
            ax.imshow(valid_data, vmin=0, vmax=1, alpha=0.8, cmap="Greys")

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)

        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    @staticmethod
    def annotate_heatmap(
        im,
        data=None,
        valfmt="{x:.2f}",
        textcolors=("black", "white"),
        threshold=None,
        **textkw,
    ):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A list or array of two color specifications.  The first is used for
            values below a threshold, the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **textkw
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """
        import matplotlib

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.0

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center", verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(
                    j, i, valfmt(data[i, j], None), fontsize="large", **kw
                )
                texts.append(text)

        return texts


class Signal:
    """
    This class creates a Signal and provides functions to deform and extract it.
    """

    def __init__(
        self,
        save_folder=None,
        duration=0,
        cadence=0,
        mean=0,
        sig_type=(True, "flat"),
        name="Unnamed",
        custom_data=False,
        **kwargs,
    ):
        """
        This function creates a single line with a given mean_1, with a selected observation cadence
        :param save_folder: Folder to save signal files into
        :param duration: Duration of the signal to be created in seconds
        :param cadence: Observation cadence, within duration
        :param mean: Average value
        :param sig_type: Whether Gaussian or not
        :param real_signal: Whether we are using a real signal or not
        :param custom_data: whether custom data or not!
        """
        assert cadence != 0, "Cadence set at 0"
        self.cadence = cadence
        self.save_folder = save_folder
        self.imfs = []
        self.residue = []
        self.name = name
        self.data_smoothed = []
        self.location_signal_peak = []
        self.saveformat = "pdf"
        # Minimum and Maximum Periods
        self.pmin = None
        self.pmax = None

        # How much the peak of the artificial Signal sticks out
        self.peak_above = None
        self.noise_perc = None

        if custom_data is False:
            self.duration = duration
            self.mean = mean
            self.noise = 0
            self.number_of_signals = 0
            self.location_signal_peak = []  # Set the location signal peak as false

            # Time and data for artificial cases
            self.time = np.arange(0, duration, step=cadence)
            self.true_time = None
            if sig_type[0] is True:
                # data for observations is a flat line that we add on to
                self.data = np.repeat(float(self.mean), duration / cadence + 1)
            else:
                raise ValueError("different types of signal not implemented")

            for key, value in kwargs.items():
                if key == "gaussian":
                    self.gaussian_curve = value
                if key == "corr_mode":
                    self.correlation_mode = value

        else:
            # Check the type of data that is being fed:
            if type(custom_data) == np.ndarray:
                self.data = custom_data  # If array, can set immediately
                self.true_time = None

            elif type(custom_data) == pd.DataFrame or type(custom_data) == pd.Series:

                if type(custom_data.index) == pd.DatetimeIndex:
                    self.true_time = pd.to_datetime(custom_data.index)

                # When using pandas dataframe
                if type(custom_data) is pd.DataFrame:
                    for column in list(custom_data):
                        self.data = custom_data[column].values
                else:
                    self.data = custom_data.values

                assert (
                    self.true_time is not None
                ), "Was unable to set true time using either index or Time Column"

                assert (
                    self.data is not None
                ), "Was unable to set true time using either index or Time Column"

            else:
                raise ValueError(f"Did not pass a valid type {type(custom_data)}")

            # Generate constant time array for IMFs
            self.duration = self.cadence * len(self.data)
            self.time = np.arange(0, self.duration, step=self.cadence)

    def __repr__(self):
        """
        Good representation of the object
        """
        return f"Signal({self.save_folder}, cadence={self.cadence})"

    def __str__(self):
        """
        What to print out when called to print
        """
        return f"Signal object with a cadence of {self.cadence}s. Length of {self.duration}"

    @staticmethod
    def normalize_signal(s):
        """
        Normalises an input signal
        ==========================
        Parameters
        :param s: Input signal

        returns normalised(s), now taking values between 0 and 1
        """
        minima = np.min(s)
        maxima = np.max(s)
        for i, x_i in enumerate(s):
            s[i] = (x_i - minima) / (maxima - minima)
        return s

    @staticmethod
    def detrend(s, box_width=200):
        """
        Detrend a dataseries and express as percentage difference from new mean
        :param s: Used to be self.data -> Change into static method
        :param box_width: WIdth of box used in averaging
        """
        from astropy.convolution import convolve, Box1DKernel

        s_conv = convolve(s, Box1DKernel(box_width), boundary="extend")
        s_pdiff = 100 * (s - s_conv) / s_conv
        return s_pdiff  # Can now set self.data to this

    def decimate_to(self, other):
        """
        Reshape the array into a given cad. Note that total time is preserved.
        :param other: other signal object, with objective cad to decimate down to
        """
        # Cut down into given cad
        cad_factor = int(other.cadence / self.cadence)

        if self.true_time is not None:
            self.true_time = self.true_time[::cad_factor]  # Lose time information

        # Get rid of NA values
        _data = self.data
        nans, x = np.isnan(_data), lambda z: z.nonzero()[0]
        _data[nans] = np.interp(x(nans), x(~nans), _data[~nans])

        _data = _data[::cad_factor]
        self.data = _data

        # Cadence and time
        self.cadence = other.cadence
        self.time = np.arange(0, len(self.data) * self.cadence, step=self.cadence)

        # Update name to reflect decimation
        self.name = f"Decimated {self.name}"

    def plot(self, save_to=False, labels=("Time (s)", "Data (arb.units)"), show=True):
        from matplotlib import rc

        with_noise = None if self.noise_perc == 0 else f"| {self.noise_perc}% noise"
        title = f"{self.name} signal | sampled at {self.cadence}s {with_noise}"

        _ = plt.figure(figsize=(10, 8))
        plt.plot(self.time, self.data, color="black")
        plt.title(title)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        if save_to:
            plt.savefig(f"{save_to}{self.name}_{self.cadence}s.{self.saveformat}")

        if show:
            plt.show()

        plt.close("all")

    #################################################################################
    # Artificial signal functions
    def add_noise(self, noise_std=0):
        """
        Add white noise with a given standard deviation - Changes data itself!
        :param noise_std: standard deviation for randomly distributed noise - can be linked to how big original data looks
        """
        self.peak_above = round(self.data.max(), 2) - self.mean
        if noise_std != 0:
            self.noise = np.random.normal(0, noise_std, size=len(self.data))
            self.data = self.data + self.noise
            self.noise_perc = f"{(noise_std/self.peak_above) * 100:.1f}"
        else:
            self.noise_perc = "0"

    def create_gaussian_add(self, where, duration, std):
        """
        Add a Gaussian into data. Changes self.data Used as "signal"
        :param duration: Duration of the signal (generally relative to the data array)
        :param std: Standard dev. for the created signal
        :param where: First value where gaussian included
        """

        # Mask an array from the initial point for a duration with a gaussian
        mask = ma.masked_inside(self.time, where, where + duration)
        mask = ma.getmask(mask)

        if len(self.data) != len(mask):
            mask = mask[:-1]
        extent = np.count_nonzero(mask)

        # Create a gaussian to put inside masked array
        gaussian = scipy_sig.general_gaussian(M=extent, p=1, sig=std)
        gaussian = gaussian - (
            gaussian[0] - self.data.mean()
        )  # Only works if flat signal!
        #

        # This is the specific signal that has been placed inside.
        self.gaussian_curve = gaussian
        try:
            self.data[mask] = self.gaussian_curve
        except IndexError:
            self.data[mask] = self.gaussian_curve[-1]

        self.location_signal_peak = where

    def mod_gaussian(self, alter_mode, show=False):
        """
        Modify the gaussian curve in a given way
        :param alter_mode: Mode to modify gaussian in. stretch, multiply, add
        :param show: Whether to show
        """
        # Start by setting it, then modify it in each of the steps.
        print(f"Modifying the signal using alteration {alter_mode}")
        gaussian = self.gaussian_curve

        if not alter_mode:
            return gaussian
        else:
            for key in alter_mode:

                # These would have to be applied multiple times
                if key == "stretch":
                    # Stretch the signal by ... Duplicating or repeating values n times?
                    temp_gaussian = []
                    for i, data in enumerate(gaussian):
                        if data == gaussian[-1] and i > (len(gaussian) / 2):
                            # For the last datapoint, have same twice
                            for n_g in range(alter_mode[key]):
                                temp_gaussian.append(gaussian[i])
                        else:
                            dy = gaussian[i + 1] - gaussian[i]
                            dx = (i + 1) - i
                            grad = dy / dx
                            # Append twice, once at same value, once at half the gradient up/down
                            for n_g in range(alter_mode[key]):
                                # Doing it like this means that you never reach the higher value - desired behaviour!
                                temp_gaussian.append(
                                    gaussian[i]
                                    + (grad * n_g / len(range(alter_mode[key])))
                                )

                    gaussian = np.array(temp_gaussian)
                elif key == "height_mod":

                    temp_gaussian = []

                    for i in range(len(gaussian)):

                        # Find gradient at each point by subtracting next to current
                        dy = gaussian[i + 1] - gaussian[i]
                        dx = (i + 1) - i
                        grad = dy / dx
                        n = alter_mode[key]

                        if i == len(gaussian) - 1:
                            temp_gaussian.append(gaussian[i])

                        else:
                            temp_gaussian.append(gaussian[i] + n * grad)

                    gaussian = np.array(temp_gaussian)
                elif key == "multiply":

                    temp_gaussian = gaussian * alter_mode[key]

                    return temp_gaussian
                else:
                    print(f"Mode {key} not supported")

        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.gaussian_curve, color="black")
        plt.subplot(1, 2, 2)
        plt.plot(gaussian, color="black")

        if show:
            plt.show()

        return gaussian

    def inc_custom_signal(self, signal, where):
        """
        Include a custom signal from another dataset
        :param signal: What signal to include
        :param where: Where in time to include signal
        """
        # In the case where there is a signal peak already
        self.location_signal_peak.append(where)
        if self.number_of_signals == 0:
            mask = ma.masked_inside(self.time, where, where + len(signal) * 12 - 1)
            mask = ma.getmask(mask)

            # Only works if flat signal!
            signal = signal - (signal[0] - self.mean)

            self.gaussian_curve = signal
            self.data[mask] = signal

            self.name = f"Modified {self.name}"  # First time change the title

        else:
            # This simply replaces and that is not correct. Should build up
            mask = ma.masked_inside(self.time, where, where + len(signal) * 12 - 1)
            mask = ma.getmask(mask)

            # Only works if flat signal!
            signal = signal - (signal[0] - self.mean)

            self.gaussian_curve = signal
            self.data[mask] = signal + self.data[mask] - self.mean
            self.name = f"Duplicated {self.name}"  # First time change the title

        self.number_of_signals += 1


class EMDFunctionality(Signal):
    """
    Class to give functionality to a given signal object. Allows for things like EMD
    """

    def __init__(
        self,
        signal: Signal,
        norm=True,
        pfilter=(0, 1000),
        saveformat="pdf",
    ):
        self.signalObject = signal
        self.name = signal.name
        self.save_folder = signal.save_folder

        self.cadence = signal.cadence
        self.base_signal = signal.data
        self.pfilter = pfilter
        self.saveformat = saveformat

        if norm:
            self.s = self.normalize_signal(signal.data)
        else:
            self.s = signal.data

        self.t = signal.time
        self.true_time = signal.true_time

        self.hitrate = 0
        self.table = None

    def __repr__(self):
        """
        Representation of Signal Object
        """
        return f"EMDFuncionality({self.name})"

    @staticmethod
    def emd_and_save(s, t, save_folder, save_name, plot=False):
        """
        Generate ALL EMDs for a relevant timeseries and save them in a numpy file for later use
        Checks if the file exists already to not repeat efforts
        Input params:
        :param s: signal (data)
        :param t: time array
        :param save_folder: Path to save relevant long or short
        :param save_name: Name to save (generally uses time references)

        """
        makedirs(save_folder, exist_ok=True)
        saved_npy = f"{save_folder}{save_name}.npy"

        try:
            imfs = np.load(saved_npy)
            return imfs
        except FileNotFoundError:
            pass

        # Will always use EMD. Will always get imfs and residue separately
        imfs = emd.emd(S=s, T=t)

        if plot:
            plt.figure()
            for imf in imfs:
                plt.plot(imf)
            # plt.show()

        # Find relevant timeID
        time_id = f"time_{t[0]:04d}-{t[-1]:04d}"
        np.save(f"{save_folder}{time_id}.npy", t)
        np.save(saved_npy, imfs)  # Keeps noise as we can do filtering later

        return imfs

    @staticmethod
    def check_imf_periods(
        t, imfs, period_filter=True, pmin=None, pmax=None, filter_low_high=(0, 0)
    ):
        """
        Check period of several IMFs in minutes. Uses values from SignalFilter class definition
        """
        if period_filter:
            if pmin == None or pmax == None:
                raise ValueError(
                    f"pmin, pmax value not valid ({pmin}, {pmax}) if IMF number filtering is set to None"
                )
        N = t[-1]

        Period_valid = np.ndarray((len(imfs), 2))

        if pmin is not None and filter_low_high == (0, 0):
            for i, imf in enumerate(imfs):
                # Maybe not the nicest use of find_extrema
                n_extrema = len(emd.find_extrema(T=t, S=imf)[0])

                if n_extrema != 0 and np.std(imf) > 0.00001:
                    P = 2 * N / n_extrema / 60  # Period of each of the IMFs in minutes
                    Period_valid[i, 0] = True if pmin < P < pmax else False
                    Period_valid[i, 1] = P

                else:
                    Period_valid[i, 0] = False
                    Period_valid[i, 1] = np.nan

        else:
            if len(imfs) < (filter_low_high[0] + filter_low_high[1]):
                raise ValueError("Filter larger than IMF length")

            Period_valid[:, 0] = False
            # print(f"Invalidating {filter_low_high[0]}:{len(imfs)} - {filter_low_high[1]}")
            Period_valid[filter_low_high[0] : len(imfs) - filter_low_high[1], 0] = True
            Period_valid[:, 1] = np.nan

        return Period_valid

    @staticmethod
    def corr_coeff_2d(A, B):
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA ** 2).sum(1)
        ssB = (B_mB ** 2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    @staticmethod
    def find_period(t, extrema):
        """
        Determine number of maxima and minima of a length of time t and a number of extrema
        returns period in same time units as t (default seconds)
        """
        if extrema != 0:
            period = 2 * t / extrema
            return np.round(
                period / 60, 2
            )  # Assume seconds and convert to minutes here

        else:
            print("Number too large")
            return np.round(t * 2 / 60, 2)  # Very big number

    @staticmethod
    def determine_periodicities(time, imfs):
        """
        Input a two-dimensional array and return an array of periodicities
        :param imfs: Numpy array with IMF information on rows
        Converts to minutes
        """
        no_extrema = [len(emd.find_extrema(T=time, S=imf)[0]) for imf in imfs]

        periodicities = [
            EMDFunctionality.find_period(time[-1], extrema + 1)
            for extrema in no_extrema
        ]
        return periodicities

    @staticmethod
    def create_period_matrix(
        pA, pB, pfilter
    ):  # Must make static method to work with Ray
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

    def generate_windows(self, other):
        """
        Generates the windows and stores in array
        Assumes dt of 1 timestep
        self must be the short array
        other is the long array
        """
        # Do I need to store IMF actual values? Maybe not?
        # Assume we will use window disp. Of 1 dt
        # TODO: Could maybe use a generator (yield keyword) to save memory?
        assert len(self.s) > len(
            other.s
        ), "Please alter order of call such that len(self) < len(other)"

        long = self
        short = other

        self.window_length = len(short.s)  # Length of the window used for long dataset
        short_imfs = emd.emd(S=short.s, T=short.t)[:-1]  # Ignore residual here as well
        short_periods = EMDFunctionality.determine_periodicities(
            time=short.t,
            imfs=short_imfs,
        )  # First determine periodicities and then use them

        # Prepare and apply multiprocessing for faster calculations
        long_data = ray.put(long.s)  # The values
        short_time = ray.put(short.t)  # The time information

        # Multiprocessing now
        no_displacements = len(long.s) - self.window_length
        start_idx = np.arange(no_displacements + 1)
        split_array = np.array_split(start_idx, num_cpus)

        all_dicts = ray.get(
            [
                find_corr_periods.remote(
                    long_data,
                    short_time,
                    short_imfs,
                    short_periods,
                    window_length=self.window_length,
                    pfilter=self.pfilter,
                    idx_list=split_array[i],
                )
                for i in range(num_cpus)
            ]
        )

        # Put dictionaries together and into a list (accessed by index)
        corr_matrix_all = list(
            {
                **all_dicts[0][0],
                **all_dicts[1][0],
                **all_dicts[2][0],
                **all_dicts[3][0],
            }.values()
        )

        period_matrix_all = list(
            {
                **all_dicts[0][1],
                **all_dicts[1][1],
                **all_dicts[2][1],
                **all_dicts[3][1],
            }.values()
        )

        # Returns lists with all relevant arrays of Corr and Periodicities
        self.correlation_matrix_all = corr_matrix_all
        self.period_matrix_all = period_matrix_all

        return corr_matrix_all, period_matrix_all

    def plot_all_results(
        self, other, corr_thr_list, expected_location_list=None, save_folder=None
    ):

        """
        Does this from memory. Might be ok? There could be issues with not storing long information
        Assumes that you have true_time
        """

        # TODO: Add functionality for test scenarios -> Just need to add new, non pfilter based filtering

        corr_locations = np.zeros(
            shape=(len(self.correlation_matrix_all), len(corr_thr_list))
        )

        mid_times = []

        for height, (corr_matrix, period_matrix) in enumerate(
            zip(self.correlation_matrix_all, self.period_matrix_all)
        ):
            mid_index = int(np.round(self.window_length / 2 + height))
            if self.true_time is not None:
                _rel_time = self.true_time[mid_index]

            else:
                _rel_time = self.t[mid_index]

            mid_times.append(_rel_time)

            # Make temporary copies to change parameters
            _corr_matrix = corr_matrix.copy()
            _corr_matrix[period_matrix == 0] = np.nan

            for index, corr_thr in enumerate(corr_thr_list):
                nhits = len(_corr_matrix[np.abs(_corr_matrix) > corr_thr])
                corr_locations[height, index] = nhits

                # Plot some just to see it's working!
                # if nhits > 0:
                #     print(_rel_time, corr_thr, nhits)

            # TODO: Add Heatmap plotting

        if self.true_time is not None:
            short_duration = self.true_time[self.window_length] - self.true_time[0]
            time_axis = self.true_time
        else:
            short_duration = self.t[self.window_length] - self.t[0]
            time_axis = self.t

        fig, axs = plt.subplots(2, sharex=True, figsize=(20, 10))

        ax = axs[0]
        ax.plot(time_axis, self.s, color="black", label=self.name)
        ax.set_title(f"Filter {self.pfilter} | Correlated against {other.name}")
        ax.set_ylabel(f"Normalised Detrended {self.name}")
        ax.xaxis.grid(True)

        if self.true_time is not None:
            ax.set_xlim(
                self.true_time[0] - timedelta(hours=3),
                self.true_time[-1] + timedelta(hours=3),
            )
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
            ax.xaxis.set_tick_params(rotation=60)

        else:  # For test cases
            ax.set_xlim(
                self.t[0] - 600,
                self.t[-1] + 600,
            )

        # SECOND AXIS
        ax2 = axs[1]
        possibleColors = {
            "1": "green",
            "2": "blue",
            "3": "red",
        }

        for height in range(len(corr_locations)):
            midpoint = mid_times[height]
            pearson_corrs = corr_locations[height, :]

            for index, corr_label in enumerate(corr_thr_list):
                if pearson_corrs[index] != 0:
                    try:
                        _color = possibleColors[f"{int(pearson_corrs[index])}"]
                    except KeyError:
                        _color = "red"

                    _alpha = 0.35
                    ax2.bar(
                        midpoint,
                        corr_label,
                        width=short_duration,
                        color=_color,
                        edgecolor="white",
                        alpha=_alpha,
                        zorder=1,
                    )

        if expected_location_list is not None:
            for expected_location in expected_location_list:
                start_expected = expected_location["start"]
                end_expected = expected_location["end"]
                color = expected_location["color"]
                label = expected_location["label"]

                ax2.axvspan(
                    xmin=start_expected,
                    xmax=end_expected,
                    ymin=0,
                    ymax=1,
                    alpha=0.3,
                    color=color,
                )

                ax2.text(
                    x=start_expected + timedelta(minutes=15),
                    y=0.95,
                    s=f"Bmapped {label}",
                )

        ax2.set_ylim(corr_thr_list[0], 1.01)
        ax2.set_ylabel("Highest corr.found")
        ax2.grid(True)
        ax2.set_xlabel("Date (dd HH:MM)")

        # Save or show the plot
        if save_folder:
            plt.savefig(f"{save_folder}{self.name}_{other.name}.png")
            print(f"Saved summary plot to {save_folder}")

        else:
            plt.show()

        plt.close()

    def generate_windows_old(
        self,
        other,
        window_displacement,
        plot_long_imfs=False,
        long_window_imf_list=[],
        use_real_time=False,
        period_filter=False,
        filter_low_high=(0, 0),
    ):
        """
        Generate array of relevant windows with two timeseries of different length
        """
        # Note: The difference between self.data and self.s is that S may be normalised
        assert (
            self.t[-1] < other.t[-1]
        ), f"Other Signal Object {other.t[-1]} is shorter than {self.t[-1]}. Please alter order of application."

        # Once asserted that the first signal is smaller, continue
        short = self
        long = other

        self.filter_low_high = filter_low_high
        self.path_to_signal = f"{short.save_folder}Split_signal_all_IMFs.npy"
        self.path_to_corr_matrix = f"{short.save_folder}IMF/Corr_matrix_all.npy"
        self.window_displacement = window_displacement

        print(f"Saving IMF files to {short.save_folder}")

        short_imfs = self.emd_and_save(
            s=short.s,
            t=short.t,
            save_folder=f"{short.save_folder}IMF/",
            save_name=f"short_{short.t[0]:04d}_{short.t[-1]:04d}",
            plot=False,
        )
        self.imfs = short_imfs

        if period_filter:
            valid_imfs_short = self.check_imf_periods(
                imfs=short_imfs,
                t=short.t,
                pmin=self.pmin,
                pmax=self.pmax,
                period_filter=True,
            )
        elif not period_filter:
            valid_imfs_short = self.check_imf_periods(
                imfs=short_imfs,
                t=short.t,
                filter_low_high=filter_low_high,
                period_filter=False,
            )

        # Setup to perform many EMDs on long dataset
        number_displacements = int(
            np.floor((long.t[-1] - short.t[-1]) / self.window_displacement)
        )  # In seconds

        # If the correlation matrix is set, skip
        try:
            np.load(self.path_to_corr_matrix)
            # When it loads correctly, it either continues or returns None
            if plot_long_imfs:
                pass
            # Found files, did not want to plot long imfs
            else:
                return None
        except FileNotFoundError:
            pass

        # Otherwise continue
        count = 0
        # Assume that two datasets are well standardised
        complete_array = np.ndarray((3, number_displacements, len(short.s)))

        if not use_real_time:
            # generate 10 by 10 matrix for IMF correlation
            corr_matrix = np.zeros(shape=(10, 10, number_displacements, 3))

        else:
            # When needed to save mid_point_time
            corr_matrix = np.zeros(shape=(10, 10, number_displacements, 4))

        # Bounds to move window
        left_bound = short.t[0]
        right_bound = short.t[-1]

        # While inside the long timeseries
        while count < number_displacements:
            # Only do if necessary!
            if (count in long_window_imf_list and plot_long_imfs) or (
                not plot_long_imfs
            ):
                # Find and set relevant window
                i = int(np.where(long.t == left_bound)[0])
                j = int(np.where(long.t == right_bound)[0])

                _data_long = long.s[i : j + 1]
                _data_long = _data_long.reshape(len(_data_long))
                _time_long = long.t[i : j + 1]

                # Set values for array
                complete_array[0, count, :] = _time_long
                complete_array[1, count, :] = _data_long
                complete_array[2, count, :] = short.s

                # Derive EMD and save to relevant folder
                _long_imfs = self.emd_and_save(
                    s=_data_long,
                    t=_time_long,
                    save_folder=f"{long.save_folder}IMF/",
                    save_name=f"long_{_time_long[0]:04d}_{_time_long[-1]:04d}",
                    plot=False,
                )

                # Uses pmin and pmax from short dataseries
                if period_filter:
                    _valid_imfs_long = self.check_imf_periods(
                        imfs=_long_imfs,
                        t=_time_long,
                        pmin=self.pmin,
                        pmax=self.pmax,
                        period_filter=True,
                    )

                else:
                    _valid_imfs_long = self.check_imf_periods(
                        imfs=_long_imfs,
                        t=_time_long,
                        filter_low_high=filter_low_high,
                        period_filter=False,
                    )

                # For all of the short, long IMFs
                for i, row in enumerate(short_imfs):
                    short_valid = valid_imfs_short[i, 0]
                    for j, col in enumerate(_long_imfs):
                        long_valid = _valid_imfs_long[j, 0]
                        if short_valid and long_valid:
                            valid = 1
                        else:
                            valid = 0

                        if count == 0:
                            # TODO: Add test to see whether normalised or not
                            pass

                        # Contains pearsonR values, SpearmanR values, and validity
                        corr_matrix[i, j, count, 0] = pearsonr(row, col)[0]
                        corr_matrix[i, j, count, 1] = spearmanr(row, col)[0]
                        corr_matrix[i, j, count, 2] = valid

                if use_real_time:  # We only have the real time in some ocasions
                    mid_point_time = np.floor((_time_long[-1] + _time_long[0]) / 2)
                    corr_matrix[0, 0, count, 3] = mid_point_time

            # Increase count by one before advancing
            count += 1
            left_bound, right_bound = (
                left_bound + self.window_displacement,
                right_bound + self.window_displacement,
            )

        np.save(self.path_to_signal, complete_array)  # Signal + IMFs
        np.save(self.path_to_corr_matrix, corr_matrix)  # IMF corrs
        return None

    def plot_all_results_old(
        self,
        other,
        Label_long_ts="No name",
        use_real_time=False,
        expected_location_list=False,
        save_path=None,
        plot_heatmaps=False,
        corr_thr_trigger=0.7,
        corr_thr_list=np.arange(0.7, 1.01, 0.1),
        margin_hours=0.5,
        bar_width=1.2,
        period_filter=True,
    ):
        """
        This function plots the number of IMFs with high correlation for all heights
        Takes signal objects.

        Parameters:
        self: EMDFuncionality object
        other: EMDFuncionality object

        """
        try:
            corr_matrix = np.load(self.path_to_corr_matrix)

        except FileNotFoundError:
            raise InterruptedError(
                "Please use generate windows before plotting results"
            )

        assert self.pmin == other.pmin and self.pmax == other.pmax, (
            f"Unequal periodicity filtering of"
            f"{self.pmin}:{self.pmax} to {other.pmin}:{other.pmax}"
        )

        assert len(self.s) < len(other.s), (
            "Data of other timeseries is smaller than given timeseries."
            "Please swap their positions"
        )

        makedirs(save_path, exist_ok=True)

        # Init the pearson, spearman and approximate location lists
        # pearsonr_array, spearmanr_array, approximate_locations = [], [], []
        corr_locations = np.ndarray(
            (len(corr_matrix[0, 0, :, 0]), len(corr_thr_list), 3)
        )

        # Derive the relevant correlation matrix for valid values
        # In the case where there is a signal peak in "Other object"
        if other.signalObject.location_signal_peak != []:
            """
            When there is a location of the signal peak,
            check start and end time, and whether
            valid pearson / spearman correlations were found
            """
            # Arrays that contains positive / negative pairs
            pe_sp_pairs = np.ndarray(
                (len(corr_matrix[0, 0, :, 0]), len(corr_thr_list), 2)
            )

        for height in range(len(corr_matrix[0, 0, :, 0])):
            # Get all pearson, spearman, and valid values
            pearson = corr_matrix[:, :, height, 0]
            spearman = corr_matrix[:, :, height, 1]
            valid = corr_matrix[:, :, height, 2]

            if use_real_time:
                midpoint = corr_matrix[0, 0, height, 3]
                midpoint_time = other.true_time[0] + timedelta(seconds=midpoint)
                time = midpoint_time

            else:
                # Only when using fake time we have location of the peak
                midpoint = self.t[int(len(self.t) / 2)]
                time = other.t[0] + midpoint + (self.window_displacement * height)

                # NUMPY INT64
                if other.signalObject.location_signal_peak:
                    start_inferred = time - midpoint
                    end_inferred = time + midpoint

                    for loc_peak in other.signalObject.location_signal_peak:
                        if start_inferred <= loc_peak <= end_inferred:
                            location = "inside"
                            break
                    else:
                        location = "outside"

            # Get rid of borders
            pearson[pearson == 0] = np.nan
            spearman[spearman == 0] = np.nan
            mask = ~np.isnan(pearson)

            # Maximum number of IMFs for each
            wind_max = 10 - np.isnan(pearson[:, 0]).sum()
            aia_max = 10 - np.isnan(pearson[0, :]).sum()

            wind_max_sp = 10 - np.isnan(spearman[:, 0]).sum()
            aia_max_sp = 10 - np.isnan(spearman[0, :]).sum()
            # Filter to only take values where considered valid due to period filtering
            pvalid = pearson[valid == 1]
            rvalid = spearman[valid == 1]

            # For all relevant correlation thresholds, depending on list
            for index, corr_thr in enumerate(corr_thr_list):
                _number_high_pe = len(pvalid[np.abs(pvalid) >= corr_thr])
                corr_locations[height, index, 1] = _number_high_pe

                _number_high_sp = len(rvalid[np.abs(rvalid) >= corr_thr])
                corr_locations[height, index, 2] = _number_high_sp

                if corr_thr == corr_thr_trigger:
                    relevant_pearson_index = index

                # Hit rate
                if other.signalObject.location_signal_peak:
                    if location == "inside":
                        pe_sp_pairs[height, index, 0] = _number_high_pe
                        pe_sp_pairs[height, index, 1] = _number_high_sp

                    else:
                        pe_sp_pairs[height, index, 0] = -_number_high_pe
                        pe_sp_pairs[height, index, 1] = -_number_high_sp

            # Only generate heatmap when above threshold
            if (
                plot_heatmaps
                and corr_locations[height, relevant_pearson_index, 1] >= 1
                and 4600 < height < 6000
            ):
                makedirs(f"{self.save_folder}corr_matrix/", exist_ok=True)

                # Reduce the arrays to get rid of the noise
                pearson_masked = pearson[mask].reshape(wind_max, aia_max)
                valid_masked = valid[mask].reshape(wind_max, aia_max)

                # Prepare heatmap
                row_labels, col_labels = [], []

                if not period_filter:
                    # In this case, need to cut by specific amount of data
                    # print(f"Filtering with {self.filter_low_high}")
                    low = self.filter_low_high[0]
                    high = -self.filter_low_high[1]

                    pearson_hmap = pearson_masked[0:-1, 0:-1]  # Get rid of edges
                    valid_hmap = valid_masked[0:-1, 0:-1]

                    # for row in range(pearson_hmap.shape[0]):
                    #     row_labels.append(
                    #         f"AIA IMF {self.filter_low_high[0] + (row + 1)}/ {pearson_hmap.shape[0] + low - high}"
                    #     )

                    # for col in range(pearson_hmap.shape[1]):
                    #     col_labels.append(
                    #         f"WIND IMF {self.filter_low_high[0] + (col+1)}/ {pearson_hmap.shape[1] + low - high}"
                    #     )

                    for row in range(pearson_hmap.shape[0]):
                        row_labels.append(
                            f"AIA IMF {0 + (row + 1)}/ {pearson_hmap.shape[0]}"
                        )

                    for col in range(pearson_hmap.shape[1]):
                        col_labels.append(
                            f"WIND IMF {(col) + 1}/ {pearson_hmap.shape[1]}"
                        )
                else:
                    low = 0
                    high = -1
                    # When filtering depending on periods, the number of ignored imfs is different per step. Is this a problem?

                    pearson_hmap = pearson_masked[
                        low:high, low:high
                    ]  # Eliminate residual
                    valid_hmap = valid_masked[low:high, low:high]

                    for row in range(pearson_hmap.shape[0]):
                        row_labels.append(
                            f"AIA IMF {low + (row + 1)}/ {pearson_hmap.shape[0]}"
                        )

                    for col in range(pearson_hmap.shape[1]):
                        col_labels.append(
                            f"WIND IMF {(col) + 1}/ {pearson_hmap.shape[1]}"
                        )

                # Plot Heatmap
                plt.figure(figsize=(12, 12))
                # Using valid_hmap and  pearson_hmap, attempt to plot just the valid numbers
                im, _ = Heatmap.heatmap(
                    pearson_hmap,
                    row_labels,
                    col_labels,
                    valid_data=valid_hmap if (period_filter == True) else valid_hmap,
                    vmin_vmax=[-1, +1],
                    cmap="RdBu",
                    cbarlabel=f"PearsonR correlation",
                )
                _ = Heatmap.annotate_heatmap(
                    im, valfmt="{x:.2f}", textcolors=["black", "black"]
                )

                # Title information is contained in filename instead
                # plt.title(f"Correlation matrix at window #{height} {time}")

                if self.signalObject.location_signal_peak is not False:
                    fig_locn = f"{self.save_folder}corr_matrix/IMF_Heatmap{height:03d}_{time}_peak_at_{self.signalObject.location_signal_peak}.{self.saveformat}"

                else:
                    fig_locn = f"{self.save_folder}corr_matrix/IMF_Heatmap{height:03d}_{time}.{self.saveformat}"

                plt.tight_layout(pad=0.001)
                plt.savefig(fig_locn, bbox_inches="tight", dpi=300)
                print(f"Saved heatmap to {fig_locn}")
                # plt.show()
                plt.close()

                def create_ts_plot(a=self.s, b=other.s, start=height):
                    c = b[start : start + len(a)]
                    t = other.t[start : start + len(a)] / 60

                    plt.figure(figsize=(16, 12))
                    plt.plot(t, a, color="black", label=r"AIA$_{synth}$")
                    plt.plot(t, c, color="blue", label=r"WIND$_{synth}$")
                    plt.ylabel("Normalised value")
                    plt.xlabel("Minutes since start")
                    plt.legend()
                    plt.title(f"Direct Pearson R correlation: {pearsonr(a, c)[0]:.2f}")
                    save_to = f"{self.save_folder}corr_matrix/IMF_Heatmap{height:03d}_{time}_plot.{self.saveformat}"
                    plt.savefig(save_to, bbox_inches="tight", dpi=300)
                    # plt.show()
                    plt.close()

                create_ts_plot()

        # Save the hitrates
        if other.signalObject.location_signal_peak:
            df_pe = pd.DataFrame({})
            df_sp = pd.DataFrame({})

            for index, corr_thr in enumerate(corr_thr_list):
                df_pe[f"{corr_thr}"] = pe_sp_pairs[:, index, 0]
                df_sp[f"{corr_thr}"] = pe_sp_pairs[:, index, 1]

            self.hitrate = (df_pe, df_sp)  # Then have the hitrate information available
            hitrate_tables = self.calculate_hitrate(save=True)

        ### Establish relevant signal Objects
        # SHORT
        short_signal = self
        short_values = short_signal.s
        short_time = short_signal.t
        # n_short_imfs = len(short_signal.imfs)

        # LONG
        long_signal = other
        # time = long_signal.t
        long_values = long_signal.s

        # window_width = max(short_time) * 12 / 60

        if use_real_time:
            short_duration = short_signal.true_time[-1] - short_signal.true_time[0]
            time_axis = long_signal.true_time
        else:
            short_duration = (short_signal.t[-1] - short_signal.t[0]) / 60
            time_axis = long_signal.t / 60

        region_string = self.name
        #######################################
        ### Figure
        fig, axs = plt.subplots(2, sharex=True, figsize=(20, 10))

        # First plot
        ax = axs[0]
        ax.plot(time_axis, long_values, color="black", label=Label_long_ts, alpha=1)
        # TODO: Plot detrended dataset

        if period_filter:
            ax.set_title(f"{self.pmin} < {r'$P_{IMF}$'} < {self.pmax} min")

        elif not period_filter:
            ax.set_title(f"Noise {self.signalObject.noise_perc}%")
        else:
            raise ValueError("Please determine whether using period or not")

        ax.set_ylabel(f"Normalised {Label_long_ts}")
        if use_real_time:
            ax.set_xlim(
                time_axis[0] - timedelta(hours=margin_hours),
                time_axis[-1] + timedelta(hours=margin_hours),
            )
            ax.xaxis.grid(True)

            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
            ax.xaxis.set_tick_params(rotation=45)

        else:
            # Set xticks every hour
            xticks = np.arange(time_axis[0], time_axis[-1] + 1, step=180)
            plt.xticks(xticks)

        #######################################
        # INSET AXIS
        # displays short signal

        # if Label_long_ts == 'Mf': --> Just /inset and change to if True
        if False:
            in_ax = inset_axes(
                ax,
                width="15%",  # width = 30% of parent_bbox
                height=1,  # height : 1 inch
                loc=1,
            )
            in_ax.plot(short_time, short_values, color="black")
            in_ax.set_xlabel(f"{region_string}")
            plt.xticks([])
            plt.yticks([])

        # DONE: Change time_mid_min to be representative of real time!
        if use_real_time:
            # Plot the hits
            try:
                true_time_secs = corr_matrix[0, 0, :, 3]  # Saved only sometimes!
            except IndexError as ind_err:
                raise IndexError("Please ensure that the true timeseries is saved!")

            ttindex = (true_time_secs / long_signal.cadence).astype(int)
            true_time_datetime = long_signal.true_time[ttindex]
            time_mid_min = true_time_datetime
            bar_width = short_duration

        else:
            time_mid_min = corr_locations[:, 0, 0]
            bar_width = short_duration

        ###################################################
        # BOTTOM PLOT
        ax2 = axs[1]

        # Adding specific colormap for pairs
        possibleColors = {"1": "green", "2": "blue", "3": "red"}

        for height in range(len(time_mid_min)):
            # Open up pearson and spearman IMF pair list
            pearson_array = corr_locations[height, :, 1]
            spearman_array = corr_locations[height, :, 2]

            if use_real_time:
                midpoint = corr_matrix[0, 0, height, 3]
                midpoint_time = other.true_time[0] + timedelta(seconds=midpoint)
                time = midpoint_time
                barchart_time = time

            else:
                midpoint = self.t[int(len(self.t) / 2)]
                time = other.t[0] + midpoint + (self.window_displacement * height)
                barchart_time = time / 60
                # NUMPY INT64

            # Bar charts for each of the heights
            for index, corr_label in enumerate(corr_thr_list):
                if pearson_array[index] != 0:  # If some pairs are found
                    try:
                        _color = possibleColors[f"{int(pearson_array[index])}"]
                    except KeyError:
                        _color = "red"

                    _alpha = 0.35 if pearson_array[index] > 0 else 0
                    ax2.bar(
                        barchart_time,
                        corr_label,
                        width=bar_width,
                        color=_color,
                        edgecolor="white",
                        alpha=_alpha,
                        zorder=1,
                    )

                # if spearman_array[index] != 0:
                #     _alpha = 0.9 if spearman_array[index] > 0 else 0
                #     ax2.bar(
                #         barchart_time,
                #         corr_label,
                #         width=bar_width,
                #         color="black",
                #         edgecolor="white",
                #         alpha=_alpha,
                #         zorder=2,
                #     )

        # Columns on bottom plot
        if use_real_time:
            margin_label = timedelta(hours=1)
            # interval = int((other.t[-1] / 3600) / 10)
            interval = 3

            ax2.xaxis.set_minor_locator(mdates.HourLocator(1))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
            ax2.xaxis.set_tick_params(rotation=10)
            # Set the x limits
            ax2.set_xlim(
                time_axis[0] - timedelta(hours=margin_hours),
                time_axis[-1] + timedelta(hours=margin_hours),
            )
            # If using some expected location dictionary add here
            if expected_location_list is not False:
                for expected_location in expected_location_list:
                    start_expected = expected_location["start"]
                    end_expected = expected_location["end"]
                    color = expected_location["color"]
                    label = expected_location["label"]

                    ax2.axvspan(
                        xmin=start_expected,
                        xmax=end_expected,
                        ymin=0,
                        ymax=1,
                        alpha=0.3,
                        color=color,
                    )

                    ax2.text(
                        x=start_expected + timedelta(minutes=15),
                        y=0.95,
                        s=f"Bmapped {label}",
                    )

        else:
            ax2.xaxis.set_tick_params(rotation=25)
            margin_label = -10
            interval = int((other.t[-1] / 3600) / 10)

        # Set out the legend and info about pairs
        legend_x = time_axis[-1] + margin_label
        # ax2.text(x=legend_x, y=0.95, s="1 pair", color=possibleColors["1"])
        # ax2.text(x=legend_x, y=0.9, s="2 pairs", color=possibleColors["2"])
        # ax2.text(x=legend_x, y=0.85, s="3 pairs", color=possibleColors["3"])

        # Print hit-rate somewhere TODO: fix the "and False"
        if other.signalObject.location_signal_peak and False:
            for corr, _ in zip(hitrate_tables["pearson"], hitrate_tables["spearman"]):
                ax2.text(
                    x=legend_x + 10,
                    y=float(corr),
                    s=f"{float(hitrate_tables['pearson'][corr].values[0]):.0f}%",
                    fontsize=14,
                    color="black",
                )

                ax2.text(
                    x=legend_x + 80,
                    y=float(corr),
                    s=f"{float(hitrate_tables['spearman'][corr].values[0]):.0f}%",
                    fontsize=14,
                    color="black",
                )

        # Divide by 2 number of ticks and add 1 to the end
        corr_thr_new = []
        for i, value in enumerate(corr_thr_list):
            if np.mod(i, 2) == 0:
                corr_thr_new.append(value)
        corr_thr_new.append(1)

        ax2.set_yticks(corr_thr_new)  # Ensure ticks are good
        ax2.set_ylim(corr_thr_list[0], 1.01)  # Allows for any correlation
        ax2.set_ylabel("Highest corr. found")

        ax2.grid(True)

        if use_real_time:
            ax2.set_xlabel("Date (dd HH:MM)")
        else:
            ax2.set_xlabel("Minutes since start")

        # Save, show and close
        save_name = f"{long_signal.name}_{region_string}_IMF_Summary"
        plt.savefig(
            f"{self.save_folder}{save_name}.{self.saveformat}",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()

    def calculate_hitrate(self, save=False):
        """
        Creates dataframe with hitrate and prints it out
        """
        assert (
            self.hitrate is not False
        ), "Please calculate hitrates with generate_windows"

        self.table = {}
        df_pe, df_sp = self.hitrate

        for label, df_hrate in zip(("pearson", "spearman"), (df_pe, df_sp)):
            results_df = pd.DataFrame({})

            # For the different correlation thresholds
            for _, corr_thr in enumerate(df_hrate):

                # Reset hit-rate to ensure working well
                _hit_rate = 0

                # Find hitrate
                ncorrect = np.sum(df_hrate[corr_thr][df_hrate[corr_thr].gt(0)])
                # nincorrect = np.sum(df_pe[corr_thr][df_pe[corr_thr].lt(0)])
                ntotal = np.sum(abs(df_hrate[corr_thr]))

                if ntotal != 0:
                    _hit_rate = ncorrect / ntotal * 100

                results_df[f"{corr_thr}"] = [f"{_hit_rate:.2f}"]

            self.table[label] = results_df

        if save:
            # This will be one of the results dataframes
            self.table["pearson"].to_csv(f"{self.save_folder}pearson_hitrate.csv")
            self.table["spearman"].to_csv(f"{self.save_folder}spearman_hitrate.csv")

        return self.table

    def plot_single_hitrate(self, mode="pearson", show=False):
        """
        Plots hitrates
        Needs to replicate  TestIMF functionality
        Input of a dictionary with several
        """
        assert (
            self.table is not None
        ), f"Hit-rate table not created. Use {self.calculate_hitrate.__name__}"

        corr_table = self.table
        n = len(list(corr_table))  # All correlation thresholds
        x = range(n)
        # font = {"family": "DejaVu Sans", "weight": "bold", "size": 18}

        matplotlib.rc("font", **font)

        fig = plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.set_prop_cycle("color", "blue")

        for corr in list(corr_table[mode]):
            # This is plot of hit rate per correlation threshold, with the windowing as colour
            plt.plot(
                float(corr),
                float(corr_table[mode][corr][0]) + 0.1,
                label=f"{self.window_displacement}s",
                marker="o",
                markersize=14,
                color="black",
            )

        plt.grid("both")
        ax.set_ylabel("Hit rate (%)")
        ax.set_xlabel("Correlation threshold")
        ax.set_ylim(-5, 105)

        plt.title(f"{self.signalObject.noise_perc}% noise")
        fig.subplots_adjust(bottom=0.25)
        if show:
            plt.show()

        plt.savefig(
            f"{self.save_folder}{self.window_displacement}"
            f"_{self.signalObject.noise_perc}%_hitrates.{self.saveformat}",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
