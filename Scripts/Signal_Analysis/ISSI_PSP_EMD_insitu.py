import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
import idlsave
import sys

# sys.path.append("/mnt/sda5/Python/InsituEMDCorrelation/Scripts")
# from Signal_Analysis.helpers import Signal, EMDFunctionality
from helpers import Signal, EMDFunctionality

from datetime import datetime, timedelta
import PyEMD
from astropy.convolution import convolve, Box1DKernel

import ray
import psutil

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

main_folder = "/mnt/sda5/Python/InsituEMDCorrelation/Scripts/ISSI/"
data_folder = main_folder + "Data/"
save_folder = main_folder + "Results/"

# In situ observations
df_is = pd.read_csv(f"{data_folder}small_ch_in_situ.csv")
df_is.index = pd.to_datetime(df_is["Time"])
del df_is["Time"]  # Remove time column to not confuse variable names

rs_171 = idlsave.read(f"{data_folder}small_ch_171_lc_in.sav", verbose=False)
rs_193 = idlsave.read(f"{data_folder}small_ch_193_lc_in.sav", verbose=False)

time_array = rs_171.date_obs_171.copy()
time_array = [t.decode() for i, t in enumerate(time_array)]

df_171 = pd.DataFrame(
    {
        "plume": rs_171.lc_171_plume_in,
        "cbpoint": rs_171.lc_171_bp_in,
        "chplume": rs_171.lc_171_ch_plume_in,
        "chole": rs_171.lc_171_ch_in,
        "qsun": rs_171.lc_171_qs_in,
    },
    index=pd.to_datetime(time_array),
)

df_193 = pd.DataFrame(
    {
        "plume": rs_193.lc_193_plume_in,
        "cbpoint": rs_193.lc_193_bp_in,
        "chplume": rs_193.lc_193_ch_plume_in,
        "chole": rs_193.lc_193_ch_in,
        "qsun": rs_193.lc_193_qs_in,
    },
    index=pd.to_datetime(time_array),
)


def extract_smp(case_study, plot=False):
    """
    Find dt and return split timeseries
    """
    case_study["dt"] = (
        case_study["end"] - case_study["start"]
    ).total_seconds() / 3600  # Hours

    case_study["data"] = case_study["data"][case_study["start"] : case_study["end"]]

    if "bmapped_start" in case_study.keys():
        case_study["bmapped_dt"] = (
            case_study["bmapped_end"] - case_study["bmapped_start"]
        ).total_seconds() / 3600  # Hours

    case_study["smp_data"] = case_study["data"].copy()

    if plot:
        plt.figure(figsize=(15, 15))
    for index, parameter in enumerate(list(case_study["data"])):
        signal = case_study["data"][parameter].copy()
        csignal = convolve(signal, Box1DKernel(200), boundary="extend")
        smp_signal = 100 * (signal - csignal) / csignal

        if plot:
            plt.subplot(len(list(case_study["data"])), 1, index + 1)
            plt.plot(smp_signal, color="black")
            if index < len(list(case_study["data"])) - 1:
                plt.tick_params(
                    axis="x",  # changes apply to the x-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                )  # labels along the bottom edge are off
            plt.ylabel(parameter)

        case_study["smp_data"][parameter] = smp_signal

    if plot:
        plt.show()
        plt.close()

    return case_study


def compare_rs_to_insitu(
    df_insitu,
    df_remote,
    rs_cad,
    is_cad,
    remote_label,
    corr_thr_list=[],
    pfilter=(1, 180),
    over_folder=None,
    expected_location_list=False,
):

    assert over_folder != None, "Please set over_folder to store relevant arrays"

    # For all of the lightcurves
    for rs_var in list(df_remote):
        rs_d = df_remote[rs_var]

        # Remote sensing signal gets set up first
        rs_signal = Signal(
            cadence=rs_cad,
            custom_data=rs_d,
            name=rs_var,
            time=df_remote.index,
            save_folder=None,
        )

        rs_signal.data = Signal.detrend(rs_signal.data)

        # Add functionality, filter and generate IMFs
        rs_func = EMDFunctionality(rs_signal, pfilter=pfilter)

        # For each of the in-situ variables
        for is_var in df_insitu:
            is_folder_base = f"{over_folder}"
            makedirs(is_folder_base, exist_ok=True)
            is_d = df_insitu[is_var]

            # In-situ signal setup
            is_signal = Signal(
                cadence=is_cad,
                custom_data=is_d,
                save_folder=is_folder_base,
                name=is_var,
            )
            is_signal.data = Signal.detrend(is_signal.data)
            is_func = EMDFunctionality(
                is_signal,
                pfilter=pfilter,
            )

            # First long, then short. This allows for easier use of plot_all
            rs_func.generate_windows(is_func)
            rs_func.plot_all_results(
                other=is_func,
                corr_thr_list=corr_thr_list,
                expected_location_list=expected_location_list,
                save_folder=is_folder_base,
            )


# Setup for in situ observations  TODO: Make into a class maybe(...)
case_study_1 = {
    "start": datetime(2018, 10, 31, 12, 0),
    "end": datetime(2018, 10, 31, 22, 0),
    # "bmapped_start": datetime(2018, 10, 30, 2, 0),
    # "bmapped_end": datetime(2018, 10, 30, 19, 0),
    "data": df_is,
    "df_rs": df_171,
    "remote_label": "171",
    "over_folder_subfolder": "171_sb",
    "corr_thr_list": np.round(np.arange(0.6, 1, 0.05), 2),
    "period_limits": (5, 200),
    "expected_location_list": [],
}

case_study_1 = extract_smp(case_study_1)
acc = {
    "start": datetime(2018, 10, 30, 2),
    "end": datetime(2018, 10, 30, 19),
    "label": "Acc. Vsw",
    "color": "blue",
    "height": 1,
}

con = {
    "start": datetime(2018, 10, 30, 11),
    "end": datetime(2018, 10, 31, 2),
    "label": "Con. Vsw",
    "color": "yellow",
    "height": 1,
}

case_study_1["expected_location_list"].append(acc)
case_study_1["expected_location_list"].append(con)

# plot_all(case_study_1)
compare_rs_to_insitu(
    df_insitu=case_study_1["smp_data"],
    df_remote=case_study_1["df_rs"],
    rs_cad=60,
    is_cad=60,
    remote_label=case_study_1["remote_label"],
    corr_thr_list=case_study_1["corr_thr_list"],
    pfilter=case_study_1["period_limits"],
    over_folder=f"{save_folder}{case_study_1['over_folder_subfolder']}/",
    expected_location_list=case_study_1["expected_location_list"],
)
