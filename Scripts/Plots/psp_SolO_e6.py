from astropy.table import QTable
from astropy.coordinates import spherical_to_cartesian
import astropy.constants as const
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
from os import makedirs
from sys import path
BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"

path.append(f"{BASE_PATH}Scripts/")

from Imports.Spacecraft import PSPSolO_e6


def psp_e6():
    """
    Additional encounter possibly currently. Check data
    """
    OBJ_CADENCE = 60  # To one minute resolution
    PLOT_ORBITS = False
    SHOW_PLOTS = True
    stepMinutes = 60

    psp_e6_overview = {
        "name": "PSPpriv_e6",
        "cadence_obj": OBJ_CADENCE,
    }

    solo_e6_overview = {
        "name": "SolOpriv_e6",
        "cadence_obj": OBJ_CADENCE,
        "show": SHOW_PLOTS,
    }

    # Prepare the objects with measurements inside DF
    psp = PSPSolO_e6(**psp_e6_overview)
    solo = PSPSolO_e6(**solo_e6_overview)

    # Solar orbiter lines up with latitude?
    # In september
    solo_zoomed = {
        "start_time": datetime(2020, 9, 27, 11, 30),
        "end_time": datetime(2020, 10, 4, 11, 53),
        "stepMinutes": stepMinutes,
        "color": "black",
    }

    psp_zoomed = {
        "start_time": datetime(2020, 9, 25),
        "end_time": datetime(2020, 9, 30),
        "stepMinutes": stepMinutes,
        "color": "red",
    }

    solo_paper = {
        "start_time": datetime(2020, 10, 1, 22, 40),
        "end_time": datetime(2020, 10, 2, 0, 13),
        "color": "black",
    }

    psp_paper = {
        "start_time": datetime(2020, 9, 27, 3, 30),
        "end_time": datetime(2020, 9, 27, 5),
        "color": "red",
    }

    # Here we save the scaled DF
    solo.plot_solo_psp_df(
        psp, zones=[solo_paper, psp_paper], saveScaledDF=True, case="orbit6"
    )
    solo.zoom_in(**solo_zoomed)
    psp.zoom_in(**psp_zoomed)

    # Resample to an objective cadence
    psp.df = psp.df.resample(f"{OBJ_CADENCE}s").mean()
    solo.df = solo.df.resample(f"{OBJ_CADENCE}s").mean()

    # Remove all blanks
    solo.df.fillna(method="pad")
    psp.df.fillna(method="pad")

    orbit_case_path = f"{BASE_PATH}Figures/Orbit_3d/"
    makedirs(orbit_case_path, exist_ok=True)

    # Spacecraft object calling the function is where the solar wind is being mapped from
    if PLOT_ORBITS:

        # Create a set of radial separations
        for minSteps in [
            stepMinutes,
            stepMinutes * 2,
            stepMinutes * 4,
            stepMinutes * 6,
            stepMinutes * 12,
            stepMinutes * 24,
        ]:
            tol = 1.5
            solo.plotOrbit_x_y(
                psp,
                objFolder=f"{orbit_case_path}/",
                plotRate=f"{minSteps}min",
                farTime="22:00",
                closeTime="04:00",
                pspiralHlight=datetime(
                    2020,
                    9,
                    28,
                    23,
                ),
                radialTolerance=tol,
            )


def psp_sda_2019():
    pass

# %%


if __name__ == "__main__":
    # Do twice so it saves with Radius hopefully
    try:
        psp_e6()
    except AttributeError or ValueError:
        psp_e6()
