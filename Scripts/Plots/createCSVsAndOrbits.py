from astropy.table import QTable
from astropy.coordinates import spherical_to_cartesian
from datetime import datetime
from os import makedirs
from sys import path

BASE_PATH = "/Users/ddp/Documents/PhD/inEMD_Github/"

path.append(f"{BASE_PATH}Scripts/")

from Imports.Spacecraft import PSPSolO_e6, EarthApril2020


def psp_e6(show=False, plot_orbit=False, radialTolerance=1.5):
    """
    Additional encounter possibly currently. Check data
    """
    OBJ_CADENCE = 60  # To one minute resolution
    PLOT_ORBITS = plot_orbit
    SHOW_PLOTS = show
    orbitStepMin = 90

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
        "stepMinutes": orbitStepMin,
    }

    psp_zoomed = {
        "start_time": datetime(2020, 9, 25),
        "end_time": datetime(2020, 9, 30),
        "stepMinutes": orbitStepMin,
    }

    # These are zones and therefore have colours
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
    solo.plot_solo_psp_df_onlyScaled(
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

    orbit_case_path = f"{BASE_PATH}Figures/PSP_SolO/"
    makedirs(orbit_case_path, exist_ok=True)

    # Spacecraft object calling the function is where the solar wind is being mapped from
    if PLOT_ORBITS:

        # Create a set of radial separations
        for minSteps in [
            orbitStepMin,
            orbitStepMin * 3,
            orbitStepMin * 4,
            orbitStepMin * 6,
            orbitStepMin * 12,
            orbitStepMin * 24,
        ]:
            # Highlight one of the parker spirals
            solo.plotOrbit_x_y(
                psp,
                objFolder=f"{orbit_case_path}",
                plotRate=f"{minSteps}min",
                farTime="22:00",
                closeTime="04:00",
                # pspiralHlight=datetime(
                #     2020,
                #     9,
                #     28,
                #     23,
                # ),
                radialTolerance=radialTolerance,
                selfName="SolO",
                otherName="PSP",
                legendLoc="best",
                plot_spirals=False,
            )


def solo_Earth_April_2020(show=False, plot_orbit=False):
    orbitStepMin = 90
    radialTolerance = 4.4
    solo = EarthApril2020(
        name="SolO_April_2020", cadence_obj=92, show=show, remakeCSV=False
    )
    earth = EarthApril2020(
        name="Earth_April_2020", cadence_obj=92, show=show, remakeCSV=False
    )
    solo.extract_orbit_data(from_data=True, stepMinutes=60)
    earth.extract_orbit_data(from_data=True, stepMinutes=60)

    earth.plot_solo_earth_df(solo)

    # Remove all blanks
    solo.df.fillna(method="pad")
    earth.df.fillna(method="pad")

    orbit_case_path = f"{BASE_PATH}Figures/SolO_Earth/"
    makedirs(orbit_case_path, exist_ok=True)

    # Spacecraft object calling the function is where the solar wind is being mapped from
    if plot_orbit:

        # Create a set of radial separations
        for minSteps in [
            orbitStepMin,
            orbitStepMin * 3,
            orbitStepMin * 4,
            orbitStepMin * 6,
            orbitStepMin * 12,
            orbitStepMin * 24,
        ]:
            # Highlight one of the parker spirals
            earth.plotOrbit_x_y(
                solo,
                objFolder=f"{orbit_case_path}",
                plotRate=f"{minSteps}min",
                # farTime="22:00",
                # closeTime="04:00",
                pspiralHlight=None,
                radialTolerance=radialTolerance,
                plot_spirals=False,
                zoomRegion=((-0.95, -0.65), (-0.6, -0.2)),
                vSW=int(earth.df["V_R"].mean()),
                selfName="WIND",
                otherName="SolO",
                selfSpiralRadii=(0.8, 1),
                otherSpiralRadii=(0.75, 1),
                legendLoc="upper left",
            )


def sta_psp(show=False, plot_orbit=False):
    from Imports.Spacecraft import STA_psp
    import astropy.units as u

    cadence_obj = 60
    orbitStepMin = 300
    radialTolerance = 0.1

    sta = STA_psp(name="STA_Nov_2019", cadence_obj=cadence_obj, show=show)
    sta.dfUnits["R"] = sta.dfUnits["R"].value * u.AU
    psp = STA_psp(name="PSP_Nov_2019", cadence_obj=cadence_obj, show=show)

    sta.plot_sta_psp_df(psp)
    # Remove all blanks
    sta.df.fillna(method="pad")
    psp.df.fillna(method="pad")

    sta.extract_orbit_data(from_data=True, stepMinutes=60)
    psp.extract_orbit_data(from_data=True, stepMinutes=60)

    orbit_case_path = f"{BASE_PATH}Figures/PSP_STA/"
    makedirs(orbit_case_path, exist_ok=True)

    # Spacecraft object calling the function is where the solar wind is being mapped from
    if plot_orbit:

        # Create a set of radial separations
        for minSteps in [
            orbitStepMin,
            orbitStepMin * 3,
            orbitStepMin * 4,
            orbitStepMin * 6,
            orbitStepMin * 12,
            orbitStepMin * 24,
        ]:
            # Highlight one of the parker spirals
            sta.plotOrbit_x_y(
                psp,
                objFolder=f"{orbit_case_path}",
                plotRate=f"{minSteps}min",
                # farTime="22:00",
                # closeTime="04:00",
                pspiralHlight=None,
                radialTolerance=radialTolerance,
                zoomRegion=((0.65, 0.95), (-0.75, -0.2)),
                vSW=int(sta.df["V_R"].mean()),
                selfName="STEREO-A",
                otherName="PSP",
                selfSpiralRadii=(0.7, 0.8),
                otherSpiralRadii=(0.8, 0.9),
                legendLoc="upper left",
                plot_spirals=False,
            )


if __name__ == "__main__":
    # Do twice so it saves with Radius hopefully
    show = False
    sta_psp(show=show, plot_orbit=True)
    try:
        psp_e6(show=show, plot_orbit=True)
        solo_Earth_April_2020(show=show, plot_orbit=True)

    except AttributeError or ValueError:
        psp_e6(show=show)
        solo_Earth_April_2020(show=show)
