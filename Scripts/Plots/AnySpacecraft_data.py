BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"

from sys import path
from collections import namedtuple

path.append(f"{BASE_PATH}Scripts/")

# from helpers import resample_and_rename
from os import makedirs
from physicsHelpers import fcl
from signalHelpers import normalize_signal as norm
from Plots.cdfReader import extractDF
from heliopy.models import ParkerSpiral

from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import spherical_to_cartesian

# To add a cycle to colours
from itertools import cycle

locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)

# Units
units_dic = {
    "density": r"#/$cm^{3}$",
    "T": r"$K$",
    "V_R": r"$km/s$",
    "V_T": r"$km/s$",
    "V_N": r"$km/s$",
    "btotal": r"$T$",
}

SOLROTRATE = 24.47 / 86400


def lineCalc(P1, P2):
    """
    Calculate line given 2 points (X1, Y1), (X2, Y2)
    """
    m = (P2[1] - P1[1]) / (P2[0] - P1[0])
    b = P1[1] + m * P1[0]
    return m, b


# Find p spiral alignment
def findSpirals(df, vSW, rads=(0, 0.9)):
    _spirals = []
    Spiral = namedtuple("Spiral", ["X", "Y", "Z"])
    for t, measurements in df.iterrows():
        r0 = measurements["rad"]
        l0 = measurements["lon"]
        spiral = ParkerSpiral(vSW * u.km / u.s, r0 * u.au, l0 * u.deg)

        if len(rads) == 2:
            rads = np.linspace(*rads)
        longs = spiral.longitude(rads * u.au)
        cartSpiral = spherical_to_cartesian(rads, np.repeat(0, len(longs)), longs)
        _spirals.append(Spiral(*cartSpiral))

    return _spirals


class Spacecraft:
    """
    The intention of this class is to enable similar download and
    storage of WIND, STEREO A, PSP and SolO observations.

    """

    def __init__(
        self,
        name="NONE",
        mid_time=None,
        margin=timedelta(days=1),
        cadence_obj=None,
        show=False,
    ):
        """
        :param name: Spacecraft name from one of [WIND, PSP, ST-A]
        :param mid_time: Time at centre of timeseries
        :param margin: Timemargin (Timedelta object) at either side of mid_time
        :param cadence_obj: Objective cadence
        """
        self.name = name
        self.obj_cad = cadence_obj
        self.show = show
        self.completeCSVPath = f"{BASE_PATH}Data/Prepped_csv/{self.name}.csv"

        try:
            self.df = pd.read_csv(self.completeCSVPath)
            self.df.index = pd.to_datetime(self.df["Time"])

            if "R" not in self.df.columns:
                self.saveWithRadius()

            del self.df["Time"]
            return None

        except FileNotFoundError:
            pass

        from glob import glob

        self.sp_coords_carrington = None  # Set a param which is empty to then fill

        # Time information for specific spacecraft
        if mid_time:
            self.mid_time = mid_time
            self.margin = margin
            self.start_time = mid_time - self.margin
            self.end_time = mid_time + self.margin
        else:
            raise ValueError("Please provid a valid mid_time and time margin")

        # Get the spacecraft data into the proper format

        if self.name == "WIND":
            from heliopy.data import wind

            df_mag = wind.mfi_h0(self.start_time, self.end_time)
            df_swe = wind.swe_h1(self.start_time, self.end_time)
            self.cadence = None
            # Output of self.df.columns
            """
            """
            self.df = pd.append(df_mag, df_swe)

        elif self.name == "PSPpub":
            from heliopy.data import psp as psp_data

            self.rel_vars = {
                "btotal": "btotal",
            }

            df_mag_data = psp_data.fields_mag_rtn_1min(self.start_time, self.end_time)
            # sweap_l3 = psp_data.sweap_spc_l2(self.start_time, self.end_time)

            # MAG
            Bx = df_mag_data.to_dataframe()["psp_fld_l2_mag_RTN_1min_0"]
            By = df_mag_data.to_dataframe()["psp_fld_l2_mag_RTN_1min_1"]
            Bz = df_mag_data.to_dataframe()["psp_fld_l2_mag_RTN_1min_2"]
            Bt = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            df_mag = pd.DataFrame({"Bx": Bx, "By": By, "Bz": Bz, "Bt": Bt})

            self.df = pd.DataFrame({})
            self.df["Time"] = df_mag_data.index
            self.df["btotal"] = Bt

            # # SWEAP
            # Np = sweap_l3.to_dataframe()['np_moment']  # Number density
            # Vx = sweap_l3.to_dataframe()['vp_moment_RTN_0']
            # Vy = sweap_l3.to_dataframe()['vp_moment_RTN_1']
            # Vz = sweap_l3.to_dataframe()['vp_moment_RTN_2']
            # Vt = np.sqrt(Vx**2 + Vy ** 2 + Vz ** 2)
            # T = sweap_l3.to_dataframe()['wp_moment']*u.K  # Temperature
            # Lat = sweap_l3.to_dataframe()['carr_latitude']
            # Lon = sweap_l3.to_dataframe()['carr_longitude']
            # P = (1/2) * Np * (Vx**2) * 100000
            # Mf = Np * Vx

        elif self.name == "SolOpriv_e6":

            df_mag_csv_path = f"{BASE_PATH}Data/Prepped_csv/solo_mag_sept.csv"
            df_rpw_csv_path = f"{BASE_PATH}Data/Prepped_csv/solo_rpw_sept.csv"
            df_sweap_csv_path = f"{BASE_PATH}Data/Prepped_csv/solo_sweap_sept.csv"

            def RPW_prep():
                cdf_path = f"{BASE_PATH}unsafe/Resources/Solo_Data/L3/RPW/"

                _rpw_df = extractDF(
                    cdf_path, vars=["DENSITY"], info=False, resample=f"{self.obj_cad}s"
                )

                _rpw_df[_rpw_df["DENSITY"] < 1] = np.nan
                _rpw_df["N_RPW"] = _rpw_df["DENSITY"]
                del _rpw_df["DENSITY"]

                _rpw_df["Time"] = _rpw_df.index
                _rpw_df.to_csv(df_rpw_csv_path, index=False)
                del _rpw_df["Time"]

                return _rpw_df

            def SWA_PAS():
                # SWA-PAS data
                """Reduced list
                'Epoch',
                'validity',
                'N',
                'V_RTN',
                'T'
                """
                cdf_path = f"{BASE_PATH}unsafe/Resources/Solo_Data/L2/GroundMom/"
                _sweap_df = extractDF(
                    cdf_path,
                    vars=["V_RTN", "N", "T", "validity"],
                    info=False,
                    resample=f"{self.obj_cad}s",
                )
                _sweap_df[_sweap_df["validity"] < 3] = np.nan
                del _sweap_df["validity"]

                # Save and delete time index
                _sweap_df["Time"] = _sweap_df.index
                _sweap_df.to_csv(df_sweap_csv_path, index=False)
                del _sweap_df["Time"]

                return _sweap_df

            def MAG_prep():
                """
                Epoch
                B_RTN
                QUALITY_FLAG
                """
                cdf_path = f"{BASE_PATH}unsafe/Resources/Solo_Data/L2/Mag/"

                # Need to separate into different relevant csv
                _mag_df = extractDF(
                    cdf_path, vars=["B_RTN", "QUALITY_FLAG"], info=False
                )
                _mag_df["mag_flag"] = _mag_df["QUALITY_FLAG"]
                del _mag_df["QUALITY_FLAG"]

                _mag_df["Time"] = _mag_df.index
                _mag_df.to_csv(df_mag_csv_path, index=False)
                return _mag_df

            _sweapdf = SWA_PAS()
            _rpwdf = RPW_prep()
            self.plasmadf = pd.concat([_sweapdf, _rpwdf], join="outer", axis=1).fillna(
                np.nan
            )
            self.magdf = MAG_prep()

            self.df = None

        elif self.name == "PSPpriv_e6":
            """
            This is the Encounter 6 data
            """
            df_fld_csv_path = f"{BASE_PATH}Data/Prepped_csv/psp_mag_sept.csv"
            df_span_csv_path = f"{BASE_PATH}Data/Prepped_csv/psp_span_sept.csv"

            def FLD_prep():
                # When file not available, derive it
                cdf_path = f"{BASE_PATH}unsafe/Resources/PSP_Data/FIELDS/"

                _fld_df = extractDF(
                    CDFfolder=cdf_path,
                    vars=["psp_fld_l2_mag_RTN_1min", "psp_fld_l2_quality_flags"],
                    timeIndex="epoch_mag_RTN_1min",
                    info=False,
                )

                _fld_df["fld_flag"] = _fld_df["psp_fld_l2_quality_flags"]
                del _fld_df["psp_fld_l2_quality_flags"]
                _fld_df["Time"] = _fld_df.index
                _fld_df.to_csv(
                    df_fld_csv_path,
                    index=False,
                )

                return _fld_df

            def SPAN_AI_prep():
                """
                This function prepares the SWEAP data for PSP
                """
                # When file not available, derive it
                cdf_path = f"{BASE_PATH}unsafe/Resources/PSP_Data/SWEAP/SPAN-AI/"

                _span_df = extractDF(
                    CDFfolder=cdf_path,
                    vars=["DENS", "VEL", "TEMP", "QUALITY_FLAG"],
                    timeIndex="Epoch",
                    info=False,
                    resample=f"{self.obj_cad}s",
                )

                for og, n in zip(
                    ["DENS", "TEMP", "QUALITY_FLAG"], ["N", "T", "span_flag"]
                ):
                    _span_df[n] = _span_df[og].copy()
                    del _span_df[og]

                _span_df["Time"] = _span_df.index
                _span_df.to_csv(
                    df_span_csv_path,
                    index=False,
                )
                return _span_df

            self.magdf = FLD_prep()
            self.plasmadf = SPAN_AI_prep()
            self.df = None

        else:
            raise NotImplementedError(f"{self.name} not implemented")

        # When we need to combine plasmadf and magdf
        if self.df is None:

            # We need to use the Time column
            # For PSP data, for SWEAP only cleaning
            if "span_flag" in self.plasmadf.columns:
                self.plasmadf[self.plasmadf["span_flag"] == 0] = np.nan
                self.plasmadf[self.plasmadf["V_R"] > 0] = np.nan
                self.plasmadf[self.plasmadf["V_T"] < -1000] = np.nan
                self.plasmadf[self.plasmadf["V_N"] < -1000] = np.nan

                self.plasmadf[self.plasmadf["T"] < 0] = np.nan
                self.plasmadf[self.plasmadf["N"] < 0] = np.nan
                del self.plasmadf["span_flag"]

            self.plasmadf.fillna(method="pad")
            self.plasmadf = self.plasmadf.resample(f"{self.obj_cad}s").mean()

            #############################################################
            # MAG #
            # Magnetic field measurements
            self.magdf.index = pd.to_datetime(self.magdf["Time"])
            del self.magdf["Time"]

            # For PSP data cleaning
            if "fld_flag" in self.magdf.columns:
                self.magdf[self.magdf["fld_flag"] >= 1] = np.nan
                del self.magdf["fld_flag"]

            # For SolO data cleaning
            if "mag_flag" in self.magdf.columns:
                self.magdf[self.magdf["mag_flag"] < 3] = np.nan
                del self.magdf["mag_flag"]

            self.magdf.fillna(method="pad")
            self.magdf = self.magdf.resample(f"{self.obj_cad}s").mean()

            df = pd.concat([self.plasmadf, self.magdf], join="outer", axis=1).fillna(
                np.nan
            )

            self.df = df

        else:
            # After loading it breaks datetime format -> We save as the csv for ease of use
            self.df.index = pd.to_datetime(self.df["Time"])
            del self.df["Time"]
            # self.df.fillna(method="pad")
            # self.df.interpolate(inplace=True)

        self.df = self.df.resample(f"{self.obj_cad}s").mean()
        self.df["Time"] = self.df.index
        self.df.to_csv(
            self.completeCSVPath,
            index=False,
        )

        del self.df["Time"]

    @staticmethod
    def traceFlines(lon, lat, r, vSW, rf=0.0046547454 * u.AU, accelerated=False):
        """
        For each of the positions in self, track to solar surface and return field lines
        lon, lat in deg
        r in AU
        """

        # Velocity and Time of observation
        vsw = (vSW["V_R"] * u.km / u.s).to(u.AU / u.s).value
        time_spc = vSW.name.to_pydatetime()

        # Calculate dt s/c -> rfinal
        dt = (r - rf).value / vsw
        dt = (4 / 3) * dt if accelerated else dt
        time_sun = time_spc - timedelta(seconds=dt)
        rotation_deg = (SOLROTRATE * r.value) / vsw

        # Steps in longitude. Main ballistic backmapping
        # First coordinate = SolO?
        lon_ss = lon + rotation_deg * u.deg
        lon_x = np.arange(lon.value, lon_ss.value, step=2)

        # Steps in latitude (is conserved)
        lat_x = np.repeat(lat.value, len(lon_x))

        # Steps in the radius
        step_rad = (r.value - rf.value) / len(lon_x)
        rad_x = np.arange(rf.value, r.value, step=step_rad)[::-1]

        # Steps in the time
        step_time = (time_spc - time_sun).total_seconds() / len(lon_x)
        fline_times = [
            (time_spc - timedelta(seconds=step_time * timestep))
            for timestep in range(len(lon_x))
        ]

        rad_x = rad_x[:-1] if (len(rad_x) > len(lon_x)) else rad_x

        # Field line. Longitude, Latitude, Radius, Timestamp
        assert len(lon_x) == len(rad_x), "Different lon, lat, r size."

        fline = (
            lon_x * u.deg,
            lat_x * u.deg,
            rad_x * u.AU,
            fline_times,
        )  # Compose long, lat, rad into fieldlines

        return fline

    def saveWithRadius(self):
        """
        Save a new csv with the radius information
        """
        self.extract_orbit_data(from_data=True, stepMinutes=self.obj_cad / 60)
        self.df = self.df[:-1]
        self.df["R"] = (self.sp_traj.r.to(u.km)).value
        self.df["Time"] = self.df.index
        self.df.to_csv(
            self.completeCSVPath,
            index=False,
        )

    def plot_solo_psp_df(self, other, zones=[]):
        """
        Plot the dataframe in the style of the previous paper
        Other must be PSP
        zone 1 and zone 2 will be highlighted in SolO and PSP respectively
        """
        assert "solo" in self.name.lower(), "Please ensure SolO is object with f call"
        import matplotlib
        from astropy import constants as c

        plt.style.use("seaborn-paper")

        matplotlib.rcParams["axes.titlesize"] = 18
        matplotlib.rcParams["axes.labelsize"] = 18
        matplotlib.rcParams["figure.titlesize"] = 20
        matplotlib.rcParams["xtick.labelsize"] = 16
        matplotlib.rcParams["ytick.labelsize"] = 16

        # Figure
        # Width and marker size
        _, axs = plt.subplots(
            8, 1, figsize=(16, 2 * 5), sharex=True, constrained_layout=True
        )

        self.df["R"] = self.df["R"] * 1000  # Change to m
        other.df["R"] = other.df["R"] * 1000  # Change to m

        Bt = np.sqrt(self.df["B_R"] ** 2 + self.df["B_T"] ** 2 + self.df["B_N"] ** 2)
        BtScaled = Bt * self.df["R"] ** 2
        Vx = self.df["V_R"]
        Np = self.df["N"]
        NpScaled = Np * self.df["R"] ** 2
        NpRPW = self.df["N_RPW"]
        NpRPWScaled = NpRPW * self.df["R"] ** 2
        T = self.df["T"]

        Mf = c.m_p.value * Np * 10 ** 15 * (-Vx * 1000)
        MfScaled = Mf * self.df["R"] ** 2

        oBTUnscaled = np.sqrt(
            other.df["B_R"] ** 2 + other.df["B_T"] ** 2 + other.df["B_N"] ** 2
        )
        oBtScaled = oBTUnscaled * other.df["R"] ** 2
        oVx = -other.df["V_R"]
        oNp = other.df["N"]
        oNpScaled = oNp * other.df["R"] ** 2
        oT = other.df["T"]
        oMf = c.m_p.value * oNp * 10 ** 15 * (oVx * 1000)
        oMfScaled = oMf * other.df["R"] ** 2

        # Magnetic field components
        ax0 = axs[0]
        ax0.set_ylabel(r"$\hat{B}_T(nT)$")
        ax0.semilogy(
            Bt,
            "k-",
            label="SolO_MAG",
            alpha=1,
            linewidth=1,
        )
        ax0.semilogy(
            oBTUnscaled,
            "r-",
            label="PSP_FLD",
            alpha=1,
        )
        ax0.legend()

        # Radial velocity
        ax1 = axs[1]
        ax1.plot(Vx, color="black", label="Vp [GSE]", linewidth=1)
        ax1.plot(oVx, color="red")
        ax1.set_ylabel(r"$-{V_x} (kms^{-1})$")

        # Temperature
        ax2 = axs[2]
        ax2.plot(
            T,
            color="black",
            label=r"$T_p$",
            linewidth=1,
        )
        ax2.plot(oT, color="red")
        ax2.set_ylabel(r"$T_p (K)$")

        # Proton Density
        ax4 = axs[3]
        ax4.semilogy(
            Np,
            color="black",
            label="Np_SolO_SWA_PAS",
            linewidth=1,
        )
        ax4.semilogy(NpRPW, color="blue", alpha=0.4, label="Np_SolO_RPW")
        ax4.semilogy(oNp, color="red", label="Np_PSP_SPAN")
        ax4.set_ylabel(r"$n_p$ (# $cm^{-3}$)")
        ax4.legend()

        # Pressure
        axMf = axs[4]
        axMf.semilogy(
            -Mf,
            label="Mass Flux",
            color="black",
            linewidth=1,
        )
        axMf.semilogy(oMf, color="red")
        axMf.set_ylabel(r"$-Mf$")

        # Magnetic field components
        axs[5].set_ylabel(r"$\hat{B}_Total$")
        axs[5].plot(
            BtScaled,
            "k-",
            label=r"$R^2$ Scaled Btotal SolO",
        )
        axs[5].plot(
            oBtScaled,
            "r-",
            label=r"$R^2$ Scaled Btotal PSP",
        )
        axs[5].legend()

        # Proton Density
        axs[6].plot(
            NpScaled,
            color="black",
            label=r"$R^2$ Scaled Np (PAS)",
            linewidth=1,
        )
        axs[6].plot(
            NpRPWScaled,
            color="blue",
            alpha=0.4,
            label=r"$R^2$ Scaled Np (RPW)",
        )
        axs[6].plot(
            oNpScaled,
            color="red",
            label=r"$R^2$ Scaled Np (PSP_SWEAP)",
        )
        axs[6].set_ylabel(r"$n_p$")
        axs[6].legend()

        axs[7].plot(
            -MfScaled, color="black", linewidth=1, label=r"$R^2$ Scaled Mf SolO"
        )
        axs[7].plot(oMfScaled, color="red", label=r"$R^2$ Scaled Mf PSP")
        axs[7].set_ylabel(r"$-Mf$")
        axs[7].legend()

        # # Plot the relevant columns
        for ax in axs:
            ax.set_xlim(datetime(2020, 9, 24, 12), datetime(2020, 10, 5))
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        # Plot the zones being considered
        for zone in zones:
            for ax in axs:
                ax.axvspan(
                    zone["start_time"], zone["end_time"], color=zone["color"], alpha=0.3
                )

        # total magnetic field strength, proton number density, solarwind bulk flow velocity and mass flux,
        # print(f"Saving to {self}")
        # plt.savefig(f"{save_folder}B_Br_V_sb_matrix_highlighted.pdf")
        if self.show:
            plt.show()
        else:
            plotPath = f"{BASE_PATH}Figures/Timeseries/"
            makedirs(plotPath, exist_ok=True)
            plt.savefig(f"{plotPath}summaryPlot.png")

        plt.close()

    def extract_orbit_data(self, from_data=False, stepMinutes=30) -> None:
        """
        Extracts orbital data
        """
        import heliopy.data.spice as spicedata
        import heliopy.spice as spice

        if "psp" in self.name.lower():
            spicedata.get_kernel("psp")
            spicedata.get_kernel("psp_pred")
            sp_traj = spice.Trajectory("SPP")

        elif "solo" in self.name.lower():
            spicedata.get_kernel("solo")
            sp_traj = spice.Trajectory("SolO")

        else:
            raise ValueError(f"{self.name} not identified")

        import astropy.units as u
        from astropy.visualization import quantity_support

        quantity_support()

        # Generate a time for every day between starttime and endtime
        if from_data == False:
            starttime = self.start_time
            endtime = self.end_time

        # When generating from data it utilises data in the index
        else:
            starttime = self.df.index[0].to_pydatetime()
            endtime = self.df.index[len(self.df.index) - 1].to_pydatetime()

        # We set a step in minutes to generate positions
        times = []
        while starttime < endtime:
            times.append(starttime)
            starttime += timedelta(minutes=stepMinutes)

        # Generate positions and store as sp_traj in AU
        sp_traj.generate_positions(times, "Sun", "IAU_SUN")
        sp_traj.change_units(u.au)
        self.sp_traj = sp_traj

        from sunpy.coordinates import frames

        self.sp_coords_carrington = sp_traj.coords.transform_to(
            frame=frames.HeliographicCarrington
        )

        self.sp_coords_hcentric = sp_traj.coords.transform_to(
            frame=frames.HeliocentricEarthEcliptic
        )

    def plot_top_down(
        farFromSun,
        closetoSun,
        objFolder=f"{BASE_PATH}Figures/Orbit_3d/",
        plotRate="24h",
        hlight=(datetime(2020, 10, 2, 4), datetime(2020, 9, 27, 0)),
        farTime="22:00",
        closeTime="04:00",
        pspiralHlight=datetime(2020, 8, 28, 15, 30),
    ):
        """
        Plots the longitude and latitude of a given spacecraft object
        plotRate = How often to plot the orbits
        hlight = for farFrom, closeTo, which date to highlight (closest to!)
        """
        makedirs(objFolder, exist_ok=True)

        assert farFromSun.sp_traj != None, "Please calculate the spacecraft trajectory"
        lon1 = farFromSun.sp_coords_hcentric.lon
        lat1 = farFromSun.sp_coords_hcentric.lat
        rad1 = farFromSun.sp_coords_hcentric.distance
        t1 = farFromSun.sp_coords_hcentric.obstime.datetime

        # When using 2
        lon2 = closetoSun.sp_coords_hcentric.lon
        lat2 = closetoSun.sp_coords_hcentric.lat
        rad2 = closetoSun.sp_coords_hcentric.distance
        t2 = closetoSun.sp_coords_hcentric.obstime.datetime

        X1, Y1, Z1 = spherical_to_cartesian(rad1, lat1, lon1)
        X2, Y2, Z2 = spherical_to_cartesian(rad2, lat2, lon2)

        # Dataframes on Cartesian coords
        df_farFromSun = pd.DataFrame(
            {
                "X": X1,
                "Y": Y1,
                "Z": Z1,
                "lon": lon1,
                "lat": lat1,
                "rad": rad1,
            },
            index=t1,
        )
        df_nextToSun = pd.DataFrame(
            {
                "X": X2,
                "Y": Y2,
                "Z": Z2,
                "lon": lon2,
                "lat": lat2,
                "rad": rad2,
            },
            index=t2,
        )

        # Reduce the datasets
        solo_times, psp_times = [], []

        for time in pd.date_range(
            datetime(2020, 9, 27, 22), datetime(2020, 10, 4, 22), freq=f"{plotRate}"
        ):
            solo_times.append(time.to_pydatetime())

        for time in pd.date_range(
            datetime(2020, 9, 25, 4), datetime(2020, 10, 1, 4), freq=f"{plotRate}"
        ):
            psp_times.append(time.to_pydatetime())

        df_farFromSun = df_farFromSun.resample(f"{plotRate}", origin=farTime).mean()
        df_nextToSun = df_nextToSun.resample(f"{plotRate}", origin=closeTime).mean()

        # Find radial alignment
        slopeIntList = []
        slopeInt = namedtuple("slopeInt", ["m", "b"])
        for r, X1 in enumerate(df_nextToSun["X"].values):
            P1 = (0, 0)
            P2 = (df_nextToSun["X"].values[r], df_nextToSun["Y"].values[r])
            m, b = lineCalc(P1, P2)
            slopeIntList.append(slopeInt(m, b))

        # Figure of the two orbits with field lines
        plt.figure(figsize=(12, 12))
        plt.xlim(-0.655, 0.094)
        plt.ylim(-0.116, 0.835)

        hlightIndices = (
            df_farFromSun.index.get_loc(hlight[0], method="nearest"),
            df_nextToSun.index.get_loc(hlight[1], method="nearest"),
        )
        pspiralIndex = df_nextToSun.index.get_loc(pspiralHlight, method="nearest")

        plt.scatter(
            df_farFromSun["X"],
            df_farFromSun["Y"],
            alpha=0.6,
        )
        plt.scatter(
            df_nextToSun["X"],
            df_nextToSun["Y"],
            alpha=0.6,
        )

        # Annotate dates
        for i, txt in enumerate(df_nextToSun.index):
            if i in [hlightIndices[1], pspiralIndex]:
                plt.gca().annotate(
                    datetime.strftime(txt, "%d-%H:%M"),
                    (df_nextToSun["X"][i], df_nextToSun["Y"][i] + 0.005),
                    size=15,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
                )

        # Radial alignment timings
        radAlignmentPSP = (
            df_nextToSun["X"][hlightIndices[1]],
            df_nextToSun["Y"][hlightIndices[1]],
        )
        radAlignmentSolO = (
            df_farFromSun["X"][hlightIndices[0]],
            df_farFromSun["Y"][hlightIndices[0]],
        )

        # For each of the measurements, calculate distance between PSP and SolO

        plt.scatter(
            radAlignmentSolO[0],
            radAlignmentSolO[1],
            color="black",
            s=100,
            label=df_farFromSun.index[hlightIndices[0]],
        )
        plt.scatter(
            radAlignmentPSP[0],
            radAlignmentPSP[1],
            color="red",
            s=30,
            label=df_nextToSun.index[hlightIndices[1]],
        )

        # plot radial
        for i, line in enumerate(slopeIntList):
            x = (
                np.linspace(0.0, -1.0)
                if df_nextToSun["X"].values[i] < 0
                else np.linspace(0.0, 0.1)
            )
            y = line.m * x + line.b

            tline = df_nextToSun.index[i]
            _col = "orange" if tline != df_nextToSun.index[hlightIndices[1]] else "red"
            plt.plot(x, y, color=_col)

        # plot spiral
        pspSpirals = findSpirals(df=df_nextToSun, vSW=314, rads=(0, 1.3))

        for i, spiral in enumerate(pspSpirals):
            x = spiral.X
            y = spiral.Y
            _col, _alpha = ("orange", 0.5) if i != pspiralIndex else ("red", 0.8)
            plt.plot(x, y, color=_col, alpha=_alpha)

        soloSpirals = findSpirals(df_farFromSun, vSW=330, rads=(0, 1.3))
        for spiral in soloSpirals:
            x = spiral.X
            y = spiral.Y
            plt.plot(x, y, color="gray", alpha=0.5)

        plt.scatter([0], [0], s=300, color="orange", marker="*")
        plt.grid(True)
        plt.xlabel("Cartesian X (AU)")
        plt.ylabel("Cartesian Y (AU)")
        plt.title("SolO and PSP positions with radial and Parker Spiral information")
        plt.legend()

        # Plot the Parker Spiral lines
        plt.savefig(f"{objFolder}soloPSP_rate_{plotRate}")
        if farFromSun.show == True:
            plt.show()

    def plot_spherical_coords(
        self, other, accelerated=False, objFolder=f"{BASE_PATH}Figures/Orbit_3d/"
    ):
        """
        Plots the longitude and latitude of a given spacecraft object
        """
        raise ValueError("Highly likely this is wrong!")

        makedirs(objFolder, exist_ok=True)

        assert self.sp_traj != None, "Please calculate the spacecraft trajectory"

        lon1 = self.sp_coords_carrington.lon
        lat1 = self.sp_coords_carrington.lat
        rad1 = self.sp_coords_carrington.radius

        # When using 2
        # lon2 = other.sp_coords_carrington.lon
        # lat2 = other.sp_coords_carrington.lat
        # rad2 = other.sp_coords_carrington.radius

        # X1, Y1, Z1 = spherical_to_cartesian(rad1, lat1, lon1)
        # X2, Y2, Z2 = spherical_to_cartesian(rad2, lat2, lon2)

        fline_set = []

        for i, t in enumerate(self.sp_coords_carrington.obstime):

            # Use the dataframe which matches cadence of carrington maps
            vsw = fcl(self.df_orbit_match, t.datetime)

            lons, lats, rs, times = self.traceFlines(
                lon1[i],
                lat1[i],
                rad1[i],
                vSW=vsw,
                accelerated=accelerated,
            )

            X, Y, Z = spherical_to_cartesian(r=rs, lat=lats, lon=lons)
            # Transform fiels line to cartesian and save timestamps.
            fline_set.append(
                pd.DataFrame(
                    {
                        "X": X,
                        "Y": Y,
                        "Z": Z,
                    },
                    index=times,
                )
            )

        scTrajDF = pd.DataFrame(
            {
                "X": other.sp_traj.x,
                "Y": other.sp_traj.y,
                "Z": other.sp_traj.z,
            },
            index=other.sp_traj.times,
        )

        def plot_over_time(
            scTrajDF,
            fline_set,
            marginHours=0.1,
            objFolder=f"{BASE_PATH}Figures/Orbit_3d/",
        ):
            """
            Plots a time dependent 3d plot
            :param scTrajDF: Sc trajectory dataframe (Index = Datetime, X, Y, Z Cartesian)
            :param fline_set: A set of field lines
            """

            # Set the colour cycle
            colors = cycle(["red", "blue", "green", "black"])
            c = {}
            for idx in range(len(fline_set)):
                c[idx] = next(colors)

            # scCoords = (other.sp_traj.x, other.sp_traj.y, other.sp_traj.z)
            # scTime = other.sp_traj.times

            for i, scCoord in enumerate(scTrajDF.iterrows()):
                # 3d figure
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection="3d")

                # Solar radius 0.0046547454 AU
                ax.scatter(0, 0, 0, label="Sun", c="orange", s=100)

                # Plot the two spacecraft (OVER TIME)
                # ax.scatter(X1, Y1, Z1, label=self.name, s=5, c="blue")
                # ax.scatter(X2, Y2, Z2, label=other.name, s=5, c="red")

                # Time for spacecraft coords
                relTime = scCoord[0].value
                start_time, end_time = (
                    relTime - timedelta(hours=marginHours),
                    relTime + timedelta(hours=marginHours),
                )
                print(scCoord[1])
                Xsc, Ysc, Zsc = scCoord[1]

                ax.scatter(Xsc, Ysc, Zsc, s=25, c="black", label="PSP")

                # For each of the relevant field lines?
                for index, flDF in enumerate(fline_set):

                    # Set the relevant fied lines with mask
                    flDF_relevant = flDF[end_time:start_time]

                    # Plot points close in time to PSP
                    if not flDF_relevant.empty:
                        solo_time = flDF.index[0]
                        fl_time = flDF_relevant.index[0]
                        solo_fl_dt = (solo_time - fl_time).total_seconds() / 3600

                        ax.scatter(
                            flDF_relevant["X"],
                            flDF_relevant["Y"],
                            flDF_relevant["Z"],
                            s=8,
                            c=c[index],
                            label=f"Fline origin {solo_time} => {fl_time} ({solo_fl_dt:.0f}h)",
                        )

                ax.set_xlim(-0.08, 0.08)
                ax.set_ylim(-0.08, 0.08)
                ax.set_zlim(-0.08, 0.08)

                plt.legend()
                # ax.view_init(elev=90)
                ax.set_title(
                    f"PSP Timestamp: {relTime.__str__()} +/- {marginHours} hours"
                )
                print(f"Saving to {objFolder}{i:02d}")
                plt.savefig(f"{objFolder}{i:02d}.png")
                # plt.show()
                plt.close()

        plot_over_time(scTrajDF, fline_set, objFolder=objFolder)

    def zoom_in(
        self,
        start_time: datetime,
        end_time: datetime,
        color=None,
        stepMinutes=30,
        extractOrbit=True,
    ):
        """
        Zoom into a specific part of the signal and store new self.df in memory
        Uses start_time and end_time to define limits
        stepMinutes: generates dataframe
        """
        self.start_time = start_time
        self.end_time = end_time

        if extractOrbit:
            self.extract_orbit_data(stepMinutes=stepMinutes)

            # Generate a dataframe that matches in cadence the orbital data
            self.df_orbit_match = self.df.resample(f"{stepMinutes}min").mean()[
                self.sp_traj.coords.obstime[0]
                .datetime : self.sp_traj.coords.obstime[-1]
                .datetime
            ]

        # Cut down the dataframe and maintain higher cadence
        self.df = self.df[self.start_time : self.end_time]


# %%
def psp_e6():
    """
    Parker Encounter 6, estimated to take from 26 Sept to 7 October 2020.
    Have Potential PSP and SolO Conjunction
    Must study the SolO data and link it back to PSP
    Then apply algorithm and observe results
    """
    # Parker Encounter 6
    # psp_fld_l2_mag_RTN_4_Sa_per_Cyc_20201002_v01
    OBJ_CADENCE = 60  # To one minute resolution
    ORBIT_GAP = 30  # Orbital gap between measurements in Minutes
    PLOT_ORBITS = True
    SHOW_PLOTS = True

    psp_e6_overview = {
        "name": "PSPpriv_e6",
        "mid_time": datetime(2020, 10, 2),
        "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
    }

    # SHOW can be changed here
    solo_e6_overview = {
        "name": "SolOpriv_e6",
        "mid_time": datetime(2020, 10, 3),
        "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
        "show": SHOW_PLOTS,
    }

    # Prepare the objects with measurements inside DF
    psp = Spacecraft(**psp_e6_overview)
    solo = Spacecraft(**solo_e6_overview)

    # Solar orbiter lines up with latitude?
    # In september
    # TODO: Should make multiple test cases using different solo_zoomed and psp_zoomed profiles.
    solo_zoomed = {
        "start_time": datetime(2020, 9, 27, 11, 30),
        "end_time": datetime(2020, 10, 4, 11, 53),
        "stepMinutes": ORBIT_GAP,
        "color": "black",
    }

    psp_zoomed = {
        "start_time": datetime(2020, 9, 25),
        "end_time": datetime(2020, 9, 30),
        "stepMinutes": ORBIT_GAP,
        "color": "red",
    }

    solo_paper = {
        "start_time": datetime(2020, 10, 2, 0),
        "end_time": datetime(2020, 10, 2, 1, 30),
        "color": "black",
    }

    psp_paper = {
        "start_time": datetime(2020, 9, 27, 4),
        "end_time": datetime(2020, 9, 27, 5, 30),
        "color": "red",
    }

    solo.plot_solo_psp_df(psp, zones=[solo_paper, psp_paper])
    solo.zoom_in(**solo_zoomed)
    psp.zoom_in(**psp_zoomed)

    # Resample to an objective cadence
    psp.df = psp.df.resample(f"{OBJ_CADENCE}s").mean()
    solo.df = solo.df.resample(f"{OBJ_CADENCE}s").mean()

    # print(solo.df)
    # print(psp.df)

    # Remove all blanks
    solo.df.fillna(method="pad")
    psp.df.fillna(method="pad")

    for case in range(1):
        # These are cut to the time
        solo.df["Time"] = solo.df.index
        psp.df["Time"] = psp.df.index
        orbit_case_path = f"{BASE_PATH}Figures/Orbit_3d/Case_{case:02d}/"
        makedirs(orbit_case_path, exist_ok=True)
        solo.df.to_csv(
            f"{orbit_case_path}solo_case{case:02d}.csv",
            index=False,
        )

        psp.df.to_csv(
            f"{orbit_case_path}psp_case{case:02d}.csv",
            index=False,
        )

        # Spacecraft object calling the function is where the solar wind is being mapped from
        if PLOT_ORBITS:
            solo.plot_top_down(
                psp,
                objFolder=f"{orbit_case_path}/",
                plotRate="12h",
                farTime="12:00",
                closeTime="15:30",
                pspiralHlight=datetime(2020, 9, 28, 15, 30),
            )


def explore_other_E():
    """
    Parker Encounter 6, estimated to take from 26 Sept to 7 October 2020.
    Have Potential PSP and SolO Conjunction
    Must study the SolO data and link it back to PSP
    Then apply algorithm and observe results
    """
    # psp_fld_l2_mag_RTN_4_Sa_per_Cyc_20201002_v01
    OBJ_CADENCE = 60  # To one minute resolution
    ORBIT_GAP = 30  # Orbital gap between measurements in Minutes
    PLOT_ORBITS = True
    SHOW_PLOTS = True

    psp_e6_overview = {
        "name": "PSPpriv_e6",
        "mid_time": datetime(2020, 10, 2),
        "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
    }

    # SHOW can be changed here
    solo_e6_overview = {
        "name": "SolOpriv_e6",
        "mid_time": datetime(2020, 10, 3),
        "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
        "show": SHOW_PLOTS,
    }

    # Prepare the objects with measurements inside DF
    psp = Spacecraft(**psp_e6_overview)
    solo = Spacecraft(**solo_e6_overview)

    # Solar orbiter lines up with latitude?
    # In september
    # TODO: Should make multiple test cases using different solo_zoomed and psp_zoomed profiles.
    solo_zoomed = {
        "start_time": datetime(2020, 9, 27, 11, 30),
        "end_time": datetime(2020, 10, 4, 11, 53),
        "stepMinutes": ORBIT_GAP,
        "color": "black",
    }

    psp_zoomed = {
        "start_time": datetime(2020, 9, 25),
        "end_time": datetime(2020, 9, 30),
        "stepMinutes": ORBIT_GAP,
        "color": "red",
    }

    solo_paper = {
        "start_time": datetime(2020, 10, 2, 0),
        "end_time": datetime(2020, 10, 2, 1, 30),
        "color": "black",
    }

    psp_paper = {
        "start_time": datetime(2020, 9, 27, 4),
        "end_time": datetime(2020, 9, 27, 5, 30),
        "color": "red",
    }

    solo.plot_solo_psp_df(psp, zones=[solo_paper, psp_paper])
    solo.zoom_in(**solo_zoomed)
    psp.zoom_in(**psp_zoomed)

    # Resample to an objective cadence
    psp.df = psp.df.resample(f"{OBJ_CADENCE}s").mean()
    solo.df = solo.df.resample(f"{OBJ_CADENCE}s").mean()

    # print(solo.df)
    # print(psp.df)

    # Remove all blanks
    solo.df.fillna(method="pad")
    psp.df.fillna(method="pad")

    for case in range(1):
        # These are cut to the time
        solo.df["Time"] = solo.df.index
        psp.df["Time"] = psp.df.index
        orbit_case_path = f"{BASE_PATH}Figures/Orbit_3d/Case_{case:02d}/"
        makedirs(orbit_case_path, exist_ok=True)
        solo.df.to_csv(
            f"{orbit_case_path}solo_case{case:02d}.csv",
            index=False,
        )

        psp.df.to_csv(
            f"{orbit_case_path}psp_case{case:02d}.csv",
            index=False,
        )

        # Spacecraft object calling the function is where the solar wind is being mapped from
        if PLOT_ORBITS:
            solo.plot_top_down(
                psp,
                objFolder=f"{orbit_case_path}/",
                plotRate="12h",
                farTime="12:00",
                closeTime="15:30",
                pspiralHlight=datetime(2020, 9, 28, 15, 30),
            )


# %%

if __name__ == "__main__":
    psp_e6()
