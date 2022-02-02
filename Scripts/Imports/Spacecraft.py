from sys import path
from os import makedirs
from copy import deepcopy
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.table import QTable
import astropy.units as u
import astropy.constants as const
from datetime import timedelta, datetime
from collections import namedtuple
import warnings

warnings.filterwarnings("ignore")

from pyparsing import col

BASE_PATH = "/Users/ddp/Documents/PhD/inEMD_Github/"

path.append(f"{BASE_PATH}Scripts/")

from Plots.cdfReader import extractDF

locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
unitsTable = {
    "V_R": u.km / u.s,
    "V_T": u.km / u.s,
    "V_N": u.km / u.s,
    "N": 1 / u.cm ** (-3),
    "N_RPW": 1 / u.cm ** (-3),
    "B_R": u.nT,
    "B_T": u.nT,
    "B_N": u.nT,
    "R": u.km,
}
SOLROTRATE = 24.47 / 86400

from cycler import cycler
import matplotlib

params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "axes.prop_cycle": cycler(color=["b", "g", "y", "k"]),
}
matplotlib.rcParams.update(params)


def lineCalc(P1, P2):
    """
    Calculate line given 2 points (X1, Y1), (X2, Y2)
    """
    m = (P2[1] - P1[1]) / (P2[0] - P1[0])
    b = P1[1] + m * P1[0]
    return m, b


# Find p spiral alignment
def findSpirals(df, vSW, rads=(0, 0.9)):
    from heliopy.models import ParkerSpiral
    from astropy.coordinates import spherical_to_cartesian

    _spirals = []
    Spiral = namedtuple("Spiral", ["X", "Y", "Z"])
    for t, measurements in df.iterrows():
        r0 = measurements["rad"]
        l0 = measurements["lon"]
        spiral = ParkerSpiral(vSW * u.km / u.s, r0 * u.au, l0 * u.deg)

        if len(rads) == 2:
            rads = np.linspace(*rads)
        longs = spiral.longitude(rads * u.au)

        # This works once the coordinate frame is fixed
        cartSpiral = spherical_to_cartesian(rads, np.repeat(0, len(longs)), longs)
        _spirals.append(Spiral(*cartSpiral))

    return _spirals


def _alignmentIdentification(soloDF, pspDF, vSWPSP):
    """
    Calculate longitude at PSP for the SolO measurements, based on SW speed at PSP
    soloSPC: Spacecraft object SolO

    pspRadius: Radius of PSP to Sun at t1
    vSWPSP: Solar wind speed at PSP
    """
    soloRadius = (soloDF["rad"].mean() * u.au).to(u.km).value
    pspRadius = (pspDF[1]["rad"] * u.au).to(u.km).value

    tPSP = pspDF[1].name.to_pydatetime()
    pspTrueLon = pspDF[1]["lon"]
    dt = (soloRadius - pspRadius) / vSWPSP  # In seconds

    tSolo = tPSP + timedelta(seconds=dt)
    indexClosest = soloDF.index.get_loc(tSolo, method="nearest")
    phiPSP = soloDF["lon"][indexClosest]

    # End of 1st October looks to have ~ 10 degree diference
    return (tPSP, tSolo, (pspTrueLon - phiPSP))


def gse_calculate_rad(coords):
    r = np.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
    return r


class Spacecraft:
    """
    The intention of this class is to enable similar download and
    storage of WIND, STEREO A, PSP and SolO observations.

    """

    def __init__(
        self,
        name,
        cadence_obj,
        show=False,
        sunEarthDist=150111200.76,
        remakeCSV=False,
    ):
        """
        :param name: Spacecraft name from one of [WIND, PSP, ST-A]
        :param cadence_obj: Objective cadence
        """
        self.name = name
        self.obj_cad = cadence_obj
        self.show = show
        self.completeCSVPath = f"{BASE_PATH}Data/Prepped_csv/{self.name}.csv"
        self.scaledCSVPath = f"{BASE_PATH}Data/Prepped_csv/{self.name}.csv"
        self.sunEarthDist = sunEarthDist

        if not remakeCSV:
            try:
                self.df = pd.read_csv(self.completeCSVPath)
                self.df.index = pd.to_datetime(self.df["Time"])

                if "R" not in self.df.columns and "Scaled" not in self.name:
                    self.saveWithRadius()
                    raise ValueError(
                        f"Saved {self.name} With Radius added to {self.completeCSVPath}. Please re-run"
                    )

                del self.df["Time"]
                self.df = self.df.resample(f"{self.obj_cad}s").mean()
                self.dfUnits = QTable.from_pandas(self.df, index=True)

                for _i in self.dfUnits.colnames:
                    if _i == "N":
                        self.dfUnits["N"] = self.dfUnits["N"] * u.cm ** (-3)

                    elif _i == "N_RPW":
                        self.dfUnits["N_RPW"] = self.dfUnits["N_RPW"] * u.cm ** (-3)

                    elif _i in unitsTable:
                        self.dfUnits[_i] = self.dfUnits[_i] * unitsTable[_i]
                return None

            except FileNotFoundError:
                pass

        self.sp_coords_carrington = None  # Set a param which is empty to then fill

        # Get the spacecraft data into the proper format

        if self.name == "SolO_Scaled_e6":
            self.df = pd.read_csv(f"{BASE_PATH}Data/Prepped_csv/SolO_POST_e6.csv")
        elif self.name == "SolOpriv_e6" or self.name == "SolO_April_2020":
            # Set the csv paths to empty
            df_mag_csv_path = None
            df_rpw_csv_path = None
            df_sweap_csv_path = None

            if self.name == "SolOpriv_e6":
                df_mag_csv_path = f"{BASE_PATH}Data/Prepped_csv/solo_mag_sept.csv"
                df_rpw_csv_path = f"{BASE_PATH}Data/Prepped_csv/solo_rpw_sept.csv"
                df_sweap_csv_path = f"{BASE_PATH}Data/Prepped_csv/solo_sweap_sept.csv"

            elif self.name == "SolO_April_2020":
                df_mag_csv_path = f"{BASE_PATH}Data/Prepped_csv/solo_mag_sept.csv"
                self.df = None

            # Functions that process the cdfs, change cdf_path to be input if required
            def RPW_prep(cdf_path=f"{BASE_PATH}unsafe/Resources/Solo_Data/L3/RPW/"):

                _rpw_df = extractDF(
                    cdf_path,
                    dfVars=["DENSITY"],
                    info=False,
                    resample=f"{self.obj_cad}s",
                )

                _rpw_df[_rpw_df["DENSITY"] < 1] = np.nan
                _rpw_df["N_RPW"] = _rpw_df["DENSITY"]
                del _rpw_df["DENSITY"]

                _rpw_df["Time"] = _rpw_df.index
                _rpw_df.to_csv(df_rpw_csv_path, index=False)
                del _rpw_df["Time"]

                return _rpw_df

            def SWA_PAS(
                cdf_path=f"{BASE_PATH}unsafe/Resources/Solo_Data/L2/GroundMom/",
            ):
                # SWA-PAS data
                """Reduced list
                'Epoch',
                'validity',
                'N',
                'V_RTN',
                'T'
                """

                _sweap_df = extractDF(
                    cdf_path,
                    dfVars=["V_RTN", "N", "T", "validity"],
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

            def MAG_prep(cdf_path):
                """
                Epoch
                B_RTN
                QUALITY_FLAG
                """
                # Need to separate into different relevant csv
                _mag_df = extractDF(
                    cdf_path, dfVars=["B_RTN", "QUALITY_FLAG"], info=False
                )
                _mag_df["mag_flag"] = _mag_df["QUALITY_FLAG"]
                del _mag_df["QUALITY_FLAG"]

                _mag_df["Time"] = _mag_df.index
                _mag_df.to_csv(df_mag_csv_path, index=False)
                return _mag_df

            if df_rpw_csv_path == None:
                self.magdf = MAG_prep(
                    cdf_path=f"{BASE_PATH}unsafe/Resources/SolO_April_2020/SOlO_Data/L2/Mag/"
                )

            else:
                _sweapdf = SWA_PAS()
                _rpwdf = RPW_prep()
                self.plasmadf = pd.concat(
                    [_sweapdf, _rpwdf], join="outer", axis=1
                ).fillna(np.nan)
                self.magdf = MAG_prep(
                    cdf_path=f"{BASE_PATH}unsafe/Resources/Solo_Data/L2/Mag/"
                )

                self.df = None
        elif self.name == "PSP_Scaled_e6":
            # Just load the measurements
            self.df = pd.read_csv(f"{BASE_PATH}Data/Prepped_csv/psp_POST_e6.csv")
        elif self.name == "PSPpriv_e6" or self.name == "PSP_Nov_2019":
            """
            This is the Encounter 6 data
            """
            if self.name == "PSPPriv_e6":
                df_fld_csv_path = f"{BASE_PATH}Data/Prepped_csv/psp_mag_sept.csv"
                df_span_csv_path = f"{BASE_PATH}Data/Prepped_csv/psp_span_sept.csv"
                fldCDFPath = f"{BASE_PATH}unsafe/Resources/PSP_Data/FIELDS/"

            elif self.name == "PSP_Nov_2019":
                df_fld_csv_path = self.completeCSVPath
                fldCDFPath = f"{BASE_PATH}unsafe/Resources/STA_PSP_Nov_2019/PSP/"

            def FLD_prep():
                # When file not available, derive it
                cdf_path = fldCDFPath

                _fld_df = extractDF(
                    CDFfolder=cdf_path,
                    dfVars=["psp_fld_l2_mag_RTN_1min", "psp_fld_l2_quality_flags"],
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
                    dfVars=["DENS", "VEL", "TEMP", "QUALITY_FLAG"],
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

            if self.name == "PSPpriv_e6":
                self.magdf = FLD_prep()
                self.plasmadf = SPAN_AI_prep()
                self.df = None

            elif self.name == "PSP_Nov_2019":
                self.magdf = FLD_prep()
                self.df = None
        elif self.name == "STA_Nov_2019":
            dataPath = f"{BASE_PATH}unsafe/Resources/STA_PSP_Nov_2019/"
            df_sta_csv_path = self.completeCSVPath

            self.df = extractDF(
                dataPath,
                dfVars=["Vp_RTN", "BFIELDRTN", "BTOTAL", "R"],
                info=False,
            )

            self.df = self.df[datetime(2019, 11, 1) : datetime(2019, 11, 30)]

            for col in self.df.columns:
                self.df[col][self.df[col] < -10000] = np.nan

            self.df["V_R"] = self.df["Vp_RTN"]
            del self.df["Vp_RTN"]

            # Rename Btotal and Vp
            self.df["B_TOTAL"] = self.df["BTOTAL"]
            del self.df["BTOTAL"]

            self.df["Time"] = self.df.index
            self.df.to_csv(df_sta_csv_path, index=False)
            del self.df["Time"]
        elif self.name == "Earth_April_2020":
            # Get the WIND data as a complete DF and process errors here?
            from heliopy.data import wind

            startTime = datetime(2020, 4, 15)
            endTime = datetime(2020, 4, 24)
            magdf = wind.mfi_h0(starttime=startTime, endtime=endTime).to_dataframe()
            swedf = wind.swe_h1(starttime=startTime, endtime=endTime).to_dataframe()

            magdf = magdf[["BGSE_0", "BGSE_1", "BGSE_2"]]
            swedf = swedf[
                [
                    "Proton_Np_moment",
                    "Proton_V_moment",
                    "Proton_W_moment",
                    "xgse",
                    "ygse",
                    "zgse",
                ]
            ]

            self.df = pd.concat([swedf, magdf], join="outer", axis=1).fillna(np.nan)
            self.df.set_axis(
                ["N V_R T X_GSE Y_GSE Z_GSE B_GSE_0 B_GSE_1 B_GSE_2".split()],
                axis=1,
                inplace=True,
            )

            # raise ValueError("Not continuing yet. Debug!")
            self.__gse_to_rtn()
        elif self.name == "ISSI_PSP_e1":
            # Load up the ISSI data
            # dataFolder only used in setup
            dataFolder = f"/Users/ddp/Documents/PhD/solo_sdo/ISSIwork/data/"
            df_is = pd.read_csv(f"{dataFolder}small_ch_in_situ.csv")
            df_is.index = pd.to_datetime(df_is["Time"])
            del df_is["Time"]
            self.df = df_is.resample(f"{self.obj_cad}s").mean()

        elif self.name == "ISSI_HMI_e1":
            import idlsave

            dataFolder = "/Users/ddp/Documents/PhD/solo_sdo/ISSIwork/data/"
            ch_flux = idlsave.read(f"{dataFolder}chflux.sav", verbose=False)
            # Open and Bright point flux
            flux_time = ch_flux.hmitimes.copy()
            flux_time = [t.decode() for t in list(flux_time)]

            df_flux = pd.DataFrame(
                {
                    "ch_open_flux": ch_flux.chofluxes,
                    "ch_bpoint_flux": ch_flux.chbpfluxes,
                },
                index=pd.to_datetime(flux_time, format="%Y.%m.%d_%H:%M:%S_TAI"),
            )

            self.df = df_flux

        elif self.name == "ISSI_AIA_e1":
            # Remote data for ISSI case (short)
            dataFolder = f"/Users/ddp/Documents/PhD/solo_sdo/ISSIwork/data/"
            try:
                df_171 = pd.read_csv(
                    f"{dataFolder}small_ch_171_lc_in.csv", index_col="Time"
                )
                df_193 = pd.read_csv(
                    f"{dataFolder}small_ch_193_lc_in.csv", index_col="Time"
                )
                print("Loaded csv successfully")

                for _df in (df_171, df_193):
                    _df.index = pd.to_datetime(_df.index)
            except FileNotFoundError:
                raise FileNotFoundError(f"Unable to open Dataframes in {dataFolder}")

            self.df171 = df_171
            self.df193 = df_193
            self.df = "Use df_171, df_193, out of object"

        else:
            raise NotImplementedError(f"{self.name} not implemented")

        # When we need to combine plasmadf and magdf
        if self.df is None:

            try:
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

            # Ignore if plasmadf is missing
            except AttributeError:
                pass

            # MAG IS GUARANTEED TO EXIST
            #############################################################
            # Magnetic field measurements
            self.magdf.index = pd.to_datetime(self.magdf["Time"])
            del self.magdf["Time"]

            # For PSP data cleaning
            if "fld_flag" in self.magdf.columns:
                if self.name == "PSP_Nov_2019":
                    # Special handling of Nov 2019 data due to different error code
                    self.magdf[self.magdf["fld_flag"] > 128] = np.nan
                else:
                    self.magdf[self.magdf["fld_flag"] >= 1] = np.nan
                del self.magdf["fld_flag"]

            # For SolO data cleaning
            if "mag_flag" in self.magdf.columns:
                self.magdf[self.magdf["mag_flag"] < 3] = np.nan
                del self.magdf["mag_flag"]

            self.magdf.fillna(method="pad")
            self.magdf = self.magdf.resample(f"{self.obj_cad}s").mean()

            try:
                df = pd.concat(
                    [self.plasmadf, self.magdf], join="outer", axis=1
                ).fillna(np.nan)

            except AttributeError:
                df = self.magdf

            self.df = df

        else:
            try:
                # After loading it breaks datetime format -> We save as the csv for ease of use
                self.df.index = pd.to_datetime(self.df["Time"])
                del self.df["Time"]
            except KeyError:
                pass

            except TypeError:
                return

        self.df = self.df.resample(f"{self.obj_cad}s").mean()

        # Set the start and end time
        self.start_time = pd.to_datetime(self.df.index[0])
        self.end_time = pd.to_datetime(self.df.index[-1])
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

    def __gse_to_rtn(self):
        """Verified against J.O Kane 2021 paper with same case study"""
        assert (
            "earth" in self.name.lower()
        ), "The GSE to RTN coordinate conversion is only necessary for Earth-based observers."
        from sunpy.coordinates import sun

        times = self.df.index

        X = self.df["B_GSE_0"]
        Y = self.df["B_GSE_1"].values
        Z = self.df["B_GSE_2"].values

        r = -X
        Ps = sun.P(times)
        t, n = [], []

        for P, y, z in zip(Ps.value, Y, Z):
            t.append(np.sin(P) * y[0] + np.cos(P) * z[0])
            n.append(np.cos(P) * z[0] - np.sin(P) * y[0])

        # Radial mag field
        self.df["B_R"] = r
        self.df["B_T"] = t
        self.df["B_N"] = n

        # del self.df["B_GSE_0"]
        # del self.df["B_GSE_1"]
        # del self.df["B_GSE_2"]

        return None

    def saveWithRadius(self):
        """
        Save a new csv with the radius information
        """
        try:
            self.extract_orbit_data(from_data=True, stepMinutes=self.obj_cad / 60)
            self.df = self.df[:-1]
            self.df["R"] = (self.sp_traj.r.to(u.km)).value

        except NotImplementedError:
            self.df["R"] = gse_calculate_rad(
                [
                    self.sunEarthDist - self.df["X_GSE"].values,
                    self.df["Y_GSE"].values,
                    self.df["Z_GSE"].values,
                ]
            )
        self.df["Time"] = self.df.index
        self.df.to_csv(
            self.completeCSVPath,
            index=False,
        )

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

        elif "earth" in self.name.lower():
            spicedata.get_kernel("planet_trajectories")
            sp_traj = spice.Trajectory("Earth")
        elif "sta" in self.name.lower():
            spicedata.get_kernel("stereo_a")
            sp_traj = spice.Trajectory("Stereo Ahead")
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
        sp_traj.generate_positions(times, "Sun", "ECLIPJ2000")
        sp_traj.change_units(u.au)

        self.sp_traj = deepcopy(sp_traj)

    def plotOrbit_x_y(
        farFromSun,
        closetoSun,
        legendLoc,
        objFolder=f"{BASE_PATH}Figures/Orbit_3d/",
        plotRate="12h",
        farTime="00:00",
        closeTime="00:00",
        pspiralHlight=None,
        radialTolerance=1.5,
        zoomRegion=((-0.79, 0.07), (-0.16, 0.8)),
        vSW=314,
        selfName="MISSING_NAME_SELF",
        otherName="MISSING_NAME_OTHER",
        selfSpiralRadii=(0.15, 1.3),
        otherSpiralRadii=(0.05, 1.3),
        plot_spirals=True,
        plotZ=False,
        zoomRegionZ=None,
        zLabelLoc="best",
    ):
        """
        Plots the longitude and latitude of a given spacecraft object
        Defaults are set to PSP SolO case
        plotRate = How often to plot the orbits
        hlight = for farFrom, closeTo, which date to highlight (closest to!)
        """
        plt.rc("legend", fontsize="x-large")  # using a named size
        makedirs(objFolder, exist_ok=True)
        assert farFromSun.sp_traj != None, "Please calculate the spacecraft trajectory"
        t1 = farFromSun.sp_traj.times.datetime
        X1, Y1, Z1 = (farFromSun.sp_traj.x, farFromSun.sp_traj.y, farFromSun.sp_traj.z)
        rad1, lat1, lon1 = (
            farFromSun.sp_traj.coords.distance,
            farFromSun.sp_traj.coords.lat,
            farFromSun.sp_traj.coords.lon,
        )

        t2 = closetoSun.sp_traj.times.datetime
        X2, Y2, Z2 = (closetoSun.sp_traj.x, closetoSun.sp_traj.y, closetoSun.sp_traj.z)
        rad2, lat2, lon2 = (
            closetoSun.sp_traj.coords.distance,
            closetoSun.sp_traj.coords.lat,
            closetoSun.sp_traj.coords.lon,
        )
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

        df_farFromSun = df_farFromSun.resample(f"{plotRate}", origin=farTime).mean()
        df_nextToSun = df_nextToSun.resample(f"{plotRate}", origin=closeTime).mean()

        MAXTICKS = round(
            (df_farFromSun.index[-1] - df_farFromSun.index[0]).total_seconds()
            / (24 * 3600)
        )

        farFromSunTimesList, tPSPList, lonDistList = [], [], []
        lonDistDF = pd.DataFrame({})

        # Get times and longitudes into a list
        for _i, dfRowPSP in enumerate(df_nextToSun.iterrows()):
            # Use the downsampled PSP data
            tPSP, tSolo, lonDist = _alignmentIdentification(
                soloDF=df_farFromSun, pspDF=dfRowPSP, vSWPSP=vSW
            )

            farFromSunTimesList.append(tSolo)
            tPSPList.append(tPSP)
            lonDistList.append(lonDist)

        lonDistDF["PSPTime"] = tPSPList
        lonDistDF["farFromSunTime"] = farFromSunTimesList
        lonDistDF["LonDist"] = lonDistList

        closestRows = lonDistDF[np.abs(lonDistDF["LonDist"]) <= radialTolerance]

        # Find radial alignment
        slopeIntList = []
        slopeInt = namedtuple("slopeInt", ["m", "b"])
        P1 = (0, 0)

        for r, X1 in enumerate(df_nextToSun["X"].values):
            P2 = (df_nextToSun["X"].values[r], df_nextToSun["Y"].values[r])
            m, b = lineCalc(P1, P2)
            slopeIntList.append(slopeInt(m, b))

        soloIndices = []
        for _t in pd.to_datetime(closestRows["farFromSunTime"].values).to_pydatetime():
            soloIndex = df_farFromSun.index.get_loc(_t, method="nearest")
            soloIndices.append(soloIndex)

        hlightIndices = {
            "farFromSunHighlightedIndices": soloIndices,
            "PSPIndices": list(
                set(
                    [
                        *closestRows.index.to_list(),
                    ]
                )
            ),
        }

        # ANNOTATION
        # Next to Sun
        pspiralIndex = (
            df_nextToSun.index.get_loc(pspiralHlight, method="nearest")
            if pspiralHlight
            else None
        )

        def plot_orbit_z():
            """
            Take the measurements from next to Sun and far from Sun
            and plot on the X/Z direction
            """
            fig = plt.figure(figsize=(11, 10), constrained_layout=True)
            ax = plt.gca()
            plt.scatter([0], [0], s=300, color="orange", marker="*")

            smap = plt.scatter(
                df_farFromSun["Y"],
                df_farFromSun["Z"],
                c=mdates.date2num(df_farFromSun.index),
                cmap="binary",
                s=250,
            )
            cax = ax.inset_axes([1.07, 0.50, 0.05, 0.45])
            cb = plt.colorbar(smap, cax=cax, orientation="vertical", shrink=0.45)
            cb.ax.yaxis.set_label_position("left")
            cb.ax.set_ylabel(f"{selfName}")
            loc = mdates.AutoDateLocator(minticks=3, maxticks=MAXTICKS)
            form = mdates.ConciseDateFormatter(loc)
            cb.ax.yaxis.set_major_formatter(form)

            # Plot the PSP orbit
            smap2 = plt.scatter(
                df_nextToSun["Y"],
                df_nextToSun["Z"],
                c=mdates.date2num(df_nextToSun.index),
                cmap="OrRd",
                s=250,
            )
            cax2 = ax.inset_axes([1.07, 0, 0.05, 0.45])
            cb2 = plt.colorbar(smap2, cax=cax2, orientation="vertical", shrink=0.45)
            cb2.ax.yaxis.set_label_position("left")
            cb2.ax.set_ylabel(f"{otherName}")
            loc2 = mdates.AutoDateLocator(minticks=3, maxticks=MAXTICKS)
            form2 = mdates.ConciseDateFormatter(loc2)
            cb2.ax.yaxis.set_major_formatter(form2)

            # Annotate dates in closetoSun
            plt.gca().set_prop_cycle(None)
            for i, txt in enumerate(df_nextToSun.index):
                # If within the list
                if i in list(set([*hlightIndices["PSPIndices"], pspiralIndex])):
                    _lab = f'{otherName} {datetime.strftime(txt, "%Y-%m-%d %H:%M")}'
                    plt.scatter(
                        df_nextToSun["Y"][i],
                        df_nextToSun["Z"][i],
                        marker="d",
                        # color="red" if i != pspiralIndex else "green",
                        s=125,
                        alpha=1,
                        label=_lab
                        if i != pspiralIndex
                        else f"Parker Alignment: {_lab}",
                    )

            plt.gca().set_prop_cycle(None)

            # Annotate dates in farFromSun (Which are highlighted)
            # Keep in red because we only use radial propagation here, not Parker Spiral???
            for i, txt in enumerate(df_farFromSun.index):
                if i in hlightIndices["farFromSunHighlightedIndices"]:
                    _lab = f'{selfName}: {datetime.strftime(txt, "%Y-%m-%d %H:%M")}'
                    plt.scatter(
                        df_farFromSun["Y"][i],
                        df_farFromSun["Z"][i],
                        # color="red",
                        s=125,
                        label=_lab,
                        marker="X",
                    )

            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.xlabel("Cartesian Y (AU)")
            plt.ylabel("Cartesian Z (AU)")

            plt.xlim(*zoomRegionZ[0])
            plt.ylim(*zoomRegionZ[1])

            plt.legend(loc=zLabelLoc)

            plt.savefig(
                f"{objFolder}Z-Axis_rate_{plotRate}_Tolerance_{radialTolerance}deg.pdf"
            )
            print("Saved Z figure in {}".format(objFolder))
            if farFromSun.show:
                plt.show()

            plt.close()

        if plotZ and zoomRegionZ != None:
            plot_orbit_z()

        # Figure of the two orbits with field lines
        fig = plt.figure(figsize=(11, 10), constrained_layout=True)
        ax = plt.gca()
        plt.scatter([0], [0], s=300, color="orange", marker="*")

        # RADIAL
        # plot radial
        for i, line in enumerate(slopeIntList):
            # Take plus or minus X values
            x = (
                np.linspace(0.0, -1.0)
                if df_nextToSun["X"].values[i] < 0
                else np.linspace(0.0, 1)
            )
            y = line.m * x + line.b

            tline = df_nextToSun.index[i]
            _col = (
                "orange"
                if tline not in df_nextToSun.index[hlightIndices["PSPIndices"]]
                else "red"
            )
            plt.plot(x, y, color=_col, alpha=0.4, linestyle="--", linewidth=1, zorder=1)

        if plot_spirals:
            # SPIRALS
            # plot spiral
            closeToSpirals = findSpirals(
                df=df_nextToSun, vSW=vSW, rads=otherSpiralRadii
            )
            for i, spiral in enumerate(closeToSpirals):
                x = spiral.X
                y = spiral.Y
                _col, _alpha = ("blue", 0.1) if i != pspiralIndex else ("green", 0.8)
                plt.plot(x, y, color=_col, alpha=_alpha, linestyle="-.", linewidth=4)

            # Plot Solo Spiral
            farFromSpirals = findSpirals(df_farFromSun, vSW=vSW, rads=selfSpiralRadii)
            for spiral in farFromSpirals:
                x = spiral.X
                y = spiral.Y
                plt.plot(x, y, color="gray", alpha=0.1, linestyle="-.")

        # Plot all Spacecraft positions
        smap = plt.scatter(
            df_farFromSun["X"],
            df_farFromSun["Y"],
            c=mdates.date2num(df_farFromSun.index),
            cmap="binary",
            s=250,
        )

        cax = ax.inset_axes([1.07, 0.50, 0.05, 0.45])
        cb = plt.colorbar(smap, cax=cax, orientation="vertical", shrink=0.45)
        cb.ax.yaxis.set_label_position("left")
        cb.ax.set_ylabel(f"{selfName}")

        loc = mdates.AutoDateLocator(minticks=3, maxticks=MAXTICKS)
        cb.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

        smap2 = plt.scatter(
            df_nextToSun["X"],
            df_nextToSun["Y"],
            c=mdates.date2num(df_nextToSun.index),
            cmap="OrRd",
            s=250,
        )
        cax2 = ax.inset_axes([1.07, 0, 0.05, 0.45])
        cb2 = plt.colorbar(smap2, cax=cax2, orientation="vertical", shrink=0.45)
        cb2.ax.yaxis.set_label_position("left")
        cb2.ax.set_ylabel(f"{otherName}")
        loc2 = mdates.AutoDateLocator(minticks=3, maxticks=MAXTICKS)
        # cb2.ax.yaxis.set_major_locator(loc)
        cb2.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc2))

        # Annotate dates in closetoSun
        plt.gca().set_prop_cycle(None)
        for i, txt in enumerate(df_nextToSun.index):
            # If within the list
            if i in list(set([*hlightIndices["PSPIndices"], pspiralIndex])):
                _lab = f'{otherName} {datetime.strftime(txt, "%Y-%m-%d %H:%M")}'
                plt.scatter(
                    df_nextToSun["X"][i],
                    df_nextToSun["Y"][i],
                    marker="d",
                    # color="red" if i != pspiralIndex else "green",
                    s=125,
                    alpha=1,
                    label=_lab if i != pspiralIndex else f"Parker Alignment: {_lab}",
                )

        plt.gca().set_prop_cycle(None)

        # Annotate dates in farFromSun (Which are highlighted)
        # Keep in red because we only use radial propagation here, not Parker Spiral???
        for i, txt in enumerate(df_farFromSun.index):
            if i in hlightIndices["farFromSunHighlightedIndices"]:
                _lab = f'{selfName}: {datetime.strftime(txt, "%Y-%m-%d %H:%M")}'
                plt.scatter(
                    df_farFromSun["X"][i],
                    df_farFromSun["Y"][i],
                    # color="red",
                    s=125,
                    label=_lab,
                    marker="X",
                )

        # Need to add labels correctly
        plt.grid(True)
        plt.xlabel("Cartesian X (AU)")
        plt.ylabel("Cartesian Y (AU)")
        plt.title(
            f"Accepted long. separation:{radialTolerance:.2f} deg. | cadence {plotRate} | Vsw: {vSW} km/s"
        )
        plt.xlim(*zoomRegion[0])
        plt.ylim(*zoomRegion[1])
        plt.legend(loc=legendLoc)

        # Plot the Parker Spiral lines
        plt.savefig(f"{objFolder}rate_{plotRate}_Tolerance_{radialTolerance}deg.pdf")
        print(f"Saved to {objFolder}rate_{plotRate}_Tolerance_{radialTolerance}deg.pdf")
        if farFromSun.show:
            plt.show()

        plt.close()

    def zoom_in(
        self,
        start_time: datetime,
        end_time: datetime,
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
                self.sp_traj.times.datetime[0] : self.sp_traj.times.datetime[-1]
            ]

        # Update the dfUnits
        if self.dfUnits is not None:
            try:
                self.df.index.get_loc(self.end_time)
                endIndex = self.end_time
            except KeyError:
                endIndex = self.df.index[-1]

            self.dfUnits = self.dfUnits[
                self.df.index.get_loc(
                    self.start_time, method="nearest"
                ) : self.df.index.get_loc(endIndex, method="nearest")
                + 1
            ]

        # Cut down the dataframe and maintain higher cadence
        self.df = self.df[self.start_time : self.end_time]


class PSPSolO_e6(Spacecraft):
    def __init__(self, name="NONE", cadence_obj=None, show=False):
        super().__init__(name=name, cadence_obj=cadence_obj, show=show)

    def plot_solo_psp_df_onlyScaled(
        self,
        other,
        zones=[],
        saveScaledDF=False,
        case=None,
        colourSolO="k-",
        colourPSP="r-",
    ):
        """
        Plot the dataframe in the style of the previous paper
        Other must be PSP
        zone 1 and zone 2 will be highlighted in SolO and PSP respectively
        """
        assert "solo" in self.name.lower(), "Please ensure SolO is object with f call"

        # Figure
        # Width and marker size
        _, axs = plt.subplots(
            5, 1, figsize=(16, 10), sharex=True, constrained_layout=True
        )

        R = (self.dfUnits["R"].to(u.m) - const.R_sun).value
        oR = (other.dfUnits["R"].to(u.m) - const.R_sun).value

        ts = self.df.index
        ots = other.df.index

        Bt = np.sqrt(
            self.dfUnits["B_R"] ** 2
            + self.dfUnits["B_T"] ** 2
            + self.dfUnits["B_N"] ** 2
        )
        BrScaled = (self.dfUnits["B_R"].to(u.T)) * R ** 2
        BtScaled = np.sqrt(
            BrScaled ** 2
            + self.dfUnits["B_T"].to(u.T) ** 2
            + self.dfUnits["B_N"].to(u.T) ** 2
        ).to(u.nT)

        Vx = np.abs(self.dfUnits["V_R"])
        Np = self.dfUnits["N"]  # In protons per cm3
        NpScaled = (Np.to(u.m ** (-3)) * R ** 2).to(u.cm ** (-3))  # To 1/m**3

        NpRPW = self.dfUnits["N_RPW"]
        NpRPWScaled = (NpRPW.to(u.m ** (-3)) * R ** 2).to(u.cm ** (-3))
        T = self.df["T"]

        # m_p is in kg
        Mf = (
            const.m_p
            * NpRPWScaled.to(u.m ** (-3))
            * np.abs(self.dfUnits["V_R"].to(u.m / u.s))
        )
        MfScaled = Mf

        # Other object (PSP)
        oBTUnscaled = np.sqrt(
            other.df["B_R"] ** 2 + other.df["B_T"] ** 2 + other.df["B_N"] ** 2
        )
        oBrScaled = other.dfUnits["B_R"].to(u.T) * oR ** 2
        oBtScaled = np.sqrt(
            oBrScaled ** 2
            + other.dfUnits["B_T"].to(u.T) ** 2
            + other.dfUnits["B_N"].to(u.T) ** 2
        ).to(u.nT)
        oVx = np.abs(other.dfUnits["V_R"])
        oNp = other.dfUnits["N"]
        oNpScaled = (oNp.to(u.m ** (-3)) * oR ** 2).to(u.cm ** (-3))
        oT = other.df["T"]
        oMf = const.m_p * oNpScaled.to(u.m ** (-3)) * np.abs(oVx).to(u.m / u.s)
        oMfScaled = oMf

        selfScaledDF = pd.DataFrame(
            {
                "V_R": Vx,
                "B_R": BrScaled,
                "Btotal": BtScaled,
                "N": NpScaled,
                "N_RPW": NpRPWScaled,
                "T": T,
                "Mf": MfScaled,
            },
        )

        otherScaledDF = pd.DataFrame(
            {
                "V_R": oVx,
                "B_R": oBrScaled,
                "Btotal": oBtScaled,
                "N": oNpScaled,
                "T": oT,
                "Mf": oMfScaled,
            },
        )

        if saveScaledDF:
            # Cases are used to save the dataframe to a csv
            if case == "orbit6":
                selfScaledDF.to_csv(f"{BASE_PATH}Data/Prepped_csv/SolO_POST_e6.csv")
                otherScaledDF.to_csv(f"{BASE_PATH}Data/Prepped_csv/psp_POST_e6.csv")

        # BEGIN PLOTS
        # Magnetic field components
        # Radial velocity
        ax1 = axs[0]
        ax1.plot(ts, Vx, colourSolO, label="SolO")
        ax1.plot(ots, oVx, colourPSP, label="PSP")
        ax1.set_ylabel(r"$-{V_x} [km/s]$")
        ax1.legend(loc="upper left")

        # Instead of Temperature show scaled Br
        ax2 = axs[1]
        ax2.plot(
            ts,
            BrScaled,
            colourSolO,
            label="SolO Br",
            linewidth=1,
        )
        ax2.plot(ots, oBrScaled, colourPSP, label="PSP Br")
        ax2.set_ylabel(r"$\hat{B}_R$[nT]")
        # ax2.legend(loc="upper left")

        # Magnetic field components
        axsBt = axs[2]
        axsBt.set_ylabel(r"S.$\hat{B}_{TOTAL}$[nT]")
        axsBt.plot(
            ts,
            BtScaled,
            colourSolO,
            label=r"S.Btotal SolO",
        )
        axsBt.plot(
            ots,
            oBtScaled,
            colourPSP,
            label=r"S.Btotal PSP",
        )
        # axsBt.legend(loc="upper left")

        # Proton Density
        axsNp = axs[3]
        axsNp.plot(
            ts,
            NpScaled,
            colourSolO,
            label=r"SolO[PAS]",
        )
        axsNp.plot(
            ts,
            NpRPWScaled,
            colourSolO,
            alpha=0.5,
            label=r"SolO[RPW]",
        )
        axsNp.plot(
            ots,
            oNpScaled,
            colourPSP,
            label=r"PSP",
        )
        axsNp.set_ylabel(r"S.$N_p$ (# $m^{-3}$)")
        axsNp.legend(loc="upper left")

        # Scaled MF
        axsMf = axs[4]
        axsMf.plot(ts, MfScaled, colourSolO, linewidth=1, label=r"S.Mf SolO")
        axsMf.plot(ots, oMfScaled, colourPSP, label=r"S.Mf PSP")
        axsMf.set_ylabel(r"S.$Mf$ [kg m$^{-2}$ s$^{-1}$]")
        # axsMf.legend(loc="upper left")

        # # Plot the relevant columns
        for ax in axs:
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
            plotPath = f"{BASE_PATH}Figures/PSP_SolO/"
            makedirs(plotPath, exist_ok=True)
            plt.savefig(f"{plotPath}onlyScaled_{self.name}__{other.name}.pdf")

            print(
                "Saved to {} as {}".format(
                    plotPath, f"onlyScaled_{self.name}__{other.name}.pdf"
                )
            )

        plt.close()

    def plot_solo_psp_df(
        self,
        other,
        zones=[],
        saveScaledDF=False,
        case=None,
        colourPSP="r-",
        colourSolO="k-",
    ):
        """
        Plot the dataframe in the style of the previous paper
        Other must be PSP
        zone 1 and zone 2 will be highlighted in SolO and PSP respectively
        """
        assert "solo" in self.name.lower(), "Please ensure SolO is object with f call"
        from astropy import constants as c

        # Figure
        # Width and marker size
        _, axs = plt.subplots(
            8, 1, figsize=(16, 2 * 5), sharex=True, constrained_layout=True
        )

        R = (self.dfUnits["R"].to(u.m) - const.R_sun).value
        oR = (other.dfUnits["R"].to(u.m) - const.R_sun).value

        ts = self.df.index
        ots = other.df.index

        Bt = np.sqrt(
            self.dfUnits["B_R"] ** 2
            + self.dfUnits["B_T"] ** 2
            + self.dfUnits["B_N"] ** 2
        )
        BrScaled = (self.dfUnits["B_R"].to(u.T)) * R ** 2
        BtScaled = np.sqrt(
            BrScaled ** 2
            + self.dfUnits["B_T"].to(u.T) ** 2
            + self.dfUnits["B_N"].to(u.T) ** 2
        ).to(u.nT)

        Vx = np.abs(self.dfUnits["V_R"])
        Np = self.dfUnits["N"]  # In protons per cm3
        NpScaled = (Np.to(u.m ** (-3)) * R ** 2).to(u.cm ** (-3))  # To 1/m**3

        NpRPW = self.dfUnits["N_RPW"]
        NpRPWScaled = (NpRPW.to(u.m ** (-3)) * R ** 2).to(u.cm ** (-3))
        T = self.df["T"]

        # m_p is in kg
        Mf = (
            const.m_p
            * NpRPWScaled.to(u.m ** (-3))
            * np.abs(self.dfUnits["V_R"].to(u.m / u.s))
        )
        MfScaled = Mf

        # Other object (PSP)
        oBTUnscaled = np.sqrt(
            other.df["B_R"] ** 2 + other.df["B_T"] ** 2 + other.df["B_N"] ** 2
        )
        oBrScaled = other.dfUnits["B_R"].to(u.T) * oR ** 2
        oBtScaled = np.sqrt(
            oBrScaled ** 2
            + other.dfUnits["B_T"].to(u.T) ** 2
            + other.dfUnits["B_N"].to(u.T) ** 2
        ).to(u.nT)
        oVx = np.abs(other.dfUnits["V_R"])
        oNp = other.dfUnits["N"]
        oNpScaled = (oNp.to(u.m ** (-3)) * oR ** 2).to(u.cm ** (-3))
        oT = other.df["T"]
        oMf = const.m_p.value * oNpScaled.to(u.m ** (-3)) * np.abs(oVx).to(u.m / u.s)
        oMfScaled = oMf

        selfScaledDF = pd.DataFrame(
            {
                "V_R": Vx,
                "B_R": BrScaled,
                "Btotal": BtScaled,
                "N": NpScaled,
                "N_RPW": NpRPWScaled,
                "T": T,
                "Mf": MfScaled,
            },
        )

        otherScaledDF = pd.DataFrame(
            {
                "V_R": oVx,
                "B_R": oBrScaled,
                "Btotal": oBtScaled,
                "N": oNpScaled,
                "T": oT,
                "Mf": oMfScaled,
            },
        )

        if saveScaledDF:
            if case == "orbit6":
                selfScaledDF.to_csv(f"{BASE_PATH}Data/Prepped_csv/SolO_POST_e6.csv")
                otherScaledDF.to_csv(f"{BASE_PATH}Data/Prepped_csv/psp_POST_e6.csv")

        # BEGIN PLOTS
        # Magnetic field components
        ax0 = axs[0]
        ax0.set_ylabel(r"$\hat{B}_T(nT)$")
        ax0.semilogy(
            ts,
            Bt,
            colourSolO,
            label="SolO_MAG",
            alpha=1,
            linewidth=1,
        )
        ax0.semilogy(
            ots,
            oBTUnscaled,
            colourPSP,
            label="PSP_FLD",
            alpha=1,
        )
        ax0.legend()

        # Radial velocity
        ax1 = axs[1]
        ax1.plot(ts, Vx, colourSolO, label="Vp [GSE]", linewidth=1)
        ax1.plot(ots, oVx, colourPSP)
        ax1.set_ylabel(r"$-{V_x} (km/s)$")

        # Temperature
        ax2 = axs[2]
        ax2.semilogy(
            ts,
            T,
            colourSolO,
            label=r"$T_p$",
            linewidth=1,
        )
        ax2.semilogy(ots, oT, colourPSP)
        ax2.set_ylabel(r"$T_p (K)$")

        # Proton Density
        ax4 = axs[3]
        ax4.semilogy(
            ts,
            Np,
            colourSolO,
            label=r"Np (SolO_PAS)",
            linewidth=1,
        )
        ax4.semilogy(ts, NpRPW, colourPSP, alpha=1, label=r"Np (SolO_RPW)")
        ax4.semilogy(ots, oNp, "k-", label=r"Np (PSP_PAS)")
        ax4.set_ylabel(r"$n_p$ (# $cm^{-3}$)")
        ax4.legend()

        # Pressure
        axMf = axs[4]
        axMf.semilogy(
            ts,
            Mf,
            colourSolO,
            label="Mass Flux",
            linewidth=1,
        )
        axMf.semilogy(
            ots,
            oMf,
            colourPSP,
        )
        axMf.set_ylabel(r"$Mf$")

        # Magnetic field components
        axs[5].set_ylabel(r"$\hat{B}_Total$")
        axs[5].plot(
            ts,
            BtScaled,
            colourSolO,
            label=r"$R^2$ Scaled Btotal SolO",
        )
        axs[5].plot(
            ots,
            oBtScaled,
            colourPSP,
            label=r"$R^2$ Scaled Btotal PSP",
        )
        axs[5].legend()

        # Proton Density
        axs[6].plot(
            ts,
            NpScaled,
            colourSolO,
            label=r"$R^2$ Scaled Np (SolO_PAS)",
        )
        axs[6].plot(
            ts,
            NpRPWScaled,
            colourSolO,
            alpha=0.5,
            label=r"$R^2$ Scaled Np (SolO_RPW)",
        )
        axs[6].plot(
            ots,
            oNpScaled,
            colourPSP,
            label=r"$R^2$ Scaled Np (PSP_SWEAP)",
        )
        axs[6].set_ylabel(r"$n_p$ (# $m^{-3}$)")
        axs[6].legend()

        # Scaled MF
        axs[7].plot(
            ts, MfScaled, colourSolO, linewidth=1, label=r"$R^2$ Scaled Mf SolO"
        )
        axs[7].plot(ots, oMfScaled, colourPSP, label=r"$R^2$ Scaled Mf PSP")
        axs[7].set_ylabel(r"$Mf$")
        axs[7].legend()

        # # Plot the relevant columns
        for ax in axs:
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
            plt.savefig(f"{plotPath}summaryPlot_{self.name}__{other.name}.pdf")

            print(
                "Saved to {} as {}".format(
                    plotPath, f"summaryPlot_{self.name}__{other.name}.pdf"
                )
            )

        plt.close()


class STA_psp(Spacecraft):
    def __init__(
        self,
        name="NONE",
        cadence_obj=None,
        show=False,
        sunEarthDist=150111200.76,
        remakeCSV=False,
    ):
        super().__init__(
            name=name,
            cadence_obj=cadence_obj,
            show=show,
            sunEarthDist=sunEarthDist,
            remakeCSV=remakeCSV,
        )

    def plot_sta_psp_df(self, other, zones=[], colourSTA="k-", colourPSP="r-"):
        assert "sta" in self.name.lower(), "Please ensure STEREO is object with f call"

        ts = self.df.index
        ots = other.df.index

        R = (self.dfUnits["R"].to(u.m) - const.R_sun).value
        oR = (other.dfUnits["R"].to(u.m) - const.R_sun).value

        Bx, By, Bz = self.df["B_R"], self.df["B_T"], self.df["B_N"]
        oBx, oBy, oBz = other.df["B_R"], other.df["B_T"], other.df["B_N"]
        Bt = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
        oBt = np.sqrt(oBx ** 2 + oBy ** 2 + oBz ** 2)

        # Figure
        # Width and marker size
        _, axs = plt.subplots(
            5, 1, figsize=(14, 2 * 5), sharex=True, constrained_layout=True
        )

        # V
        axV = axs[0]
        axV.set_ylabel(r"$\hat{V}_R [km/s]$")
        axV.plot(ts, self.df["V_R"], colourSTA, alpha=0.7, label="ST-A")
        axV.legend()

        # Bx
        axBR = axs[1]
        axBR.set_ylabel(r"$\hat{B}_{R} [nT]$")
        axBR.plot(ts, Bx, colourSTA, alpha=0.7, label="ST-A")
        axBR.plot(ots, oBx, colourPSP, alpha=0.6, label="PSP [+3hr]")
        axBR.legend()

        # By
        axBT = axs[2]
        axBT.set_ylabel(r"$\hat{B}_{T} [nT]$")
        axBT.plot(ts, By, colourSTA, alpha=0.7)
        axBT.plot(ots, oBy, colourPSP, alpha=0.6)

        # Bz
        axBN = axs[3]
        axBN.set_ylabel(r"$\hat{B}_{N} [nT]$")
        axBN.plot(ts, Bz, colourSTA, alpha=0.7)
        axBN.plot(ots, oBz, colourPSP, alpha=0.6)

        # Btotal
        axBTOTAL = axs[4]
        axBTOTAL.set_ylabel(r"$\hat{B}_{TOTAL} [nT]$")
        axBTOTAL.plot(ts, Bt, colourSTA, alpha=0.7)
        axBTOTAL.plot(ots, oBt, colourPSP, alpha=0.6)

        # # Plot the relevant columns
        for ax in axs:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        # Plot the zones being considered
        for zone in zones:
            if zone != None:
                for ax in axs:
                    ax.axvspan(
                        zone["start_time"],
                        zone["end_time"],
                        color=zone["color"],
                        alpha=0.3,
                        ymin=0,
                        ymax=0.15,
                    )

        # total magnetic field strength, proton number density, solarwind bulk flow velocity and mass flux,
        # print(f"Saving to {self}")
        # plt.savefig(f"{save_folder}B_Br_V_sb_matrix_highlighted.pdf")
        if self.show:
            plt.show()
        else:
            plotPath = f"{BASE_PATH}Figures/PSP_STA/"
            makedirs(plotPath, exist_ok=True)
            print(f"Saving to {plotPath}")
            plt.savefig(f"{plotPath}summaryPlot_{self.name}__{other.name}.pdf")

        plt.close()


class EarthApril2020(Spacecraft):
    def __init__(
        self,
        cadence_obj,
        name="NONE",
        show=False,
        sunEarthDist=150111200.76,
        remakeCSV=False,
    ):
        super().__init__(
            name=name,
            cadence_obj=cadence_obj,
            show=show,
            sunEarthDist=sunEarthDist,
            remakeCSV=remakeCSV,
        )

    def plot_solo_earth_df(self, other, zones=[], colourWIND="k-", colourSOLO="r-"):
        assert "earth" in self.name.lower(), "Please ensure Earth is object with f call"

        ts = self.df.index
        ots = other.df.index

        R = (self.dfUnits["R"].to(u.m) - const.R_sun).value
        oR = (other.dfUnits["R"].to(u.m) - const.R_sun).value

        Bx, By, Bz = self.df["B_R"], self.df["B_T"], self.df["B_N"]
        Bgse0, Bgse1, Bgse2 = self.df["B_GSE_0"], self.df["B_GSE_1"], self.df["B_GSE_2"]
        oBx, oBy, oBz = other.df["B_R"], other.df["B_T"], other.df["B_N"]
        # Bt = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
        Bt = np.sqrt(Bgse0 ** 2 + Bgse1 ** 2 + Bgse2 ** 2)
        oBt = np.sqrt(oBx ** 2 + oBy ** 2 + oBz ** 2)

        # Figure
        # Width and marker size
        _, axs = plt.subplots(
            5, 1, figsize=(14, 2 * 5), sharex=True, constrained_layout=True
        )

        # Plots
        # V
        axs[0].set_ylabel(r"$\hat{V}_R$ [km/s]")
        axs[0].plot(ts, self.df["V_R"], colourWIND, label="WIND")
        axs[0].legend()

        axs[1].set_ylabel(r"$\hat{B}_{R}$ [nT]")
        axs[1].plot(ts, Bx, colourWIND, label="WIND")
        axs[1].plot(ots, oBx, colourSOLO, label="SolO")
        axs[1].legend()

        # By
        axs[2].set_ylabel(r"$\hat{B}_{T}$ [nT]")
        axs[2].plot(ts, By, colourWIND)
        axs[2].plot(ots, oBy, colourSOLO)

        # Bz
        axs[3].set_ylabel(r"$\hat{B}_{N}$ [nT]")
        axs[3].plot(ts, Bz, colourWIND)
        axs[3].plot(ots, oBz, colourSOLO)

        # Bt0tal
        axs[4].set_ylabel(r"$\hat{B}_{TOTAL}$ [nT]")
        axs[4].plot(ts, Bt, colourWIND)
        axs[4].plot(ots, oBt, colourSOLO)

        # # Plot the relevant columns
        for ax in axs:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        # Plot the zones being considered
        for zone in zones:
            for ax in axs:
                ax.axvspan(
                    zone["start_time"], zone["end_time"], color=zone["color"], alpha=0.3
                )

        if self.show:
            plt.show()
        else:
            plotPath = f"{BASE_PATH}Figures/SolO_Earth/"
            makedirs(plotPath, exist_ok=True)
            print(f"Saving to {plotPath}")
            plt.savefig(f"{plotPath}summaryPlot_{self.name}__{other.name}.pdf")

        plt.close()


class ISSISpc(Spacecraft):
    def __init__(
        self,
        name="NONE",
        cadence_obj=None,
        show=False,
        sunEarthDist=150111200.76,
        remakeCSV=False,
    ):
        super().__init__(
            name=name,
            cadence_obj=cadence_obj,
            show=show,
            sunEarthDist=sunEarthDist,
            remakeCSV=remakeCSV,
        )

    def plot_issi_psp_e1(self, zones=[]):
        from astropy import constants as const
        from astropy import units as u

        ts = self.df.index
        R = (self.dfUnits["R"].to(u.m) - const.R_sun).value
        Br = self.df["B_R"] * R ** 2
        Mf = self.df["Mf"]
        hour = pd.to_timedelta(ts.hour, unit="H")

        # Figure
        _, axs = plt.subplots(
            6, 1, figsize=(14, 10), sharex=True, constrained_layout=True
        )

        # Plots
        axs[0].set_ylabel(r"R [AU]")
        axs[0].plot(ts, R * u.m.to(u.AU))
        axs[0].grid(True)
        axs[0].set_ylim(0.1, 0.3)

        # Bx
        axs[1].set_ylabel(r"$\hat{B}_{R}$ R$^2$ [nT AU$^2$]")
        axs[1].plot(ts, Br, label="PSP")
        # Horizontal bar at 0
        axs[1].axhline(0, color="black", linestyle="--")

        # V
        axs[2].set_ylabel(r"$\hat{V}_R$ [km/s]")
        axs[2].plot(ts, self.df["V_R"])

        axs[3].set_ylabel("Tp [K]")
        axs[3].plot(ts, self.df["T"])

        axs[4].set_ylabel(r"Np [#]")
        axs[4].plot(ts, self.df["N"])

        axs[5].semilogy(
            ts,
            Mf,
            label="Mass Flux",
            linewidth=1,
        )
        axs[5].set_ylabel("Mf [kg km/s]")

        # # Plot the relevant columns
        for ax in axs:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        if self.show:
            plt.show()
        else:
            plotPath = f"{BASE_PATH}Figures/Timeseries/"
            makedirs(plotPath, exist_ok=True)
            print(f"Saving to {plotPath}")
            plt.savefig(f"{plotPath}summaryPlot_{self.name}.pdf")


if __name__ == "__main__":
    raise ImportError(
        "Please use Scripts/Plots/createCSVsAndOrbits to generate CSV plots."
    )
