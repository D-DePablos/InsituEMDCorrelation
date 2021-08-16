BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"

from sys import path

path.append(f"{BASE_PATH}Scripts/")

# from helpers import resample_and_rename
from os import makedirs
from physicsHelpers import fcl
from signalHelpers import normalize_signal as norm
from EMD.applyEMD import comparePSPtoSolo

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import spherical_to_cartesian, SkyCoord

# To add a cycle to colours
from itertools import cycle

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

        # Flag: Whether to add 'extra' measurements to dataframe
        add_all_heliopy_vars = True
        self.sp_coords_carrington = None

        # Time information for specific spacecraft
        if mid_time:
            self.mid_time = mid_time
            self.margin = margin
            self.start_time = mid_time - self.margin
            self.end_time = mid_time + self.margin
        else:
            raise ValueError("Please provid a valid mid_time and time margin")

        # Get the spacecraft data into the proper format
        if self.name == "ST-A":
            from heliopy.data import stereo

            # In this ocassion magplasma L2 has both of the useful datasets
            self.data = stereo.magplasma_l2("sta", self.start_time, self.end_time)
            self.cadence = 60

            # Output of self.data.columns
            """
            'BFIELDRTN_0', 'BFIELDRTN_1','BFIELDRTN_2', 'BTOTAL',
            'HAE_0', 'HAE_1', 'HAE_2',
            'HEE_0', 'HEE_1', 'HEE_2',
            'HEEQ_0','HEEQ_1','HEEQ_2',
            'CARR_0','CARR_1','CARR_2',
            'RTN_0', 'RTN_1', 'RTN_2',
            'R',  'Np', 'Vp',  'Tp', 'Vth',
            'Vr_Over_V_RTN', 'Vt_Over_V_RTN', 'Vn_Over_V_RTN',
            'Vp_RTN',
            'Entropy', 'Beta', 'Total_Pressure',
            'Cone_Angle', 'Clock_Angle',
            'Magnetic_Pressure',
            'Dynamic_Pressure'
            """
            self.rel_vars = {
                "density": "Np",
                "temperature": "Tp",
                "velocity": "Vp",
                "btotal": "BTOTAL",
            }

            # Can set the dataframe empty now
            self.df = pd.DataFrame({})

        elif self.name == "WIND":
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

        elif self.name == "SOLOpub":
            from heliopy.data import solo

            # Public solar Orbiter data TODO
            mag = solo.download(
                self.start_time, self.end_time, descriptor="MAG", level="LL02"
            )

            print(mag.columns)

        elif self.name == "SolOpriv_e6":

            # Not using Heliopy vars
            add_all_heliopy_vars = False

            # Load saved file, otherwise generate
            try:
                swedf = pd.read_csv(f"{BASE_PATH}Data/Prepped_csv/solo_swe_sept.csv")
                magdf = pd.read_csv(f"{BASE_PATH}Data/Prepped_csv/solo_mag_sept.csv")

                self.swedf = swedf
                self.magdf = magdf
                self.df = None

            except FileNotFoundError:
                import cdflib
                from glob import glob

                cdf_epoch = cdflib.cdfepoch()

                def SWE_prep():
                    # SWEAP data
                    cdf_path = f"{BASE_PATH}unsafe/Resources/Solo_Data/L2/GroundMom/"

                    """ 
                    Variables within the cdf:
                        [
                            'Epoch', 
                            'Half_interval', 
                            'SCET', 
                            'Info', 
                            'validity', 
                            'N', 
                            'V_SRF', 
                            'V_RTN', 
                            'P_SRF', 
                            'P_RTN', 
                            'TxTyTz_SRF', 
                            'TxTyTz_RTN', 
                            'T'
                        ] 
                    """
                    # Need to separate into different relevant csv
                    for day, file in enumerate(sorted(glob(f"{cdf_path}*.cdf"))):
                        cdf = cdflib.CDF(file)
                        time = cdf_epoch.to_datetime(cdf["Epoch"])
                        _df = pd.DataFrame({}, index=time)

                        for i in ("V_RTN", "N", "T", "validity"):
                            if i == "V_RTN":  # Allow for multidimensional
                                for n, arg in zip(
                                    (0, 1, 2), ("_R", "_T", "_N")
                                ):  # R is radial velocity
                                    _df[f"V{arg}"] = cdf[i][:, n]

                            # Treat solo swe flag differently
                            elif i == "validity":
                                _df["solo_plasma_flag"] = cdf[i]
                            else:
                                _df[i] = cdf[i]

                        # Join the dataframes only after the first instance
                        if day == 0:
                            _swe_df = _df
                        else:
                            _swe_df = _swe_df.append(_df)

                    _swe_df["Time"] = _swe_df.index
                    return _swe_df

                def MAG_prep():
                    # SWEAP data
                    cdf_path = f"{BASE_PATH}unsafe/Resources/Solo_Data/L2/Mag/"

                    """ 
                    Variables within the cdf:
                        [

                            'EPOCH'
                            'B_RTN'
                            'LBL1_B_RTN'
                            'REP1_B_RTN'
                            'SCET'
                            'B_SRF'
                            'LBL1_B_SRF'
                            'REP1_B_SRF'
                            'QUALITY_FLAG'
                            'PRIMARY_SENSOR_FLAG'
                            'VECTOR_RANGE'
                            
                        ] 
                    """
                    # Need to separate into different relevant csv
                    for day, file in enumerate(sorted(glob(f"{cdf_path}*.cdf"))):
                        cdf = cdflib.CDF(file)
                        time = cdf_epoch.to_datetime(cdf["EPOCH"])
                        _df = pd.DataFrame({}, index=time)

                        for i in ("B_RTN", "QUALITY_FLAG"):
                            if i == "B_RTN":  # Allow for multidimensional
                                for n, arg in zip(
                                    (0, 1, 2), ("_R", "_T", "_N")
                                ):  # R is radial velocity
                                    _df[f"B{arg}"] = cdf[i][:, n]
                            else:
                                _df["MAG_FLAG"] = cdf[i]

                        # Join the dataframes only after the first instance
                        if day == 0:
                            _mag_df = _df
                        else:
                            _mag_df = _mag_df.append(_df)

                    _mag_df["Time"] = _mag_df.index
                    return _mag_df

                # Slightly different to how we save PSP. why?
                self.swedf = SWE_prep()
                self.magdf = MAG_prep()
                self.df = None

            # SWEAP and MAG csv save at highest cadence
            self.swedf.to_csv(
                f"{BASE_PATH}Data/Prepped_csv/solo_swe_sept.csv",
                index=False,
            )

            self.magdf.to_csv(
                f"{BASE_PATH}Data/Prepped_csv/solo_mag_sept.csv",
                index=False,
            )

        elif self.name == "PSPpriv_e6":
            """
            This is the Encounter 6 data
            """
            import cdflib
            from glob import glob

            cdf_epoch = cdflib.cdfepoch()
            add_all_heliopy_vars = False

            def FLD_prep():
                # Fields data
                try:
                    _fld_df = pd.read_csv(
                        f"{BASE_PATH}Data/Prepped_csv/psp_mag_sept.csv"
                    )

                except FileNotFoundError:
                    # When file not available, derive it
                    cdf_path = f"{BASE_PATH}unsafe/Resources/PSP_Data/FIELDS/"
                    for day, file in enumerate(sorted(glob(f"{cdf_path}*.cdf"))):
                        print(f"Day {day} of Magnetic data")
                        cdf = cdflib.CDF(file)
                        time = cdf_epoch.to_datetime(cdf["epoch_mag_RTN_1min"])
                        mag = cdf["psp_fld_l2_mag_RTN_1min"]
                        fldflag = cdf["psp_fld_l2_quality_flags"]
                        # cdf.close()

                        # Dataframe creation and time
                        _df = pd.DataFrame({}, index=time)
                        for i, label in enumerate(cdf["label_RTN"][0]):
                            _df[str(label)] = mag[:, i]

                        _df["fld_flag"] = fldflag

                        # Join the dataframes
                        if day == 0:
                            _fld_df = _df
                        else:
                            _fld_df = _fld_df.append(_df)

                    _fld_df["Time"] = _fld_df.index
                    _fld_df.to_csv(
                        f"{BASE_PATH}Data/Prepped_csv/psp_mag_sept.csv",
                        index=False,
                    )

                return _fld_df

            def SWE_prep():
                """
                This function prepares the SWEAP data for PSP
                """
                try:
                    _swe_df = pd.read_csv(
                        f"{BASE_PATH}Data/Prepped_csv/psp_swe_sept.csv"
                    )

                except FileNotFoundError:
                    cdf_path = f"{BASE_PATH}unsafe/Resources/PSP_Data/SWEAP/"
                    for day, file in enumerate(sorted(glob(f"{cdf_path}*.cdf"))):
                        cdf = cdflib.CDF(file)
                        time = cdf_epoch.to_datetime(cdf["Epoch"])
                        np = cdf["np_moment"]
                        vp = cdf["vp_moment_RTN"]
                        T = cdf["wp_moment"]
                        flag = cdf["general_flag"]
                        cdf.close()

                        _df = pd.DataFrame({}, index=time)
                        for i, label in enumerate(cdf["RTN_FRAME"][0]):
                            _df["V_" + str(label)] = vp[:, i]

                        _df["N"] = np
                        _df["T"] = T
                        _df["swe_flag"] = flag

                        # Join dataframes
                        if day == 0:
                            _swe_df = _df

                        else:
                            _swe_df = _swe_df.append(_df)

                    _swe_df["Time"] = _swe_df.index
                    _swe_df.to_csv(
                        f"{BASE_PATH}Data/Prepped_csv/psp_swe_sept.csv",
                        index=False,
                    )

                return _swe_df

                # raise NotImplementedError("The SWEAP instrument is not implemented yet")

            self.magdf = FLD_prep()
            self.swedf = SWE_prep()
            self.df = None

        else:
            raise NotImplementedError(f"{self.name} not implemented")

        ## After loading measurements, quality flags, do:

        # Add all variables if using Heliopy datasets
        if add_all_heliopy_vars:  # Add more parameters only if add_extra flag is true
            for parameter in self.rel_vars:
                if parameter not in list(self.df):
                    _data = self.data.to_dataframe()[self.rel_vars[parameter]]
                    _data[_data < 0.001] = np.nan  # Does this hold for all parameters?
                    self.df[parameter] = _data

        # When we need to combine swedf and magdf
        if self.df is None:

            # We need to use the Time column
            self.swedf.index = pd.to_datetime(self.swedf["Time"])
            del self.swedf["Time"]

            # For PSP data cleaning
            if "swe_flag" in self.swedf.columns:
                self.swedf[self.swedf["swe_flag"] >= 1] = np.nan
                self.swedf[self.swedf["V_R"] < 0] = np.nan
                self.swedf[self.swedf["V_T"] < -1000] = np.nan
                self.swedf[self.swedf["V_N"] < -1000] = np.nan

                self.swedf[self.swedf["T"] < 0] = np.nan
                self.swedf[self.swedf["N"] < 0] = np.nan

            # For SolO data cleaning
            if "solo_plasma_flag" in self.swedf.columns:
                self.swedf[self.swedf["solo_plasma_flag"] < 3] = np.nan

            self.swedf.fillna(method="pad")
            self.swedf = self.swedf.resample(f"{self.obj_cad}s").mean()

            #############################################################
            # MAG #
            # Magnetic field measurements
            self.magdf.index = pd.to_datetime(self.magdf["Time"])
            del self.magdf["Time"]

            # For PSP data cleaning
            if "fld_flag" in self.magdf.columns:
                self.magdf[self.magdf["fld_flag"] >= 1] = np.nan

            if "MAG_FLAG" in self.magdf.columns:
                self.magdf[self.magdf["MAG_FLAG"] < 3] = np.nan

            self.magdf.fillna(method="pad")
            self.magdf = self.magdf.resample(f"{self.obj_cad}s").mean()

            df = pd.concat([self.swedf, self.magdf], join="outer", axis=1).fillna(
                np.nan
            )

            self.df = df

        else:
            # After loading it breaks datetime format -> We save as the csv for ease of use
            self.df.index = pd.to_datetime(self.df["Time"])
            del self.df["Time"]

            self.df.fillna(method="pad")
            self.df = self.df.resample(f"{self.obj_cad}s").mean()
            # self.df.interpolate(inplace=True)

        self.df.resample(f"{self.obj_cad}s").mean()
        self.df["Time"] = self.df.index
        self.df.to_csv(
            f"{BASE_PATH}Data/Prepped_csv/{self.name}.csv",
            index=False,
        )

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

    def plot_solo_psp_df(self, other, zone1=None, zone2=None):
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
        fig, axs = plt.subplots(
            5, 1, figsize=(16, 2 * 5), sharex=True, constrained_layout=True
        )

        # TODO: Differentiate between SolO and PSP

        Bt = np.sqrt(self.df["B_R"] ** 2 + self.df["B_T"] ** 2 + self.df["B_N"] ** 2)
        Vx = self.df["V_R"]
        Np = self.df["N"]
        T = self.df["T"]
        Mf = c.m_p.value * Np * 10 ** 15 * (-Vx)

        oBt = np.sqrt(
            other.df["B_R"] ** 2 + other.df["B_T"] ** 2 + other.df["B_N"] ** 2
        )
        oVx = other.df["V_R"]
        oNp = other.df["N"]
        oT = other.df["T"]
        oMf = c.m_p.value * oNp * 10 ** 15 * (-oVx)

        # Magnetic field components
        ax0 = axs[0]
        ax0.plot(
            norm(Bt),
            "k-",
            label="Bt",
            alpha=1,
            linewidth=1,
        )
        # ax0.set_ylabel(r"$\vec{B}_{T}$ (nT)")
        ax0.set_ylabel(r"Normalised BTotal")

        if other is not None:
            Bt_PSP = np.sqrt(
                other.df["B_R"] ** 2 + other.df["B_T"] ** 2 + other.df["B_N"] ** 2
            )
            ax0.plot(
                norm(Bt_PSP),
                "r-",
                label="Bt_PSP",
                alpha=1,
            )

        # Radial velocity
        ax1 = axs[1]
        ax1.plot(Vx, color="black", label="Vp [GSE]", linewidth=1)
        ax1.plot(oVx, color="red")
        ax1.set_ylabel(r"$-{V_x}(km s^{-1})$")

        # Temperature
        ax2 = axs[2]
        ax2.plot(
            T,
            color="black",
            label=r"$T_p$",
            linewidth=1,
        )
        ax2.plot(oT, color="red")
        # ax2.legend([f"T ({(T.index[1]-T.index[0]).total_seconds()}s cadence)"])
        ax2.set_ylabel(r"$T_p (K)$")

        # Proton Density
        ax4 = axs[3]
        ax4.semilogy(
            Np,
            color="black",
            label="N_p",
            linewidth=1,
        )
        ax4.semilogy(oNp, color="red")

        # ax4.legend([f"Np ({(Np.index[1]-Np.index[0]).total_seconds()}s cadence)"])
        ax4.set_ylabel(r"$n_p$ (# $cm^{-3}$)")

        # Pressure
        ax5 = axs[4]
        ax5.semilogy(
            -Mf,
            label="Mass Flux",
            color="black",
            linewidth=1,
        )
        ax5.semilogy(-oMf, color="red")
        # ax5.legend([f"Mf ({(Mf.index[1]-Mf.index[0]).total_seconds()}s cadence)"])
        ax5.set_ylabel(r"$- m_{p}n_{p} V_x (kg km s^{-1})$")

        # # Plot the relevant columns
        if zone1 and zone2:
            for ax in axs:
                # ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                for zone, colour in zip((zone1, zone2), ("black", "red")):
                    ax.axvspan(
                        zone["start_time"], zone["end_time"], color=colour, alpha=0.3
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

    def extract_orbit_data(self, from_data=False, stepMinutes=30):
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

    def plot_spherical_coords(
        self, other, accelerated=False, objFolder=f"{BASE_PATH}Figures/Orbit_3d/"
    ):
        """
        Plots the longitude and latitude of a given spacecraft object
        """

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

                        # TODO: Potentially add a plot of the solar orbiter observations and highlight times which are used

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

    def zoom_in(self, start_time: datetime, end_time: datetime, stepMinutes=30):
        """
        Zoom into a specific part of the signal and store new self.df in memory
        Uses start_time and end_time to define limits
        """
        self.start_time = start_time
        self.end_time = end_time

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
    - TODO: Make relevant case studies (e.g., a couple of hours of obs on SolO to more time in PSP)
    - TODO: apply algo
    - TODO: Show all relevant results at the same time (similar to P1 but with same variables?)
    - TODO: how to make statistically sound? -> Use significance through P-values?
    """
    # Parker Encounter 6
    # psp_fld_l2_mag_RTN_4_Sa_per_Cyc_20201002_v01
    OBJ_CADENCE = 60  # To one minute resolution
    ORBIT_GAP = 30  # Orbital gap between measurements in Minutes
    PLOT_ORBITS = True

    psp_e6_overview = {
        "name": "PSPpriv_e6",
        "mid_time": datetime(2020, 10, 2),
        "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
    }

    solo_e6_overview = {
        "name": "SolOpriv_e6",
        "mid_time": datetime(2020, 10, 3),
        "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
        "show": False,
    }

    # Prepare the objects with measurements inside DF
    psp = Spacecraft(**psp_e6_overview)
    solo = Spacecraft(**solo_e6_overview)

    # Solar orbiter lines up with latitude?
    # In september
    # TODO: Should make multiple test cases using different solo_zoomed and psp_zoomed profiles.
    solo_zoomed = {
        "start_time": datetime(2020, 10, 2, 11, 30),
        "end_time": datetime(2020, 10, 3, 11, 53),
        "stepMinutes": ORBIT_GAP,
    }

    psp_zoomed = {
        "start_time": datetime(2020, 9, 27),
        "end_time": datetime(2020, 9, 30, 13, 7),
        "stepMinutes": ORBIT_GAP,
    }

    solo.plot_solo_psp_df(psp, zone1=solo_zoomed, zone2=psp_zoomed)
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

    # TODO: Make framework that plots the orbits and extracts the data to the side
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

        # TODO: Save these resulting coordinates in some file. They could be useful
        # Spacecraft object calling the function is where the solar wind is being mapped from

        if PLOT_ORBITS:
            solo.plot_spherical_coords(
                psp,
                accelerated=True,
                objFolder=f"{orbit_case_path}/Images/",
            )


# %%

if __name__ == "__main__":
    psp_e6()
