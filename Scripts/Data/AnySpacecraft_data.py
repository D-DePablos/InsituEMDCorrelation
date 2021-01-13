#%%
BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"

from sys import path

path.append(f"{BASE_PATH}Scripts/")

# from helpers import resample_and_rename
from Data.helpers import backmap_calculation, fcl
from decorators_utils import trace

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

# Units
units_dic = {
    "density": r"#/$cm^{3}$",
    "temperature": r"$K$",
    "velocity": r"$km/s$",
    "velocity_x": r"$km/s$",
    "velocity_y": r"$km/s$",
    "velocity_z": r"$km/s$",
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
        name="ST-A",
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

        elif self.name == "WIND":
            from heliopy.data import wind

            df_mag = wind.mfi_h0(self.start_time, self.end_time)
            df_swe = wind.swe_h1(self.start_time, self.end_time)
            self.cadence = None
            # Output of self.df.columns
            """
            """
            self.df = pd.append(df_mag, df_swe)

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
                df = pd.read_csv(f"{BASE_PATH}Scripts/Data/Prepped_csv/solo_sept.csv")

                self.df = df

            except FileNotFoundError:
                import cdflib
                from glob import glob

                cdf_epoch = cdflib.cdfepoch()

                def SWEAP_prep():
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
                    for day, file in enumerate(sorted(glob(f"{cdf_path}*.cdf"))):
                        cdf = cdflib.CDF(file)
                        time = cdf_epoch.to_datetime(cdf["Epoch"])
                        _df = pd.DataFrame({}, index=time)

                        for i in ("V_RTN", "N", "validity"):
                            if i == "V_RTN":  # Allow for multidimensional
                                for n, arg in zip(
                                    (0, 1, 2), ("", "_T", "_N")
                                ):  # R is radial velocity
                                    _df[f"velocity{arg}"] = cdf[i][:, n]
                            else:
                                _df[i] = cdf[i]

                        # Join the dataframes
                        if day == 0:
                            self.df = _df
                        else:
                            self.df = self.df.append(_df)

                    self.df["Time"] = self.df.index

                SWEAP_prep()
                # TODO: Add function for Magnetic field

            self.df.to_csv(
                f"{BASE_PATH}Scripts/Data/Prepped_csv/solo_sept.csv",
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
                    self.df = pd.read_csv(
                        f"{BASE_PATH}Scripts/Data/Prepped_csv/psp_sept.csv"
                    )

                except FileNotFoundError:
                    # When file not available, derive it
                    cdf_path = "/mnt/sda5/Python/InsituEMDCorrelation/unsafe/Resources/PSP_Data/Encounter6/FLD/"
                    for day, file in enumerate(sorted(glob(f"{cdf_path}*.cdf"))):
                        cdf = cdflib.CDF(file)
                        time = cdf_epoch.to_datetime(cdf["epoch_mag_RTN_4_Sa_per_Cyc"])
                        mag = cdf["psp_fld_l2_mag_RTN_4_Sa_per_Cyc"]
                        cdf.close()

                        # Dataframe creation and time
                        _df = pd.DataFrame({}, index=time)
                        for i, label in enumerate(cdf["label_RTN"][0]):
                            _df[str(label)] = mag[:, i]

                        # Join the dataframes
                        if day == 0:
                            self.df = _df
                        else:
                            self.df = self.df.append(_df)

                    self.df["Time"] = self.df.index
                    self.df.to_csv(
                        "/mnt/sda5/Python/InsituEMDCorrelation/Scripts/Data/Prepped_csv/psp_sept.csv",
                        index=False,
                    )

            FLD_prep()
            # TODO : Get and add kinetic data!

        else:
            raise NotImplementedError(f"{self.name} not implemented")

        # Add all variables if using Heliopy datasets
        if add_all_heliopy_vars:  # Add more parameters only if add_extra flag is true
            for parameter in self.rel_vars:
                if parameter not in list(self.df):
                    _data = self.data.to_dataframe()[self.rel_vars[parameter]]
                    _data[_data < 0.001] = np.nan  # Does this hold for all parameters?
                    self.df[parameter] = _data

        # After loading it breaks datetime format
        self.df.index = pd.to_datetime(self.df["Time"])
        del self.df["Time"]
        # TODO: Verify that the interpolation is working!
        self.df.fillna(method="pad")
        self.df = self.df.resample(f"{self.obj_cad}s").mean()

    @staticmethod
    def trace_field_lines(lon, lat, r, v_sw, rf=0.0046547454 * u.AU):
        """
        For each of the positions in self, track to origin
        lon, lat in deg
        r in AU
        """
        lon_ss = (
            lon
            + (
                (SOLROTRATE * r.value)
                / (v_sw["velocity"] * u.km / u.s).to(u.AU / u.s).value
            )
            * u.deg
        )

        lon_x = np.arange(lon.value, lon_ss.value, step=5)
        lat_x = np.repeat(lat.value, len(lon_x))
        rad_x = np.arange(rf.value, r.value, step=(r.value - rf.value) / len(lon_x))[
            ::-1
        ]
        fline = (
            lon_x * u.deg,
            lat_x * u.deg,
            rad_x * u.AU,
        )  # Compose long, lat, rad into fieldlines

        return fline

    def plot_df_values(self):
        # Plot all the available values

        for parameter in list(self.df):
            if parameter not in ["validity"]:
                if "velocity" in parameter:
                    units_dic[parameter] = units_dic["velocity"]

                if "N" in parameter:
                    units_dic[parameter] = units_dic["density"]

                if "B" in parameter:
                    units_dic[parameter] = units_dic["btotal"]

                data = self.df[parameter]
                # data[self.df["validity"] < 3] = np.nan

                plt.figure(figsize=(16, 12))
                plt.scatter(self.df.index, data, color="black", linestyle="--")

                # Labels and titles
                plt.title(f"{self.name} {parameter.capitalize()}")
                plt.xlabel("Date")
                plt.ylabel(units_dic[parameter])

                # X-axis reformatting
                # ax = plt.gca()
                # ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
                # ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

                if self.show:
                    plt.show()

                else:
                    plt.savefig(
                        f"/mnt/sda5/Python/InsituEMDCorrelation/Figures/{self.name}_{parameter.capitalize()}.png"
                    )

    def extract_orbit_data(self, from_data=False):
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
        else:
            starttime = self.df.index[0].to_pydatetime()
            endtime = self.df.index[len(self.df.index) - 1].to_pydatetime()

        times = []
        while starttime < endtime:
            times.append(starttime)
            starttime += timedelta(hours=1)

        # Generate positions
        sp_traj.generate_positions(times, "Sun", "IAU_SUN")
        sp_traj.change_units(u.au)
        self.sp_traj = sp_traj

        from sunpy.coordinates import frames

        self.sp_coords_carrington = sp_traj.coords.transform_to(
            frame=frames.HeliographicCarrington
        )

    def plot_orbit(self):
        """
        Plot the spacecraft orbit using the initialised data
        """
        sp_traj = self.sp_traj
        times_float = (sp_traj.times - sp_traj.times[0]).value
        plot_times = sp_traj.times.to_datetime()

        # Generate a set of timestamps to color the orbits by
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        kwargs = {"s": 3, "c": times_float}
        ax.scatter(sp_traj.x, sp_traj.y, sp_traj.z, **kwargs)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ###############################################################################
        # Plot radial distance and elevation as a function of time
        elevation = np.rad2deg(np.arcsin(sp_traj.z / sp_traj.r))

        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(plot_times, sp_traj.r)
        axs[0].set_ylim(0, 1.1)
        axs[0].set_ylabel("r (AU)")

        axs[1].plot(plot_times, elevation)
        axs[1].set_ylabel("Elevation (deg)")

        axs[2].plot(plot_times, sp_traj.speed)
        axs[2].set_ylabel("Speed (km/s)")

        plt.show()

    def multi_orbits(self, other):
        """
        Plot the spacecraft orbit using the initialised data
        """
        import matplotlib.dates as dates

        nsubplots = 7

        fig, axs = plt.subplots(nsubplots, 1, sharex=True, figsize=(20, 10))
        colors = ["red", "blue"]
        labels = ["solo", "psp"]

        for i, Spacecraft_ins in enumerate([self, other]):
            sp_traj = Spacecraft_ins.sp_traj
            plot_times = sp_traj.times.to_datetime()

            # Plot radial distance and elevation as a function of time
            elevation = np.rad2deg(np.arcsin(sp_traj.z / sp_traj.r))
            # Sunpy coordinates how?
            theta = np.rad2deg(
                2
                * np.arctan(
                    sp_traj.y / (sp_traj.x + np.sqrt(sp_traj.x ** 2 + sp_traj.y ** 2))
                )
            )

            axs[0].set_title("Orbital measurements")
            axs[0].plot(plot_times, sp_traj.r, color=colors[i], label=labels[i])
            axs[0].set_ylim(0, 1.1)
            axs[0].set_ylabel("r (AU)")

            axs[1].plot(plot_times, elevation, color=colors[i], label=labels[i])
            axs[1].set_ylabel("Elevation (deg)")

            axs[2].plot(
                plot_times,
                Spacecraft_ins.sp_coords_carrington.lon,
                color=colors[i],
                label=labels[i],
            )
            axs[2].set_ylabel("Longitude (deg)")

            # Plot the different parameters
            for parameter in list(Spacecraft_ins.df):
                if "velocity" in parameter.lower():
                    axs[3].plot(
                        Spacecraft_ins.df[parameter],
                        label=f"{labels[i]}_{parameter}",
                    )

                if "N" == parameter:
                    axs[4].plot(
                        Spacecraft_ins.df[parameter],
                        label=f"{labels[i]}_{parameter}",
                    )

                if "B" in parameter:
                    axs[5].plot(
                        Spacecraft_ins.df[parameter],
                        label=f"{labels[i]}{parameter}",
                    )

            # Here we plot the original values and such
            if self.backmap_df is not None:
                """
                "solo_rad": df_rf["r"].values,
                "solo_lat": df_rf["lat"].values,
                "solo_long": df_rf["lon"].values,
                "solo_v": df_rf["v"].values,
                "dt_psp": dt_list,
                "long_psp": long_list,
                "accdt_psp": accdt_list,
                "acclong_psp": acclong_list,
                """
                bar_width = 3600 / 2
                for index, row in enumerate(self.backmap_df.itertuples()):
                    start_end_calc = lambda t, halfwidth_s: [
                        t - timedelta(seconds=halfwidth_s),
                        t + timedelta(seconds=halfwidth_s),
                    ]
                    time = row.Index
                    dt_psp = row.dt_psp
                    st_dt, ed_dt = start_end_calc(dt_psp, bar_width)
                    long_psp = row.long_psp

                    # Accelerated
                    accdt_psp = row.accdt_psp
                    accst_dt, acced_dt = start_end_calc(accdt_psp, bar_width)
                    acc_long_psp = row.acclong_psp

                    axs[6].hlines(index, xmin=st_dt, xmax=ed_dt, color="blue")
                    axs[6].text(x=st_dt, y=index + 0.2, s=index)

                    axs[6].hlines(index, xmin=st_dt, xmax=ed_dt, color="yellow")
                    axs[6].text(x=accst_dt, y=index + 0.2, s=index)

            # # Average backmapped time
            # if i == 1 and project_into_other:
            #     # When required to plot more information
            #     vmean = np.mean(self.df["velocity"]) * u.km / u.s
            #     rf = self.sp_traj.r.mean().to(u.km)
            #     r0 = other.sp_traj.r.mean().to(u.km)

            #     dt, dtacc = backmap_calculation(vmean, r0, rf)

            #     tmean = (
            #         self.df.index[int(np.round(len(self.df.index) / 2))]
            #     ).to_pydatetime()

            #     tstart, tstartacc = (
            #         self.start_time - timedelta(seconds=dt.value),
            #         self.start_time - timedelta(seconds=dtacc.value),
            #     )
            #     tend, tendacc = (
            #         self.end_time - timedelta(seconds=dt.value),
            #         self.end_time - timedelta(seconds=dtacc.value),
            #     )

            # for j, ax in enumerate(axs):
            #     ax.legend()

            #     if j == 0 and project_into_other and i == 1:
            #         ax.axvspan(tstart, tend, color="yellow", alpha=0.3)
            #         ax.axvspan(tstartacc, tendacc, color="green", alpha=0.3)

            # Fix the labels
            # ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
            # ax.xaxis.set_major_formatter(dates.DateFormatter(fmt="%H:%M %d"))
            # ax.xaxis.set_minor_locator(dates.HourLocator(interval=4))

        plt.show()

    def plot_spherical_coords(self, other):
        """
        Plots the longitude and latitude of a given spacecraft object
        """
        assert self.sp_traj != None, "Please calculate the spacecraft trajectory"

        lon1 = self.sp_coords_carrington.lon
        lat1 = self.sp_coords_carrington.lat
        rad1 = self.sp_coords_carrington.radius

        lon2 = other.sp_coords_carrington.lon
        lat2 = other.sp_coords_carrington.lat
        rad2 = other.sp_coords_carrington.radius

        X1, Y1, Z1 = spherical_to_cartesian(rad1, lat1, lon1)
        X2, Y2, Z2 = spherical_to_cartesian(rad2, lat2, lon2)

        fline_set = []

        for i, t in enumerate(self.sp_coords_carrington.obstime):
            vsw = fcl(self.df, t.datetime)
            fline = self.trace_field_lines(lon1[i], lat1[i], rad1[i], v_sw=vsw)
            fline_set.append(
                spherical_to_cartesian(r=fline[2], lat=fline[1], lon=fline[0])
            )

        # Generate a set of timestamps to color the orbits by
        times_float1 = (
            self.sp_coords_carrington.obstime - self.sp_coords_carrington.obstime[0]
        ).value
        times_float2 = (
            other.sp_coords_carrington.obstime - other.sp_coords_carrington.obstime[0]
        ).value

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        kwargs1 = {"s": 3, "c": times_float1}  # Size and colour
        kwargs2 = {"s": 3, "c": times_float2}  # Size and colour

        ax.scatter(X1, Y1, Z1, label=self.name, **kwargs1)
        ax.scatter(X2, Y2, Z2, label=other.name, **kwargs2)
        for fline in fline_set:
            ax.scatter(fline[0], fline[1], fline[2])

        # Solar radius 0.0046547454 AU
        ax.scatter(0, 0, 0, label="Sun", c="orange", s=100)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.legend()

        plt.show()

    def backmap_to_other(self, other):
        """
        Backmaps one spacecraft to the other given that self.df has velocity data
        The time period is the amount of data being backmapped and can be part of the self.df
        """

        def df_gen(spc_trajs, variable_list):
            _df = pd.DataFrame({})

            for variable in variable_list:
                _df[variable] = spc_trajs.variable

        assert (
            self.sp_traj.r.value.mean() > other.sp_traj.r.value.mean()
        ), "Please use spacecraft furhter from sun as first one"

        _hour_df = (self.df.resample("3600s").mean()).copy()
        _hour_df.interpolate(inplace=True)

        # dataframes for easier splicing
        df_rf = pd.DataFrame(
            {
                "r": (self.sp_traj.r).to(u.km),
                "lon": self.sp_coords_carrington.lon,
                "lat": self.sp_coords_carrington.lat,
                "v": _hour_df["velocity"] * u.km / u.s,
            },
            index=_hour_df.index,
        )

        # r0 is spacecraft that is closest to Sun
        df_r0 = pd.DataFrame(
            {
                "r": (other.sp_traj.r).to(u.km),
                "lon": other.sp_coords_carrington.lon,
                "lat": other.sp_coords_carrington.lat,
            },
            index=other.sp_traj.times,
        )

        # We interpolate values which are missing and drop missing
        df_rf.interpolate(inplace=True)
        df_rf.dropna(inplace=True)
        df_r0.dropna(inplace=True)

        r_list = []
        dt_list, accdt_list = [], []
        long_list, acclong_list = [], []

        # TODO: Dinamically get radius
        r0 = df_r0["r"].mean()
        find_stamp = lambda time, dt: time - timedelta(
            seconds=dt
        )  # Calculate the time taken

        for row in df_rf.itertuples():
            _rf = row.r
            _long = row.lon
            _v = row.v
            _time = (row.Index).to_pydatetime()

            _dt, _acc_dt, _long, _acc_long = backmap_calculation(
                _v, r0=r0, rf=_rf, spcf_coords=_long
            )  # Time in seconds

            r_list.append(_rf)
            dt_list.append(find_stamp(_time, _dt))
            accdt_list.append(find_stamp(_time, _acc_dt))
            long_list.append(_long)
            acclong_list.append(_acc_long)

        self.backmap_df = pd.DataFrame(
            {
                "solo_rad": df_rf["r"].values,
                "solo_lat": df_rf["lat"].values,
                "solo_long": df_rf["lon"].values,
                "solo_v": df_rf["v"].values,
                "dt_psp": dt_list,
                "long_psp": long_list,
                "accdt_psp": accdt_list,
                "acclong_psp": acclong_list,
            },
            index=df_rf.index,
        )

        # For PSP and SolO It is approximately 1 day less than Sun-Earth, as PSP is at around 0.2 AU
        # How to do a good ballistic backmapping? -> Start by selecting less data!

    def zoom_in(self, start_time: datetime, end_time: datetime):
        """
        Zoom into a specific part of the signal and store new self.df in memory
        """
        self.start_time = start_time
        self.end_time = end_time
        self.extract_orbit_data()

        self.df = self.df[self.start_time : self.end_time]
        pass


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
    psp_e6_overview = {
        "name": "PSPpub",
        "mid_time": datetime(2020, 10, 2),
        # "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
        "show": False,
    }

    solo_e6_overview = {
        "name": "SolOpriv_e6",
        "mid_time": datetime(2020, 10, 3),
        "margin": timedelta(days=5),
        "cadence_obj": OBJ_CADENCE,
        "show": True,
    }

    # Prepare the objects with measurements inside DF
    psp = Spacecraft(**psp_e6_overview)
    solo = Spacecraft(**solo_e6_overview)

    # Solar orbiter lines up with latitude?
    solo_zoomed = {
        "start_time": datetime(2020, 10, 3, 20, 0),
        "end_time": datetime(2020, 10, 4, 0, 30),
    }

    psp_zoomed = {
        "start_time": datetime(2020, 9, 27),
        "end_time": datetime(2020, 10, 1, 13, 7),
    }
    solo.zoom_in(**solo_zoomed)
    psp.zoom_in(**psp_zoomed)
    print(solo.df)

    solo.plot_spherical_coords(psp)

    # Do the backmapping
    solo.backmap_to_other(psp)
    solo.multi_orbits(psp)


# def psp_sta():
#     # PSP conjunction with STEREO-A
#     # ST-A
#     stereo_a = {
#         "name": "ST-A",
#         "mid_time": datetime(2019, 11, 4, 12, 0),
#         "margin": timedelta(days=3),
#     }
#     sta = Spacecraft(**stereo_a)
#     sta.plot_df_values()

#     # Parker
#     parker_sta = {
#         "name": "PSP",
#         "mid_time": stereo_a["mid_time"],
#         "margin": timedelta(days=2),
#     }

#     psp = Spacecraft(**parker_sta)
#     psp.plot_df_values()


# %%
psp_e6()