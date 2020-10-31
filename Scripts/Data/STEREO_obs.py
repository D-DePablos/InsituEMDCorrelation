#%%
from datetime import datetime, timedelta
from helpers import resample_and_rename
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# TODO: Add positional information

# Units
units_dic = {
    "density": r"#/$cm^{3}$",
    "temperature": r"$K$",
    "velocity": r"$km/s$",
    "btotal": r"$T$",
}


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
    ):
        """
        :param name: Spacecraft name from one of [WIND, PSP, ST-A]
        :param mid_time: Time at centre of timeseries
        :param margin: Timemargin (Timedelta object) at either side of mid_time
        """

        self.name = name

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

        elif self.name == "PSP":
            from heliopy.data import psp as psp_data

            df_mag_data = psp_data.fields_mag_rtn_1min(self.start_time, self.end_time)
            sweap_l3 = psp_data.sweap_spc_l2(self.start_time, self.end_time)

            # MAG
            Bx = df_mag_data.to_dataframe()["psp_fld_l2_mag_RTN_1min_0"]
            By = df_mag_data.to_dataframe()["psp_fld_l2_mag_RTN_1min_1"]
            Bz = df_mag_data.to_dataframe()["psp_fld_l2_mag_RTN_1min_2"]
            Bt = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
            df_mag = pd.DataFrame({"Bx": Bx, "By": By, "Bz": Bz, "Bt": Bt})

            self.df = pd.DataFrame({})
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

        else:
            raise NotImplementedError(f"{self.name} not implemented")

        for parameter in self.rel_vars:
            if parameter not in list(self.df):
                _data = self.data.to_dataframe()[self.rel_vars[parameter]]
                _data[_data < 0.001] = np.nan  # Does this hold for all parameters?

                self.df[parameter] = _data

                # TODO: Resample the data to objective cadence as well

    def plot_df_values(self):
        for parameter in self.rel_vars:
            data = self.df[parameter]
            data[data < 0.001] = np.nan

            plt.figure(figsize=(8, 6))
            ax = plt.gca()
            plt.scatter(data.index, data, color="black", linestyle="--")

            plt.title(f"{self.name} {parameter.capitalize()}")
            plt.xlabel("Date")
            plt.ylabel(units_dic[parameter])

            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
            # ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

            plt.show()


# %%
# PSP conjunction
stereo_a = {
    "name": "ST-A",
    "mid_time": datetime(2019, 11, 4, 12, 0),
    "margin": timedelta(days=3),
}

parker_sta = {
    "name": "PSP",
    "mid_time": stereo_a["mid_time"],
    "margin": timedelta(days=2),
}

psp = Spacecraft(**parker_sta)
psp.plot_df_values()

sta = Spacecraft(**stereo_a)
sta.plot_df_values()

# %%
# Other random tests
# wind = Spacecraft(
#     name="WIND",
#     mid_time=stereo_a_PSP_conjuction["mid_time"]-timedelta(days=1),
#     margin=timedelta(days=1),
#     )

# wind.plot_df_values()
