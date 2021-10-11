# This file should be used for routines and information that can be shared accross test cases
BASE_PATH = "/home/diegodp/Documents/PhD/Paper_3/SolO_SDO_EUI/"

from sys import path

path.append("{BASE_PATH}Scripts/")
path.append(f"/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs

from datetime import datetime
from EMD.importsProj3.signalAPI import (
    emdAndCompareCases,
    caseCreation,
    shortDFDic,
    longDFDic,
    superSummaryPlotGeneric,
)


class baseEMD:
    def __init__(self,
                 caseName,
                 shortParams,
                 longParams,
                 PeriodMinMax,
                 showFig=True,
                 detrendBoxWidth=200,
                 corrThrPlotList=np.arange(0.65, 1, 0.05),
                 multiCPU=None,
                 speedSet=None
                 ):

        # The job of the init should be to get to a standardised format
        # Not to use it!
        self.PeriodMinMax = PeriodMinMax
        self.showFig = showFig
        self.multiCPU = multiCPU
        self.detrendBoxWidth = detrendBoxWidth
        self.corrThrPlotList = corrThrPlotList
        self.speedSet = speedSet

        possibleCaseNames = ["ISSIcasesAIA", "PSP_SolO_e6"]
        if caseName == "ISSIcasesAIA":
            # The directories and save folders are on a per-case basis
            unsafe_dir = "/home/diegodp/Documents/PhD/Paper_3/SolO_SDO_EUI/unsafe/"
            self.saveFolder = f"{unsafe_dir}ISSI/New_Method/"
            makedirs(self.saveFolder, exist_ok=True)

            # dataFolder only used in setup
            dataFolder = f"/home/diegodp/Documents/PhD/Paper_3/SolO_SDO_EUI/Scripts/ISSI/data/"

            # Get the cases and put them together with respective AIA observations in Dic
            AIACases = {
                "shortTimes": (datetime(2018, 10, 29, 16), datetime(2018, 10, 30, 23, 50)),
                "longTimes": (datetime(2018, 10, 31, 8), datetime(2018, 11, 2, 8)),
                "shortDuration": 3,
                "caseName": "SDO_AIA",
                "shortDisplacement": 3,
                "savePicklePath": "/home/diegodp/Documents/PhD/Paper_3/SolO_SDO_EUI/Scripts/ISSI/cases/AIAcases.pickle",
                "forceCreate": True,
            }
            cases = caseCreation(**AIACases)

            # IN SITU DATA (LONG)
            df_is = pd.read_csv(f"{dataFolder}small_ch_in_situ.csv")
            df_is.index = pd.to_datetime(df_is["Time"])
            del df_is["Time"]
            longDFParams = longParams if longParams != None else df_is.columns
            df_is = df_is[longDFParams]

            # REMOTE DATA (SHORT)
            try:
                df_171 = pd.read_csv(
                    f"{dataFolder}small_ch_171_lc_in.csv", index_col="Time")
                df_193 = pd.read_csv(
                    f"{dataFolder}small_ch_193_lc_in.csv", index_col="Time")
                print("Loaded csv successfully")

                for _df in (df_171, df_193):
                    _df.index = pd.to_datetime(_df.index)

            except FileNotFoundError:
                raise FileNotFoundError("Unable to open Dataframes")

            # Store the shortDFDics and longDFDic
            shortDFParams = (df_171.columns, df_193.columns) if shortParams == None else (
                shortParams, shortParams)
            self.shortDFDics = [
                shortDFDic(df_171.copy(), "171", cases,
                           shortDFParams[0], "SDO_AIA", "sun"),
                shortDFDic(df_193.copy(), "193", cases,
                           shortDFParams[1], "SDO_AIA", "sun")
            ]
            self.longDFDic = longDFDic(df_is.copy(), "PSP")

        elif caseName == "PSP_SolO_e6":
            from Imports.Spacecraft import Spacecraft
            self.saveFolder = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/unsafe/EMD_Results/encounter6_Parker_1_5/"
            objCadenceSeconds = 60

            # Create or read existing cases
            PSPSolO_e6_cases = {
                "shortTimes": (datetime(2020, 9, 30), datetime(2020, 10, 2, 23)),
                "longTimes": (datetime(2020, 9, 24, 12), datetime(2020, 10, 3)),
                "shortDuration": 1.5,
                "caseName": "SolO",
                "shortDisplacement": 1.5,
                "savePicklePath": "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/EMD/cases/cases_1-5.pickle",
                "forceCreate": True,
            }
            cases = caseCreation(**PSPSolO_e6_cases)

            # Long SPC (PSP)(IN-SITU)
            # See if we can make a Spacecraft object etc, with more data to then select from it
            long_SPC = Spacecraft(name="PSP_Scaled_e6",
                                  cadence_obj=objCadenceSeconds,
                                  )
            short_SPC = Spacecraft(name="SolO_Scaled_e6",
                                   cadence_obj=objCadenceSeconds, )
            for _df in (long_SPC.df, short_SPC.df):
                _df = _df.interpolate()

            self.shortDFDics = [shortDFDic(short_SPC.df, "SolO", cases,
                                           shortParams, "SolO", "solo")]

            self.longDFDic = longDFDic(
                long_SPC.df.copy(), "PSP", "psp", 1, self.speedSet)

        elif caseName == "SolO_Earth_April_2020":
            from Imports.Spacecraft import Spacecraft
            self.saveFolder = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/unsafe/EMD_Results/SolO_Earth_April_2020/"
            objCadenceSeconds = 60

            # Create or read existing cases
            soloEarthcases = {
                "shortTimes": None,
                "longTimes": None,
                "shortDuration": None,
                "caseName": None,
                "shortDisplacement": None,
                "savePicklePath": "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/EMD/cases/cases_April_2020_SolO.pickle",
                "forceCreate": True,
            }
            cases = caseCreation(**soloEarthcases)

            # Long SPC (PSP)(IN-SITU)
            # See if we can make a Spacecraft object etc, with more data to then select from it
            long_SPC = Spacecraft(name="Earth_April_2020",
                                  cadence_obj=objCadenceSeconds,
                                  )
            short_SPC = Spacecraft(name="SolO",
                                   cadence_obj=objCadenceSeconds, )
            for _df in (long_SPC.df, short_SPC.df):
                _df = _df.interpolate()

            self.shortDFDics = [shortDFDic(short_SPC.df, "SolO", cases,
                                           shortParams, "SolO", "solo")]

            self.longDFDic = longDFDic(long_SPC.df.copy(), "PSP", "psp")

        else:
            raise NotImplementedError(
                f"{caseName} not implemented. Use one of {possibleCaseNames}")

    def plotSeparately(self):
        emdAndCompareCases(
            self.shortDFDics,
            self.longDFDic,
            saveFolder=self.saveFolder,
            PeriodMinMax=self.PeriodMinMax,
            showFig=self.showFig,
            detrendBoxWidth=self.detrendBoxWidth,
            corrThrPlotList=self.corrThrPlotList,
            multiCPU=self.multiCPU,
            speedLim=self.speedSet,
        )

    def plotTogether(self, showBox=None):
        # TODO: Fix diagonal lines
        superSummaryPlotGeneric(
            shortDFDic=self.shortDFDics,
            longDFDic=self.longDFDic,
            unsafeEMDataPath=self.saveFolder,
            PeriodMinMax=self.PeriodMinMax,
            corrThrPlotList=self.corrThrPlotList,
            showFig=self.showFig,
            showBox=showBox,
        )
        raise NotImplementedError("Still unable to plot all together")


def PSPSolOCase():
    # Dictionary intro
    PSP_SolOVars = {
        "caseName": "PSP_SolO_e6",
        "shortParams": ["Btotal", "B_R", "V_R", "Mf", "N_RPW"],
        "longParams": ["Btotal", "B_R", "V_R", "Mf", "N"],
        "PeriodMinMax": [5, 22],
        "showFig": True,
        "detrendBoxWidth": None,
        "corrThrPlotList": np.arange(0.65, 1, 0.05),
        "multiCPU": 4,
        "caseName": "PSP_SolO_e6",
        "speedSet": (300, 200, 250),  # High - low - mid
    }

    box = ((datetime(2020, 9, 27, 0), datetime(2020, 9, 27, 5)),
           (datetime(2020, 10, 1, 20, ), datetime(2020, 10, 2, 0, 13)))
    pspSolOe6EMD = baseEMD(**PSP_SolOVars)
    # pspSolOe6EMD.plotSeparately()
    pspSolOe6EMD.corrThrPlotList = np.arange(0.75, 1, 0.1)
    pspSolOe6EMD.plotTogether(showBox=box)


def ISSICase():
    # Is a dictionary
    ISSI_AIAVars = {
        "caseName": "ISSIcasesAIA",
        # Use all Parameters by setting to None
        "shortParams": None,
        "longParams": None,
        "PeriodMinMax": [5, 30],
        "showFig": False,
        "detrendBoxWidth": 200,
        "corrThrPlotList": np.arange(0.65, 1, 0.05),
        "multiCPU": 4,
    }

    issiEMD = baseEMD(**ISSI_AIAVars)
    issiEMD.plotSeparately()
    issiEMD.plotTogether(showBox=None)


if __name__ == "__main__":
    # ISSICase()
    PSPSolOCase()
