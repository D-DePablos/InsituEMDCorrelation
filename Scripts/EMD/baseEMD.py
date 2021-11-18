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
                 speedSet=None,
                 shortDuration=1.5,
                 shortDisplacement=1.5,
                 MARGIN=48,
                 inKind=False,
                 windDispParam: int = 1,  # How many measurements to move by
                 ):

        # The job of the init should be to get to a standardised format
        # Not to use it!
        self.PeriodMinMax = PeriodMinMax
        self.showFig = showFig
        self.multiCPU = multiCPU
        self.detrendBoxWidth = detrendBoxWidth
        self.corrThrPlotList = corrThrPlotList
        self.speedSet = speedSet
        self.shortDuration = shortDuration
        self.shortDisplacement = shortDisplacement
        self.inKind = inKind
        self.windDispParam = windDispParam

        possibleCaseNames = ["ISSIcasesAIA",
                             "PSP_SolO_e6", "April2020_SolO_WIND", "STA_PSP"]
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
                "forceCreate": False,
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
                           ["171", "193"], shortDFParams[0], "sun"),
                shortDFDic(df_193.copy(), "193", cases,
                           shortDFParams[1], "193", "sun")
            ]
            self.longDFDic = longDFDic(
                df_is.copy(), "PSP", "psp", 1, self.speedSet)

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
                                           shortParams, ["SolO"], "solo")]

            self.longDFDic = longDFDic(
                long_SPC.df.copy(), "PSP", "psp", 1, self.speedSet)

        elif caseName == "April2020_SolO_WIND":
            from Imports.Spacecraft import Spacecraft
            self.saveFolder = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/unsafe/EMD_Results/SolO_Earth_April_2020/"
            objCadenceSeconds = 92

            # Create or read existing cases
            soloEarthcases = {
                "longTimes": (datetime(2020, 4, 15), datetime(2020, 4, 23)),
                "shortTimes": (datetime(2020, 4, 15), datetime(2020, 4, 20, 23, 58)),
                "shortDuration": self.shortDuration,
                "caseName": f"{self.shortDuration}_By_{self.shortDisplacement}_Hours/SolO",
                "shortDisplacement": self.shortDisplacement,
                "savePicklePath": "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/EMD/cases/cases_April_2020_SolO.pickle",
                "forceCreate": True,
                "firstRelevantLongTime": datetime(2020, 4, 15, 20),
                "MARGIN": MARGIN,
            }
            cases = caseCreation(**soloEarthcases)

            # Long SPC (PSP)(IN-SITU)
            # See if we can make a Spacecraft object etc, with more data to then select from it
            long_SPC = Spacecraft(name="Earth_April_2020",
                                  cadence_obj=objCadenceSeconds,
                                  )
            short_SPC = Spacecraft(name="SolO_April_2020",
                                   cadence_obj=objCadenceSeconds, )

            long_SPC.df = long_SPC.df[longParams] if longParams != None else long_SPC.df
            short_SPC.df = short_SPC.df[shortParams] if shortParams != None else short_SPC.df

            for _df in (long_SPC.df, short_SPC.df):
                _df = _df.interpolate()

            self.shortDFDics = [shortDFDic(short_SPC.df, "SolO", cases,
                                           shortParams, ["SolO"], "solo")]

            self.longDFDic = longDFDic(
                long_SPC.df, "WIND", "L1", 1, self.speedSet)

        # NOTE: Need to fix error in saving / opening files
        elif caseName == "STA_PSP":
            from Imports.Spacecraft import Spacecraft
            self.saveFolder = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/unsafe/EMD_Results/STA_PSP/"
            objCadenceSeconds = 60
            staPSPCases = {
                "longTimes": (datetime(2019, 11, 11), datetime(2019, 11, 20)),
                "shortTimes": (datetime(2019, 11, 12, 6), datetime(2019, 11, 18, 6)),
                "shortDuration": self.shortDuration,
                "shortDisplacement": self.shortDisplacement,
                "caseName": f"{self.shortDuration}_By_{self.shortDisplacement}_Hours/PSP",
                "savePicklePath": "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/EMD/cases/cases_STA_PSP.pickle",
                "forceCreate": True,
                "firstRelevantLongTime": datetime(2019, 11, 10),
                "MARGIN": MARGIN,
            }
            cases = caseCreation(**staPSPCases)

            long_SPC = Spacecraft(name="STA_Nov_2019",
                                  cadence_obj=objCadenceSeconds, remakeCSV=True)
            short_SPC = Spacecraft(name="PSP_Nov_2019",
                                   cadence_obj=objCadenceSeconds)

            long_SPC.df = long_SPC.df[longParams] if longParams != None else long_SPC.df
            short_SPC.df = short_SPC.df[shortParams] if shortParams != None else short_SPC.df

            for _df in (long_SPC.df, short_SPC.df):
                _df = _df.interpolate()

            self.shortDFDics = [shortDFDic(short_SPC.df, "PSP", cases,
                                           shortParams, ["PSP"], "psp")]

            self.longDFDic = longDFDic(
                long_SPC.df.copy(), "STA", "stereo_a", 1, self.speedSet)
        else:
            raise NotImplementedError(
                f"{caseName} not implemented. Use one of {possibleCaseNames}")

    def fixDFNames(self, name):
        for column in list(self.longDFDic.df):
            if name not in column:
                self.longDFDic.df[f"{name}_{column}"] = self.longDFDic.df[column]
                del self.longDFDic.df[column]

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({self.caseName},{self.shortParams}, {self.longParams},{self.PeriodMinMax})".format(self)

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
            inKind=self.inKind,
            windDispParam=self.windDispParam,
        )

    def plotTogether(self, showBox=None, gridRegions=False, shortName="", longName="", missingData=None, skipParams=[]):
        superSummaryPlotGeneric(
            shortDFDic=self.shortDFDics,
            longDFDic=self.longDFDic,
            unsafeEMDataPath=self.saveFolder,
            PeriodMinMax=self.PeriodMinMax,
            corrThrPlotList=self.corrThrPlotList,
            showFig=self.showFig,
            showBox=showBox,  # showBox = ([X0, XF], [Y0, YF]) - in datetime
            gridRegions=gridRegions,
            shortName=shortName,
            longName=longName,
            missingData=missingData,
            inKind=self.inKind,
            baseEMDObject=self,
            skipParams=skipParams,
        )


def PSPSolOCase(show=False):
    # Dictionary intro
    PSP_SolOVars = {
        "caseName": "PSP_SolO_e6",
        "shortParams": ["Btotal", "B_R", "V_R", "Mf", "N_RPW"],
        "longParams": ["Btotal", "B_R", "V_R", "Mf", "N"],
        "PeriodMinMax": [5, 22],
        "showFig": show,
        "detrendBoxWidth": None,
        "corrThrPlotList": np.arange(0.65, 1, 0.05),
        "multiCPU": 4,
        "caseName": "PSP_SolO_e6",
        "speedSet": (300, 200, 250),  # High - low - mid
    }

    # showBox = ([X0, XF], [Y0, YF]) - in datetime
    box = ((datetime(2020, 9, 27, 0), datetime(2020, 9, 27, 5)),
           (datetime(2020, 10, 1, 20, ), datetime(2020, 10, 2, 0, 13)))
    pspSolOe6EMD = baseEMD(**PSP_SolOVars)
    # pspSolOe6EMD.plotSeparately()
    pspSolOe6EMD.corrThrPlotList = np.arange(0.75, 1, 0.1)
    pspSolOe6EMD.plotTogether(showBox=box, gridRegions=True)


def ISSICase(show=False):
    # Is a dictionary
    ISSI_AIAVars = {
        "caseName": "ISSIcasesAIA",
        # Use all Parameters by setting to None
        "shortParams": None,
        "longParams": None,
        "PeriodMinMax": [5, 30],
        "showFig": show,
        "detrendBoxWidth": 200,
        "corrThrPlotList": np.arange(0.75, 1, 0.1),
        "multiCPU": 4,
        "speedSet": None,
    }

    issiEMD = baseEMD(**ISSI_AIAVars)
    # issiEMD.plotSeparately()
    # issiEMD.plotTogether(showBox=None, gridRegions=(2, 3, True, True))
    issiEMD.plotTogether(showBox=None, gridRegions=True)


def STAPSPCase(show=True):
    MARGIN = 48
    Kwargs = {
        "caseName": "STA_PSP",
        "shortParams": ["B_R", "B_T", "B_N"],
        "longParams": ["B_R", "B_T", "B_N", "V_R"],
        "PeriodMinMax": [60, 720],  # Very long periods
        "showFig": show,
        "detrendBoxWidth": None,
        "corrThrPlotList": np.arange(0.75, 1, 0.1),
        "multiCPU": 4,
        "speedSet": None,
        "shortDuration": 30,  # In hours
        "shortDisplacement": 6,
        "MARGIN": MARGIN,
        "inKind": True,
        "windDispParam": 10,
    }

    box = None
    mData = None

    stapspEMD = baseEMD(**Kwargs)
    stapspEMD.plotSeparately()
    stapspEMD.fixDFNames("STA")
    stapspEMD.plotTogether(showBox=box, gridRegions=True,
                           missingData=mData, shortName="ST-A (0.95A.U.)", longName="PSP", skipParams=["STA_V_R"])


def SolOEarth2020Case(show=True):
    MARGIN = 60
    Vars = {
        "caseName": "April2020_SolO_WIND",
        "shortParams": ["B_R", "B_T", "B_N"],
        "longParams": ["B_R", "B_T", "B_N", "V_R"],
        "PeriodMinMax": [60, 720],  # Very long periods
        "showFig": show,
        "detrendBoxWidth": None,
        "corrThrPlotList": np.arange(0.75, 1, 0.1),
        "multiCPU": 3,
        "speedSet": None,
        "shortDuration": 20,  # In hours
        "shortDisplacement": 1,
        "MARGIN": MARGIN,
        "inKind": True,
        "windDispParam": 10,
    }

    # showBox = ([X0, XF], [Y0, YF]) - in datetime

    # from datetime import timedelta
    # box = ([datetime(2020, 4, 20, 1, 0) - timedelta(hours=MARGIN), datetime(2020, 4, 21, 0, 0) - timedelta(hours=MARGIN)],
    #    [datetime(2020, 4, 19, 8, 50) - timedelta(hours=MARGIN), datetime(2020, 4, 20, 9) - timedelta(hours=MARGIN)])
    box = None

    # missingData = namedtuple("missingData", [
    #                          "shortMissing", "longMissing", "colour"], defaults=(None, None, "red"))
    shortDataGaps = None
    longDataGaps = None
    # For SolO - Earth there is no missing data essentially
    mData = None

    soloAprilEMD = baseEMD(**Vars)
    soloAprilEMD.plotSeparately()
    soloAprilEMD.fixDFNames("WIND")
    soloAprilEMD.plotTogether(
        showBox=box,
        gridRegions=True,
        shortName="SolO (0.8A.U.)",
        longName="WIND (1 A.U.)",
        missingData=mData,
        skipParams=["WIND_V_R"]
    )


if __name__ == "__main__":
    # TODO: Need to plot some more SolO - WIND cases
    # Need to write up the paper
    # Shoould make summary plot

    # ISSICase(show=False)
    # SolOEarth2020Case(show=False)
    # PSPSolOCase()
    STAPSPCase(show=False)
