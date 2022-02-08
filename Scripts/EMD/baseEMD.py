# This file should be used for routines and information that can be shared accross test cases
BASE_PATH = "/Users/ddp/Documents/PhD/solo_sdo/"

from os import makedirs
from sys import path
import time

path.append("{BASE_PATH}Scripts/")
path.append(f"/Users/ddp/Documents/PhD/inEMD_Github/Scripts/")
import numpy as np
from datetime import datetime, timedelta
from EMD.importsProj3.signalAPI import (
    emdAndCompareCases,
    caseCreation,
    shortDFDic,
    longDFDic,
    superSummaryPlotGeneric,
)

import functools
import time
from collections import namedtuple

boxTuple = namedtuple(
    "boxTuple", ["longData", "shortData", "color"], defaults=(None, None, "red")
)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time / 3600} hours")
        return value

    return wrapper_timer


class baseEMD:
    def __init__(
        self,
        caseName,
        shortParams,
        longParams,
        PeriodMinMax,
        shortDuration,
        shortDisplacement,
        showFig=True,
        detrendBoxWidth=200,
        corrThrPlotList=np.arange(0.65, 1, 0.05),
        multiCPU=None,
        speedSet=None,
        MARGIN=48,
        inKind=False,
        windDispParam: int = 1,  # How many measurements to move by
        accelerated=1,
        equal=False,
        relSpeeds=range(100, 501, 100),
        plot_allResults=False,
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
        self.accelerated = accelerated
        self.equal = equal
        self.relSpeeds = relSpeeds
        self.plot_allResults = plot_allResults

        self.caseSpecificName = (
            f"{self.shortDuration}_By_{self.shortDisplacement}_Hours"
        )

        possibleCaseNames = [
            "ISSIcasesAIA",
            "ISSIcasesHMI",
            "PSP_SolO_e6",
            "April2020_SolO_WIND",
            "STA_PSP",
        ]
        if caseName == "ISSIcasesAIA":
            # Long = PSP
            # Short = AIA
            from Imports.Spacecraft import ISSISpc

            # The directories and save folders are on a per-case basis
            _unsFolder = "/Users/ddp/Documents/PhD/solo_sdo/ISSIwork/unsafe/"
            self.saveFolder = _unsFolder + "ISSIcasesAIA/"
            objCadenceSeconds = 60

            self.shortTimes = (
                datetime(2018, 10, 29, 16),
                datetime(2018, 10, 30, 23, 50),
            )
            self.longTimes = (datetime(2018, 10, 31, 8), datetime(2018, 11, 2, 8))
            # Get the cases and put them together with respective AIA observations in Dic
            AIACases = {
                "shortTimes": self.shortTimes,
                "longTimes": self.longTimes,
                "shortDuration": self.shortDuration,
                "caseName": f"{self.caseSpecificName}/AIA",
                "shortDisplacement": self.shortDisplacement,
                "savePicklePath": f"{self.saveFolder}{self.caseSpecificName}.pickle",
                "forceCreate": True,
                # "firstRelevantLongTime": datetime(2018, 10, 31, 8) + timedelta(hours=MARGIN),
                "MARGIN": MARGIN,
            }
            cases = caseCreation(**AIACases, equal=self.equal)

            long_SPC = ISSISpc("ISSI_PSP_e1", cadence_obj=objCadenceSeconds)
            longDFParams = longParams if longParams != None else long_SPC.df.columns
            df_is = long_SPC.df[longDFParams]
            self.long_SPC = long_SPC

            # REMOTE DATA (SHORT)
            short_SPC = ISSISpc("ISSI_AIA_e1", cadence_obj=objCadenceSeconds)
            df_171 = short_SPC.df171
            df_193 = short_SPC.df193

            for df in (df_171, df_193):
                # Rename by referring specifically, more consistent
                df.rename({
                    "plume": "PL",
                    "cbpoint": "BP",
                    "chplume": "CHPL",
                    "chole": "CH",
                    "qsun": "QS",
                }, axis=1, inplace=True)

                # Unsafe renaming is cool but not necessary
                # df.set_axis("PL BP CHPL CH QS".split(), axis=1, inplace=True)

            # Store the shortDFDics and longDFDic
            shortDFParams = (
                (df_171.columns, df_193.columns)
                if shortParams == None
                else (shortParams, shortParams)
            )
            self.shortDFDics = [
                shortDFDic(
                    df_171.copy(), "171", cases, ["171", "193"], shortDFParams[0], "sun"
                ),
                shortDFDic(df_193.copy(), "193", cases, shortDFParams[1], "193", "sun"),
            ]
            self.longDFDic = longDFDic(
                df_is.copy(), "PSP", "psp", self.accelerated, self.speedSet
            )

        elif caseName == "ISSIcasesHMI":
            from Imports.Spacecraft import ISSISpc

            self.shortTimes = (
                datetime(2018, 10, 28, 0, 0),
                datetime(2018, 10, 30, 7, 48),
            )
            self.longTimes = (
                datetime(2018, 10, 31, 8),
                datetime(2018, 11, 2, 8),
            )

            # Short = HMI
            # Long = PSP

            _unsFolder = "/Users/ddp/Documents/PhD/solo_sdo/ISSIwork/unsafe/"
            self.saveFolder = _unsFolder + "ISSIcasesHMI/"
            objCadenceSeconds = 60 * 12  # Get to HMI cadence
            HMICases = {
                "shortTimes": self.shortTimes,
                "longTimes": self.longTimes,
                "shortDuration": self.shortDuration,
                "shortDisplacement": self.shortDisplacement,
                "caseName": f"{self.caseSpecificName}/HMI",
                "savePicklePath": f"{self.saveFolder}{self.caseSpecificName}.pickle",
                "forceCreate": True,
                "MARGIN": MARGIN,
            }
            cases = caseCreation(**HMICases, equal=True)

            # PSP is the long data
            long_SPC = ISSISpc("ISSI_PSP_e1", cadence_obj=objCadenceSeconds)
            longDFParams = longParams if longParams != None else long_SPC.df.columns
            df_is = long_SPC.df[longDFParams]
            self.long_SPC = long_SPC

            # HMI is the short data
            short_SPC = ISSISpc(
                "ISSI_HMI_e1", cadence_obj=objCadenceSeconds, remakeCSV=True
            )
            long_SPC.zoom_in(self.longTimes[0], self.longTimes[1])
            short_SPC.zoom_in(self.shortTimes[0], self.shortTimes[1])
            df_hmi = short_SPC.df
            shortDFParams = shortParams if shortParams != None else df_hmi.columns
            self.shortDFDics = [
                shortDFDic(df_hmi.copy(), "HMI", cases, ["HMI"], shortDFParams, "sun")
            ]

            self.longDFDic = longDFDic(
                df_is.copy(), "PSP", "psp", self.accelerated, self.speedSet
            )

        elif caseName == "PSP_SolO_e6":
            # PSP is long and SolO is short
            from Imports.Spacecraft import PSPSolO_e6 as Spacecraft

            self.saveFolder = "/Users/ddp/Documents/PhD/inEMD_Github/unsafe/EMD_Results/encounter6_Parker_1_5/"
            objCadenceSeconds = 60

            self.shortTimes = (datetime(2020, 10, 1), datetime(2020, 10, 3))
            self.longTimes = (datetime(2020, 9, 25, 0), datetime(2020, 9, 29))

            # Create or read existing cases
            PSPSolO_e6_cases = {
                "shortTimes": self.shortTimes,
                "longTimes": self.longTimes,
                "shortDuration": shortDuration,
                "caseName": f"{self.shortDuration}_By_{self.shortDisplacement}_Hours/SolO",
                "shortDisplacement": shortDisplacement,
                "savePicklePath": f"{self.saveFolder}{self.caseSpecificName}.pickle",
                "forceCreate": True,
                "equal": self.equal,
                "MARGIN": MARGIN,
            }
            cases = caseCreation(**PSPSolO_e6_cases)

            # Long SPC (PSP)(IN-SITU)
            # See if we can make a Spacecraft object etc, with more data to then select from it
            long_SPC = Spacecraft(
                name="PSP_Scaled_e6",
                cadence_obj=objCadenceSeconds,
            )
            short_SPC = Spacecraft(
                name="SolO_Scaled_e6",
                cadence_obj=objCadenceSeconds,
            )

            # Change N to N_old and N_RPW column to N in short_SPC
            short_SPC.df.rename(columns={"N": "N_old", "N_RPW": "N"}, inplace=True)

            long_SPC.zoom_in(self.longTimes[0], self.longTimes[1])
            short_SPC.zoom_in(self.shortTimes[0], self.shortTimes[1])
            long_SPC.df = long_SPC.df[longParams] if longParams != None else long_SPC.df
            short_SPC.df = (
                short_SPC.df[shortParams] if shortParams != None else short_SPC.df
            )

            for _df in (long_SPC.df, short_SPC.df):
                _df = _df.interpolate()

            self.shortDFDics = [
                shortDFDic(short_SPC.df, "SolO", cases, shortParams, ["SolO"], "solo")
            ]

            self.longDFDic = longDFDic(
                long_SPC.df.copy(), "PSP", "psp", 1, self.speedSet
            )

        elif caseName == "April2020_SolO_WIND":
            from Imports.Spacecraft import Spacecraft

            # Long is WIND, short is SolO

            self.saveFolder = "/Users/ddp/Documents/PhD/inEMD_Github/unsafe/EMD_Results/SolO_Earth_April_2020/"
            makedirs(self.saveFolder, exist_ok=True)
            objCadenceSeconds = 92

            self.longTimes = ((datetime(2020, 4, 18)), datetime(2020, 4, 22))
            self.shortTimes = ((datetime(2020, 4, 17)), datetime(2020, 4, 21))
            # Create or read existing cases
            soloEarthcases = {
                "longTimes": self.longTimes,
                "shortTimes": self.shortTimes,
                "shortDuration": self.shortDuration,
                "caseName": f"{self.caseSpecificName}/SolO",
                "shortDisplacement": self.shortDisplacement,
                "savePicklePath": f"{self.saveFolder}{self.caseSpecificName}.pickle",
                "forceCreate": True,
                # First relevant long time is important as otherwise margin does not work
                # "firstRelevantLongTime": self.longTimes[0]
                # + timedelta(hours=MARGIN / 2),
                "MARGIN": MARGIN,
                "equal": self.equal,
            }
            cases = caseCreation(**soloEarthcases)

            # Long SPC (PSP)(IN-SITU)
            # See if we can make a Spacecraft object etc, with more data to then select from it
            long_SPC = Spacecraft(
                name="Earth_April_2020",
                cadence_obj=objCadenceSeconds,
            )
            short_SPC = Spacecraft(
                name="SolO_April_2020",
                cadence_obj=objCadenceSeconds,
            )

            long_SPC.zoom_in(self.longTimes[0], self.longTimes[1])
            short_SPC.zoom_in(self.shortTimes[0], self.shortTimes[1])

            long_SPC.df["Btotal"] = np.sqrt(
                long_SPC.df["B_GSE_0"] ** 2
                + long_SPC.df["B_GSE_1"] ** 2
                + long_SPC.df["B_GSE_2"] ** 2
            )

            short_SPC.df["Btotal"] = np.sqrt(
                short_SPC.df["B_R"] ** 2
                + short_SPC.df["B_T"] ** 2
                + short_SPC.df["B_N"] ** 2
            )

            long_SPC.df = long_SPC.df[longParams] if longParams != None else long_SPC.df
            short_SPC.df = (
                short_SPC.df[shortParams] if shortParams != None else short_SPC.df
            )

            for _df in (long_SPC.df, short_SPC.df):
                _df = _df.interpolate()

            self.shortDFDics = [
                shortDFDic(short_SPC.df, "SolO", cases, shortParams, ["SolO"], "solo")
            ]

            self.longDFDic = longDFDic(
                long_SPC.df, "WIND", "L1", self.accelerated, self.speedSet
            )

        elif caseName == "STA_PSP":
            from Imports.Spacecraft import Spacecraft

            _unsFolder = "/Users/ddp/Documents/PhD/inEMD_Github/"
            self.saveFolder = f"{_unsFolder}unsafe/EMD_Results/STA_PSP/"
            makedirs(self.saveFolder, exist_ok=True)
            # Long = STA
            # Short = PSP
            # self.longTimes = datetime(2019, 11, 1), datetime(2019, 11, 10)
            # self.shortTimes = datetime(2019, 11, 1), datetime(2019, 11, 10)
            self.longTimes = datetime(2019, 11, 1), datetime(2019, 11, 6, 0, 1)
            self.shortTimes = datetime(2019, 11, 1), datetime(2019, 11, 6, 0, 1)
            objCadenceSeconds = 60
            staPSPCases = {
                "longTimes": self.longTimes,
                "shortTimes": self.shortTimes,
                "shortDuration": self.shortDuration,
                "shortDisplacement": self.shortDisplacement,
                "caseName": f"{self.caseSpecificName}/PSP",
                "savePicklePath": f"{self.saveFolder}{self.caseSpecificName}.pickle",
                "forceCreate": True,
                "MARGIN": MARGIN,
                "equal": self.equal,
                "firstRelevantLongTime": self.longTimes[0]
                + timedelta(hours=MARGIN / 2),
            }
            cases = caseCreation(**staPSPCases)

            # Long spacecraft is STA
            long_SPC = Spacecraft(
                name="STA_Nov_2019", cadence_obj=objCadenceSeconds, remakeCSV=False
            )
            short_SPC = Spacecraft(name="PSP_Nov_2019", cadence_obj=objCadenceSeconds)
            long_SPC.zoom_in(self.longTimes[0], self.longTimes[1])
            short_SPC.zoom_in(self.shortTimes[0], self.shortTimes[1])
            short_SPC.df.fillna(method="ffill", inplace=True)
            long_SPC.df.fillna(method="ffill", inplace=True)

            short_SPC.df["Btotal"] = np.sqrt(
                short_SPC.df["B_R"] ** 2
                + short_SPC.df["B_T"] ** 2
                + short_SPC.df["B_N"] ** 2
            )

            long_SPC.df.rename(columns={"B_TOTAL": "Btotal"}, inplace=True)

            short_SPC.df = (
                short_SPC.df[shortParams] if shortParams != None else short_SPC.df
            )
            long_SPC.df = long_SPC.df[longParams] if longParams != None else long_SPC.df

            self.shortDFDics = [
                shortDFDic(short_SPC.df, "PSP", cases, shortParams, ["PSP"], "psp")
            ]

            self.longDFDic = longDFDic(
                long_SPC.df.copy(), "STA", "stereo_a", self.accelerated, self.speedSet
            )
        else:
            raise NotImplementedError(
                f"{caseName} not implemented. Use one of {possibleCaseNames}"
            )

    def fixDFNames(self, name):
        for column in list(self.longDFDic.df):
            if name not in column:
                self.longDFDic.df[f"{name}_{column}"] = self.longDFDic.df[column]
                del self.longDFDic.df[column]

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({self.caseName},{self.shortParams}, {self.longParams},{self.PeriodMinMax})".format(
            self
        )

    def showCases(self):
        for case in self.shortDFDics[0].cases:
            print(case)

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
            plot_allResults=self.plot_allResults,
        )

    def plotTogether(
        self,
        showBox=None,
        gridRegions=False,
        shortName="",
        longName="",
        missingData=None,
        skipParams=[],
        forceRemake=True,
        yTickFrequency=[0],
        xTickFrequency=[0],
        legendLocForce="upper right",
        onlySomeLegends=[],
    ):
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
            forceRemake=forceRemake,
            yTickFrequency=yTickFrequency,
            xTickFrequency=xTickFrequency,
            legendLocForce=legendLocForce,
            onlySomeLegends=onlySomeLegends,
            relSpeeds=self.relSpeeds,
        )


def PSPSolOCase(show=False):
    MARGIN = 0
    PSP_SolOVars = {
        "caseName": "PSP_SolO_e6",
        # "shortParams": ["Btotal", "B_R", "V_R", "Mf", "N"],  # Does N break?
        # "longParams": ["Btotal", "B_R", "V_R", "Mf", "N"],
        "shortParams": ["Btotal", "B_R", "V_R", "N"],  # Does N break?
        "longParams": ["Btotal", "B_R", "V_R", "N"],
        "PeriodMinMax": [5, 22],
        "showFig": show,
        "detrendBoxWidth": None,
        "corrThrPlotList": np.arange(0.75, 1, 0.05),
        "multiCPU": 8,
        "caseName": "PSP_SolO_e6",
        "speedSet": (320, 200, 250),  # High - low - mid
        "shortDuration": 1.5,
        "shortDisplacement": 1.5,
        "MARGIN": MARGIN,
        "inKind": True,
        "windDispParam": 1,
        "equal": True if MARGIN == 0 else False,
    }

    # showBox = ([X0, XF], [Y0, YF]) - in datetime
    # Expect correlation of 0.72 here
    # Box for telloni

    box = [
        # boxTuple(
        #     longData=(datetime(2020, 9, 27, 3, 15), datetime(2020, 9, 27, 4, 45)),
        #     shortData=(
        #         datetime(
        #             2020,
        #             10,
        #             1,
        #             21,
        #             34,
        #         ),
        #         datetime(2020, 10, 1, 23, 4),
        #     ),
        #     color="blue",
        # ),
        boxTuple(
            longData=(datetime(2020, 9, 27, 4), datetime(2020, 9, 27, 5, 30)),
            shortData=(
                (
                    datetime(
                        2020,
                        10,
                        2,
                        1,
                    ),
                    datetime(2020, 10, 2, 2, 30),
                )
            ),
        ),
    ]

    pspSolOe6EMD = baseEMD(**PSP_SolOVars)
    pspSolOe6EMD.plotSeparately()
    pspSolOe6EMD.corrThrPlotList = np.arange(0.8, 0.91, 0.05)
    pspSolOe6EMD.plotTogether(
        showBox=box,
        gridRegions=True,
        legendLocForce="upper right",
        longName="PSP",
        onlySomeLegends=["B_R"],
    )


@timer
def STAPSPCase(show=True):
    # Short is PSP, Long is STA
    MARGIN = 0
    BOX_WIDTH = 600
    SHORTDURN = 30
    SHORTDISPL = 2

    # periodList = [[60, 180], [60, 240], [120, 240]]
    periodList = [[60, 720], [60, 1400]]
    for p in periodList:
        Kwargs = {
            "caseName": "STA_PSP",
            "shortParams": ["B_R", "B_T", "B_N", "Btotal"],
            "longParams": ["B_R", "B_T", "B_N", "V_R", "Btotal"],
            "PeriodMinMax": p,
            "showFig": show,
            "detrendBoxWidth": BOX_WIDTH,
            "corrThrPlotList": np.arange(0.7, 1, 0.05),
            "multiCPU": 8,
            "shortDuration": SHORTDURN,  # In hours
            "shortDisplacement": SHORTDISPL,  # In hours
            "MARGIN": MARGIN,
            "inKind": True,
            "windDispParam": SHORTDURN / 2,
            "relSpeeds": [],
            "equal": True if MARGIN == 0 else False,
        }

        # Box is moved from middle to start
        box = [
            boxTuple(
                longData=(
                    datetime(2019, 11, 3, 0),
                    datetime(2019, 11, 3, 12, 0),
                ),
                shortData=(
                    datetime(
                        2019,
                        11,
                        2,
                        21,
                    ),
                    datetime(2019, 11, 3, 9),
                ),
                color="blue",
            ),
        ]
        mData = None

        stapspEMD = baseEMD(**Kwargs)
        stapspEMD.plotSeparately()
        stapspEMD.fixDFNames("STA")
        stapspEMD.corrThrPlotList = np.arange(0.85, 1, 0.1)
        stapspEMD.plotTogether(
            showBox=box,
            gridRegions=True,
            missingData=mData,
            shortName="PSP",
            longName="ST-A",
            skipParams=["STA_V_R"],
            legendLocForce="lower right",
            onlySomeLegends=["B_R"],
        )

    print("Done summary plots STA_PSP")


# @timer
def SolOEarth2020Case(show=True):
    # Short is SolO, long is Earth
    MARGIN = 0

    Vars = {
        "caseName": "April2020_SolO_WIND",
        # "shortParams": ["B_R", "B_T", "B_N", "Btotal"],
        # "longParams": ["B_R", "B_T", "B_N", "Btotal", "V_R"],
        "shortParams": ["B_R", "B_T", "B_N", "Btotal"],
        "longParams": ["B_R", "B_T", "B_N", "Btotal", "V_R"],
        "PeriodMinMax": [60, 60 * 6],  # 1 to 6 hours
        "showFig": show,
        "detrendBoxWidth": None,
        "corrThrPlotList": np.arange(0.75, 1, 0.1),
        "multiCPU": 6,
        "speedSet": None,
        "shortDuration": 18,  # In hours
        "shortDisplacement": 2,
        "MARGIN": MARGIN,
        "equal": True if MARGIN == 0 else False,
        "inKind": True,
        "windDispParam": 1,
        "relSpeeds": [450],  # Speeds which are plotted
    }

    # Box for orbital match (reference)
    box = []
    # box = [
    #     boxTuple(
    #         shortData=(datetime(2020, 4, 19, 2), datetime(2020, 4, 20, 4, 30)),
    #         longData=(datetime(2020, 4, 20, 0, 0), datetime(2020, 4, 21, 12, 0)),
    #     ),
    # ]
    # For SolO - Earth there is no missing data essentially
    mData = None

    soloAprilEMD = baseEMD(**Vars)
    soloAprilEMD.plotSeparately()
    soloAprilEMD.fixDFNames("WIND")
    soloAprilEMD.corrThrPlotList = np.arange(0.85, 1, 0.1)
    soloAprilEMD.plotTogether(
        showBox=box,
        gridRegions=True,
        shortName="SolO",
        longName="WIND",
        missingData=mData,
        skipParams=["WIND_V_R"],
        legendLocForce="lower right",
        onlySomeLegends=["B_T"],
    )
    print(
        f"Done April2020_SOLO_WIND with {soloAprilEMD.multiCPU} CPUs and {soloAprilEMD.windDispParam} minutes of window displacement"
    )


# @timer
def ISSICase(show=False):
    WINDDISP = 1
    MARGIN = 0
    ISSI_AIAVars = {
        "caseName": "ISSIcasesAIA",
        # Use all Parameters by setting to None
        "shortParams": None,
        "longParams": "B_R V_R N Mf T".split(),
        "PeriodMinMax": [5, 180],
        "showFig": show,
        "detrendBoxWidth": 200,
        "corrThrPlotList": np.arange(0.75, 1, 0.1),
        "multiCPU": 6,
        "shortDuration": 9,
        "shortDisplacement": 1,
        "MARGIN": MARGIN,  # If margin is set to 0 all long data is used
        "inKind": False,
        "windDispParam": WINDDISP,  # In Minutes
        "accelerated": 1,
    }

    issiEMD = baseEMD(**ISSI_AIAVars)
    issiEMD.long_SPC.plot_issi_psp_e1()
    issiEMD.plotSeparately()

    issiEMD.plotTogether(
        showBox=None,
        gridRegions=(2, 3, True, True),
        yTickFrequency=[0, 6, 12, 18],
        xTickFrequency=[0, 12],
        forceRemake=True,
    )
    print(
        f"Done ISSIcasesAIA with {issiEMD.multiCPU} CPUs and {issiEMD.windDispParam} minutes of window displacement"
    )



# @timer
def ISSIHMICase(show=False):
    WINDDISP = 1
    ISSI_HMIVars = {
        "caseName": "ISSIcasesHMI",
        "shortParams": ["ch_bpoint_flux", "ch_open_flux"],
        "longParams": "B_R V_R".split(),
        "PeriodMinMax": [24, 240],
        "showFig": show,
        "detrendBoxWidth": None,
        "corrThrPlotList": np.arange(0.75, 1, 0.1),
        "multiCPU": 6,
        "shortDuration": 9,
        "shortDisplacement": 1,
        "MARGIN": 0,
        "windDispParam": WINDDISP,
        "accelerated": 1,

    }

    hmiEMD = baseEMD(**ISSI_HMIVars)
    hmiEMD.plotSeparately()

    hmiEMD.plotTogether(
        showBox=None,
        gridRegions=(1, 2, True, True),
        yTickFrequency=[0, 6, 12, 18],
        xTickFrequency=[0, 12],
    )

    print(
        f"ISSI HMI CASE with {hmiEMD.multiCPU} CPUs and {hmiEMD.windDispParam} minutes of window displacement"
    )


if __name__ == "__main__":
    show = False
    ISSICase(show=show)
    # ISSIHMICase(show=show)

    # In situ
    # PSPSolOCase(show=show)
    # SolOEarth2020Case(show=show)
    # STAPSPCase(show=show)
