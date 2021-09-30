# Set up UNSAFE_EMD_DATA_PATH: global variable
from Plots.AnySpacecraft_data import Spacecraft
from astropy import units as u
from datetime import datetime, timedelta
import numpy as np
from EMD.importsProj3.signalAPI import (
    compareTS,
    new_plot_format,
    plot_super_summary,
    extractDiscreteExamples,
)
from os import makedirs
from collections import namedtuple
from sys import path

BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"
path.append(f"{BASE_PATH}Scripts/")


# Different imports on Project 2


"""Main routine to compare remote and in-situ observations"""
SUBPATH = "encounter6_Parker/"
UNSAFE_EMD_DATA_PATH = f"{BASE_PATH}unsafe/EMD_Data/{SUBPATH}"
makedirs(UNSAFE_EMD_DATA_PATH, exist_ok=True)

# Set parameters here
objCad = 60  # cadence in seconds for comparisons
PERIODMINMAX = [5, 22]  # The period might be better if longer

shortRegs = [""]  # Set to empty string

DELETE = False  # I believe this is not working at all as intended
SHOWSPEED = False

# Show figures as they are created
SHOWFIG = False

# Add residual to non-super summary?
ADDRESIDUAL = False

# Filter by periodicity
FILTERP = True

# Plot all in-situ variables?
PLOT_ALL_TOGETHER = True

# Box that's shown above the plots
SHOWBOX = ((datetime(2020, 9, 27, 0), datetime(2020, 9, 27, 5)),
           (datetime(2020, 10, 1, 20, ), datetime(2020, 10, 2, 0, 13)))
# Plot summary? should be done after plotting together
SUPER_SUMMARY_PLOT = True
# Whether to accelerate speed (relevant for backmapping, coloured columns)
accelerated = (1)

with open(
    "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/EMD/cases/cases.pickle",
    "rb",
) as f:
    import pickle

    cases = pickle.load(f)

MARGINHOURSLONG = cases[0]["MARGINHOURSLONG"]

# Import the following functions into the AnySpacecraft_data script


def comparePSPtoSOLO(
    shortObj,
    longObj,
    shortVars,
    longVars,
    shortTimes,
    longTimes,
    shortName,
    longName,
    shortCad,
    longCad,
    objDirExt,
    corrThrPlotList=np.arange(0.65, 1, 0.05),
    expectedLocationList=False,
    PeriodMinMax=[1, 20],
    filterPeriods=False,
    showFig=True,
    renormalize=False,
    DETREND_BOX_WIDTH=None,
    HISPEED_BMAPPING=None,
    LOSPEED_BMAPPING=None,
):
    """Compares PSP to SolO Measurements

    Args:
        shortObj (Object with df): Short object (usually SolO)
        longObj (Object with df): Long object (usually PSP)
        shortVars ([type]): [description]
        longVars ([type]): [description]
        shortTimes ([type]): [description]
        longTimes ([type]): [description]
        shortName ([type]): [description]
        longName ([type]): [description]
        shortCad ([type]): [description]
        longCad ([type]): [description]
        objDirExt ([type]): [description]
        corrThrPlotList ([type], optional): [description]. Defaults to np.arange(0.65, 1, 0.05).
        expectedLocationList (bool, optional): [description]. Defaults to False.
        PeriodMinMax (list, optional): [description]. Defaults to [1, 20].
        filterPeriods (bool, optional): [description]. Defaults to False.
        showFig (bool, optional): [description]. Defaults to True.
        renormalize (bool, optional): [description]. Defaults to False.
        DETREND_BOX_WIDTH ([type], optional): [description]. Defaults to None.
        HISPEED_BMAPPING ([type], optional): [description]. Defaults to None.
        LOSPEED_BMAPPING ([type], optional): [description]. Defaults to None.
    """
    assert HISPEED_BMAPPING != None, "No High speed set"
    assert LOSPEED_BMAPPING != None, "No Low speed set"

    # Set header of directories
    makedirs(UNSAFE_EMD_DATA_PATH, exist_ok=True)

    # Directory structure
    # Specific folder to have all extracted datasets and plots
    mainDir = f"{UNSAFE_EMD_DATA_PATH}{objDirExt}/"
    makedirs(mainDir, exist_ok=True)

    # Set the Self and Other dataframe to those within the Spacecraft object
    dfShort = shortObj.df[shortVars]
    # Rename the columns
    dfShort.columns = [f"{shortName}_{i}" for i in shortVars]

    dfLong = longObj.df[longVars]
    dfLong.columns = [f"{longName}_{i}" for i in longVars]

    # Cut down the self and other dataseries
    dfShort = dfShort[shortTimes[0]: shortTimes[1]]
    dfLong = dfLong[longTimes[0]: longTimes[1]]
    cadSelf = shortCad
    cadOther = longCad

    compareTS(
        dfShort,
        dfLong,
        cadSelf,
        cadOther,
        labelOther=longName,
        winDispList=[60],
        PeriodMinMax=PeriodMinMax,
        filterPeriods=filterPeriods,
        savePath=mainDir,
        useRealTime=True,
        expectedLocationList=expectedLocationList,
        detrend_box_width=DETREND_BOX_WIDTH,
        showFig=showFig,
        renormalize=renormalize,
        showSpeed=SHOWSPEED,
        LOSPEED=HISPEED_BMAPPING,
        corrThrPlotList=corrThrPlotList,
        HISPEED=LOSPEED_BMAPPING,
    )


def deriveAndPlotSeparatelyPSPE6(
    longObjectVars=["N", "T", "V_R", "Mf", "Btotal"],
    shortObjectVars=["N", "T", "V_R", "Mf", "Btotal"],
):
    """
    PSP variables
    V_R	V_T	V_N	N	T	swe_flag	B_R	B_T	B_N	fld_flag	Time
    """

    longObject = Spacecraft(
        name="PSP_Scaled_e6",
        cadence_obj=objCad,
    )

    longObject.df = longObject.df.interpolate()  # Fill gaps

    # Velocities are sometimes modified with 4/3 factor
    HISPEED_BMAPPING, LOSPEED_BMAPPING, MEAN = (
        int(longObject.df["V_R"].max() / accelerated),
        int(longObject.df["V_R"].min() / accelerated),
        int(longObject.df["V_R"].mean() / accelerated),
    )

    """
    Solo Vars (Short TS)
    
    """
    # Short Object (Using Spacecraft)
    shortObject = Spacecraft(
        name="SolO_Scaled_e6",
        cadence_obj=objCad,
    )
    # Interpolate after forming lc object
    shortObject.df = shortObject.df.interpolate()

    # We set a margin around original obs.
    (
        shortTimesList,
        longTimesList,
        caseNamesList,
        refLocations,
    ) = extractDiscreteExamples(
        cases,
        margin=MARGINHOURSLONG,
    )

    print(
        f"""
            Will be creating {len(shortTimesList)} short windows. \n
            Variables to be used - short: {shortObject.name}: [{shortObjectVars}] \n 
            long: {longObject.name}: [{longObjectVars}] \n
         """
    )

    for index, shortTimes in enumerate(shortTimesList):

        # Minimise number of prints
        dirName = f"""{caseNamesList[index]}"""
        print(f"Creating and Saving to {dirName}")

        comparePSPtoSOLO(
            shortObj=shortObject,
            longObj=longObject,
            shortVars=shortObjectVars,
            longVars=longObjectVars,
            shortTimes=shortTimes,
            longTimes=longTimesList[index],
            shortName="SolO",
            longName="PSP",
            shortCad=objCad,
            longCad=objCad,
            objDirExt=dirName,
            filterPeriods=FILTERP,
            PeriodMinMax=PERIODMINMAX,
            showFig=SHOWFIG,
            expectedLocationList=[refLocations[index]],
            renormalize=False,
            HISPEED_BMAPPING=HISPEED_BMAPPING,
            LOSPEED_BMAPPING=LOSPEED_BMAPPING,
            corrThrPlotList=np.arange(0.8, 1.01, 0.1),
        )


# Combined Plot
def combinedPlot(
    shortParamList=[],
    speedSet=None,
    superSummaryPlot=False,
    longObjectZOOM=False,
    corrThrPlotList=np.arange(0.7, 1, 0.05),
    showBox=None,
):
    """
    speedSet: (MAX, MIN, AVG)
    longObjectZOOM: {"start_time": datetime, "end_time": datetime, "stepMinutes": Time step for zoomed version}
    """
    # The shortDFDic collects
    shortDFDic = {}
    for _shortParam in shortParamList:
        shortDFDic[f"{_shortParam}"] = Spacecraft(
            name="SolO_Scaled_e6",
            cadence_obj=objCad,
        )
        shortDFDic[f"{_shortParam}"].df = shortDFDic[f"{_shortParam}"].df.interpolate()
        try:
            del shortDFDic[f"{_shortParam}"].df["Unnamed: 0"]
        except KeyError:
            pass

    # Long Object will be PSP measurements as more continuous
    longObject = Spacecraft(
        name="PSP_Scaled_e6",
        cadence_obj=objCad,
    )

    if longObjectZOOM != False:
        longObject.zoom_in(**longObjectZOOM)

    longObject.df = longObject.df.interpolate()  # Fill gaps
    # Velocities are modified with 4/3 factor. Gives slightly better idea
    HISPEED_BMAPPING, LOSPEED_BMAPPING, soloAVG = (
        (
            int(longObject.df["V_R"].max() / accelerated),
            int(longObject.df["V_R"].min() / accelerated),
            int(longObject.df["V_R"].mean() / accelerated),
        )
        if speedSet == (None, None, None)
        else speedSet
    )

    # Variables for PSP (long)
    longObjectVars = generalVars
    longObject.df = longObject.df[longObjectVars]

    figName = "accelerated" if accelerated == 4 / 3 else "constant"

    # We set a margin around original obs.
    (shortTimesList, longTimesList, caseNamesList, _,) = extractDiscreteExamples(
        cases,
        margin=MARGINHOURSLONG,
    )

    # When necessary to make summary of all summaries
    if superSummaryPlot:

        # Possibly this is not great
        soloStendTotal = (
            longObject.df.index[0].to_pydatetime(),
            longObject.df.index[-1].to_pydatetime(),
        )

        allCases = []
        Casetuple = namedtuple(
            "Case", ["dirExtension", "isStend_t", "rsStend_t"])
        for index, shortTimes in enumerate(shortTimesList):
            _isT = longTimesList[index]
            dirExtension = f"{caseNamesList[index]}"
            allCases.append(
                Casetuple(
                    dirExtension, (_isT[0], _isT[1]
                                   ), (shortTimes[0], shortTimes[1])
                )
            )

        # Figure out whether to show yellow bar - DONE
        longObject.df.columns = [
            "PSP_" + param for param in longObject.df.columns]

        for longObjectParam in longObject.df.columns:
            plot_super_summary(
                allCasesList=allCases,
                longSpan=soloStendTotal,
                shortParamList=shortParamList,
                longObjectParam=longObjectParam,
                regions=["SolO"],
                unsafeEMDDataPath=UNSAFE_EMD_DATA_PATH,
                period=PERIODMINMAX,
                SPCKernelName="solo",
                speedSuper=HISPEED_BMAPPING,
                speedSuperLow=LOSPEED_BMAPPING,
                speedAVG=soloAVG,
                showFig=SHOWFIG,
                figName=figName,
                otherObject="psp",
                corrThrPlotList=corrThrPlotList,
                showBox=showBox,
            )

    else:
        for index, shortTimes in enumerate(shortTimesList):
            # Need to cut up dataframes
            isTimes = longTimesList[index]
            dfLongCut = longObject.df[isTimes[0]: isTimes[1]]
            dfLongCut = dfLongCut[longObjectVars]

            shortDFDicCut = {}
            for _shortParam in shortDFDic:
                shortDFDicCut[f"{_shortParam}"] = (
                    shortDFDic[_shortParam].df[shortTimes[0]
                        : shortTimes[1]].copy()
                )

            dirExtension = f"{caseNamesList[index]}"
            base_folder = f"{UNSAFE_EMD_DATA_PATH}{dirExtension}/"
            new_plot_format(
                dfInsitu=dfLongCut,
                lcDic=shortDFDicCut,
                regions=shortRegs,
                base_folder=base_folder,
                period=PERIODMINMAX,
                addResidual=ADDRESIDUAL,
                SPCKernelName="psp",
                spcSpeeds=(LOSPEED_BMAPPING, HISPEED_BMAPPING),
                showFig=SHOWFIG,
            )


if __name__ == "__main__":
    generalVars = [
        "V_R",
        "Btotal",
        "N",
        "Mf",
        "T",
    ]
    shortParamList = ["Btotal", "B_R", "V_R", "Mf", "N_RPW"]
    if not PLOT_ALL_TOGETHER:
        deriveAndPlotSeparatelyPSPE6(
            longObjectVars=generalVars,
            shortObjectVars=shortParamList,
        )

    else:
        corrThrPlotList = (
            np.arange(0.6, 1, 0.05)
            if SUPER_SUMMARY_PLOT == False
            else np.arange(0.75, 0.94, 0.1)
        )
        combinedPlot(
            shortParamList=shortParamList,
            speedSet=(
                300,
                200,
                250,
            ),  # Speeds must be negative as short dataset no longer SUN!
            superSummaryPlot=SUPER_SUMMARY_PLOT,
            longObjectZOOM={
                "start_time": datetime(2020, 9, 25),
                "end_time": datetime(2020, 10, 1),
                "stepMinutes": 1,
                "extractOrbit": False,
            },
            corrThrPlotList=corrThrPlotList,
            showBox=SHOWBOX,
        )
