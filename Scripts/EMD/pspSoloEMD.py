# Set up UNSAFE_EMD_DATA_PATH: global variable
from sys import path

BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"
path.append(f"{BASE_PATH}Scripts/")

from collections import namedtuple
from os import makedirs

# Different imports on Project 2
from EMD.importsProj3.signalHelpers import (
    compareTS,
    new_plot_format,
    plot_super_summary,
)
import numpy as np
from datetime import datetime, timedelta
from astropy import constants as const
from astropy import units as u

from Scripts.Plots.AnySpacecraft_data import Spacecraft

"""
# TODO: Should ensure that we can swap out of SolO short / PSP 
"""


"""Universal Constants"""
mp = const.m_p

"""Main routine to compare remote and in-situ observations"""
UNSAFE_EMD_DATA_PATH = f"{BASE_PATH}unsafe/EMD_Data/"
makedirs(UNSAFE_EMD_DATA_PATH, exist_ok=True)

# Set parameters here
objCad = 60  # cadence in seconds for comparisons
LONGPARAMLIST = ["B_R"]
PERIODMINMAX = [3, 20]  # The period might be better if longer

shortRegs = [""]

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

# Plot summary? should be done after plotting together
SUPER_SUMMARY_PLOT = False
accelerated = (
    1  # Whether to accelerate speed (relevant for backmapping, coloured columns)
)

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
    expectedLocationList=False,
    PeriodMinMax=[1, 20],
    filterPeriods=False,
    delete=DELETE,
    showFig=True,
    renormalize=False,
    DETREND_BOX_WIDTH=None,
    HISPEED_BMAPPING=None,
    LOSPEED_BMAPPING=None,
):
    """
    Feed in Objects which have a .df property that is a pandas dataframe
    Choose some variables to correlate for short, long dataset
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
    dfShort.columns = [f"{shortName}_{i}" for i in shortVars]  # Rename the columns

    dfLong = longObj.df[longVars]
    dfLong.columns = [f"{longName}_{i}" for i in longVars]

    # Cut down the self and other dataseries
    dfShort = dfShort[shortTimes[0] : shortTimes[1]]
    dfLong = dfLong[longTimes[0] : longTimes[1]]
    cadSelf = shortCad
    cadOther = longCad

    compareTS(
        dfShort,
        dfLong,
        cadSelf,
        cadOther,
        labelOther=longName,
        winDispList=[60],
        corrThrPlotList=np.arange(0.65, 1, 0.05),
        PeriodMinMax=PeriodMinMax,
        filterPeriods=filterPeriods,
        savePath=mainDir,
        useRealTime=True,
        expectedLocationList=expectedLocationList,
        detrend_box_width=DETREND_BOX_WIDTH,
        delete=delete,
        showFig=showFig,
        renormalize=renormalize,
        showSpeed=SHOWSPEED,
        LOSPEED=HISPEED_BMAPPING,
        HISPEED=LOSPEED_BMAPPING,
    )


def extractDiscreteExamples(Caselist, margin, shortDuration=1.5, **kwargs):
    """Extracts discrete PSP - longObject pairs

    Args:
        Caselist ([type]): [description]
        margin ([type]): [description]
        pspDuration (int, optional): [description]. Defaults to 1.
        noColumns (bool, optional): Whether to skip plotting backmapped time columns
        Kwargs_construct = _exp_loc_color, _exp_loc_label
    """

    def _constructExpectedLocation(
        _times, _exp_loc_color="blue", _exp_loc_label="BBMatch"
    ):
        """Construct the Expected Location dic

        Args:
            _times ([type]): Tuple of two datetimes, start and end
            _exp_loc_color (str, optional): [description]. Defaults to "orange".
            label (str, optional): [description]. Defaults to "BBMatch".

        Returns:
            Dictionary with proper formatting
        """

        return {
            "start": _times[0],
            "end": _times[1],
            "color": _exp_loc_color,
            "label": _exp_loc_label,
        }

    shortTimes = []
    matchTimes = []
    longTimes = []
    caseNames = []
    refLocations = []

    # Open each of the list dictionaries
    for case in Caselist:
        # Get the PSP start and end
        shortStart = case["shortTime"]
        shortEnd = shortStart + timedelta(hours=shortDuration)
        shortTimes.append((shortStart, shortEnd))

        # Get the match, which is used for reference later
        matchStart = case["matchTime"]

        # longObjectDurn gives reference to amount of longObject datapoints that match
        matchEnd = matchStart + timedelta(hours=case["shortDurn"])
        matchAvg = matchStart + (matchEnd - matchStart) / 2
        matchTimes.append((matchStart, matchEnd))

        # Get the Solar Orbiter measurements
        longObjectStart = matchAvg - timedelta(hours=margin)
        longObjectEnd = matchAvg + timedelta(hours=margin)
        longTimes.append((longObjectStart, longObjectEnd))

        # Get the specific case Name
        caseNames.append(case["caseName"])

        refLocations.append(
            _constructExpectedLocation(_times=(matchStart, matchEnd), **kwargs)
        )

    return shortTimes, longTimes, caseNames, refLocations


def first_DeriveAndPlotSeparately(
    longObjectVars=["N", "T", "V_R", "Mf", "Btotal"],
    shortObjectVars=["N", "T", "V_R", "Mf", "Btotal"],
):
    # For each of the parameters in LONGPARAMLIST

    """
    PSP variables
    V_R	V_T	V_N	N	T	swe_flag	B_R	B_T	B_N	fld_flag	Time
    """
    longObject = Spacecraft(
        name="PSPpriv_e6",
        mid_time=datetime(2020, 10, 2),
        margin=timedelta(days=5),
        cadence_obj=objCad,
    )

    longObject.df = longObject.df.interpolate()  # Fill gaps
    # Velocities are sometimes modified with 4/3 factor
    HISPEED_BMAPPING, LOSPEED_BMAPPING, MEAN = (
        int(longObject.df["V_R"].max() / accelerated),
        int(longObject.df["V_R"].min() / accelerated),
        int(longObject.df["V_R"].mean() / accelerated),
    )

    # Calculate mass flux for long
    Vx = (longObject.df["V_R"].values * (u.km / u.s)).to(u.m / u.s)
    N = (longObject.df["N"].values * (u.cm ** (-3))).to(u.m ** (-3))
    longObject.df["Mf"] = (N * mp * Vx).value

    longObject.df["Btotal"] = np.sqrt(
        longObject.df["B_R"] ** 2
        + longObject.df["B_T"] ** 2
        + longObject.df["B_N"] ** 2
    )

    """
    Solo Vars (Short TS)
    V_R	V_T	V_N	N	T	solo_plasma_flag	B_R	B_T	B_N	MAG_FLAG	Time
    """
    # Short Object (Using Spacecraft)
    shortObject = Spacecraft(
        name="SolOpriv_e6",
        mid_time=datetime(2020, 10, 3),
        margin=timedelta(days=5),
        cadence_obj=objCad,
    )
    shortObject.df = shortObject.df.interpolate()  # Interpolate after forming lc object

    # Calculate mass flux & Btotal short df
    Vxshort = (shortObject.df["V_R"].values * (u.km / u.s)).to(u.m / u.s)
    Nshort = (shortObject.df["N"].values * (u.cm ** (-3))).to(u.m ** (-3))
    shortObject.df["Mf"] = (Nshort * mp * Vxshort).value

    shortObject.df["Btotal"] = np.sqrt(
        shortObject.df["B_R"] ** 2
        + shortObject.df["B_T"] ** 2
        + shortObject.df["B_N"] ** 2
    )
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
            Will be creating {len(shortTimesList)} windows. \n
            Variables to be used will be short: {shortObject.name}: [{shortObjectVars}] and long: {longObject.name}: [{longObjectVars}] \n
            """
    )

    for index, shortTimes in enumerate(shortTimesList):

        dirName = f"""{caseNamesList[index]}"""

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
            delete=DELETE,
            showFig=SHOWFIG,
            expectedLocationList=[refLocations[index]],
            renormalize=False,
            HISPEED_BMAPPING=HISPEED_BMAPPING,
            LOSPEED_BMAPPING=LOSPEED_BMAPPING,
        )


# Combined Plot
def combinedPlot(
    shortParamList=[], speedSet=None, superSummaryPlot=False, longObjectZOOM=False
):
    """
    speedSet: (MAX, MIN, AVG)
    longObjectZOOM: {"start_time": datetime, "end_time": datetime, "stepMinutes": Time step for zoomed version}
    """
    # The shortDFDic collects
    shortDFDic = {}
    for _shortParam in shortParamList:
        shortDFDic[f"{_shortParam}"] = Spacecraft(
            name="SolOpriv_e6",
            mid_time=datetime(2020, 10, 3),
            margin=timedelta(days=5),
            cadence_obj=objCad,
        )
        shortDFDic[f"{_shortParam}"].df = shortDFDic[f"{_shortParam}"].df.interpolate()
        try:
            del shortDFDic[f"{_shortParam}"].df["Unnamed: 0"]
        except KeyError:
            pass

    # Long Object will be PSP measurements as more continuous
    longObject = Spacecraft(
        name="PSPpriv_e6",
        mid_time=datetime(2020, 10, 2),
        margin=timedelta(days=5),
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

    # Calculate mass flux and Btotal
    Vx = (longObject.df["V_R"].values * (u.km / u.s)).to(u.m / u.s)
    mp = const.m_p
    N = (longObject.df["N"].values * (u.cm ** (-3))).to(u.m ** (-3))
    longObject.df["Mf"] = (N * mp * Vx).value

    longObject.df["Btotal"] = np.sqrt(
        longObject.df["B_R"] ** 2
        + longObject.df["B_T"] ** 2
        + longObject.df["B_N"] ** 2
    )

    # Variables for PSP (long)
    longObjectVars = generalVars
    longObject.df = longObject.df[longObjectVars]

    figName = "accelerated" if accelerated == 4 / 3 else "constant"

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

    # When necessary to make summary of all summaries
    if superSummaryPlot:

        # Possibly this is not great
        soloStendTotal = (
            longObject.df.index[0].to_pydatetime(),
            longObject.df.index[-1].to_pydatetime(),
        )

        shortParamList = shortParamList

        allCases = []
        Casetuple = namedtuple("Case", ["dirExtension", "isStend_t", "rsStend_t"])
        for index, shortTimes in enumerate(shortTimesList):
            _isT = longTimesList[index]
            dirExtension = f"{caseNamesList[index]}"
            allCases.append(
                Casetuple(
                    dirExtension, (_isT[0], _isT[1]), (shortTimes[0], shortTimes[1])
                )
            )

        # Figure out whether to show yellow bar - DONE

        longObject.df.columns = ["PSP_" + param for param in longObject.df.columns]

        for longObjectParam in longObject.df.columns:
            plot_super_summary(
                allCasesList=allCases,
                longSpan=soloStendTotal,
                shortParamList=shortParamList,
                longObjectParam=longObjectParam,
                regions=["SolO"],
                # gridRegions=[1, 1, True, True],
                unsafeEMDDataPath=UNSAFE_EMD_DATA_PATH,
                period=PERIODMINMAX,
                SPCKernelName="solo",
                speedSuper=HISPEED_BMAPPING,
                speedSuperLow=LOSPEED_BMAPPING,
                speedAVG=soloAVG,
                showFig=SHOWFIG,
                figName=figName,
                otherObject="psp",
            )

    else:
        for index, shortTimes in enumerate(shortTimesList):
            # Need to cut up dataframes
            isTimes = longTimesList[index]
            dfLongCut = longObject.df[isTimes[0] : isTimes[1]]
            dfLongCut = dfLongCut[longObjectVars]

            shortDFDicCut = {}
            for _shortParam in shortDFDic:
                shortDFDicCut[f"{_shortParam}"] = (
                    shortDFDic[_shortParam].df[shortTimes[0] : shortTimes[1]].copy()
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
    generalVars = ["V_R", "B_R", "Btotal", "Mf", "T"]
    if not PLOT_ALL_TOGETHER:

        first_DeriveAndPlotSeparately(
            longObjectVars=generalVars, shortObjectVars=["B_R"]
        )

    else:
        combinedPlot(
            shortParamList=["B_R"],
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
        )