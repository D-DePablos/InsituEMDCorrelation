"""Helper functions"""
from typing import Type
from .SignalAndSfunc import Signal, SignalFunctions, transformTimeAxistoVelocity
import astropy.units as u
from collections import namedtuple
from datetime import timedelta
import matplotlib.dates as mdates
from PyEMD import EMD, Visualisation
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import makedirs
import warnings

warnings.filterwarnings("ignore")


locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)

Hmin = mdates.DateFormatter("%H:%M")
# Quick fix for cross-package import

emd = EMD()
vis = Visualisation()

# Named tuples
caseTuple = namedtuple(
    "caseTuple", ["dirExtension", "shortTimes", "longTimes"])
shortDFDic = namedtuple(
    "shortDFDic", ["df", "name", "cases", "paramList", "regionName", "kernelName"], defaults=(None,) * 6)
longDFDic = namedtuple(
    "longDFDic", ["df", "name", "kernelName", "accelerated", "speedSet"], defaults=(None, None, None, 1, None))

missingData = namedtuple("missingData", [
                         "shortMissing", "longMissing", "colour"], defaults=(None, None, "red"))

ColumnColours = {
    "Btotal": "pink",
    "B_R": "blue",
    "B_T": "green",
    "B_N": "red",
    "N_RPW": "green",
    "Mf": "purple",
    "V_R": "red",
    "plume": "pink",
    "chplume": "blue",
    "qsun": "green",
    "cbpoint": "purple",
    "chole": "red",
    "171": "blue",
    "193": "orange",
}
alphaWVL = {
    "94": 0.9,
    "171": 0.9,
    "193": 0.7,
    "211": 0.5,
    "HMI": 0.9,
    "Btotal": 0.9,
    "B_R": 0.7,
    "B_T": 0.7,
    "B_N": 0.7,
    "N_RPW": 0.9,
    "V_R": 0.9,
    "Mf": 0.9,
    "plume": 0.9,
    "chplume": 0.7,
    "qsun": 0.4,
    "cbpoint": 0.4,
    "chole": 0.4
}

titleDic = {
    "SolO_V_R": "Vsw",
    "SolO_T": "Tp",
    "SolO_N": "#p",
    "SolO_Mf": "Mass Flux",
    "PSP_Vr": "Vsw",
    "V_R": "Vsw",
    "BTotal": "Bt",
    "N": "#p",
    "T": "Tp",
    "Mf": "Mass Flux",
    "PSP_V_R": "Vsw",
    "PSP_T": "Tp",
    "PSP_Np": "#p",
    "PSP_N": "#p",
    "PSP_Mf": "Mass Flux",
    "PSP_Br": "Br",
    "PSP_B_R": "Br",
    "PSP_Btotal": "Bt",
    "B_R": "Br",
}

# Set general font size
plt.rcParams.update({'font.size': 22})

# Dictionary which contains relevant axis for a 9x9 grid for each of the regions
axDic = {
    "11": [0, 0],
    "12": [0, 1],
    "13": [0, 2],
    "16": [1, 0],
    "17": [1, 1],
    "18": [1, 2],
    "21": [2, 0],
    "22": [2, 1],
    "23": [2, 2],
    "open": [0],
    "bpoint": [0],
    "BASE": [0],
    "plume": [0, 0],
    "cbpoint": [0, 1],
    "chplume": [0, 2],
    "chole": [1, 0],
    "qsun": [1, 1],
}


# font = {"family": "DejaVu Sans", "weight": "normal", "size": 25}
# rc("font", **font)


def extractDiscreteExamples(Caselist, shortDuration, **kwargs):
    """Extracts discrete PSP - longObject pairs

    Args:
        Caselist ([type]): [description]
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
    longTimes = []
    matchTimes = []
    caseNames = []
    refLocations = []

    # Open each of the list dictionaries
    for case in Caselist:
        # Get the short start and end
        shortStart = case["shortTime"]
        shortEnd = shortStart + timedelta(hours=shortDuration)
        shortTimes.append((shortStart, shortEnd))

        # Get the match, which is used for reference later
        matchStart = case["matchTime"]

        # longObjectDurn gives reference to amount of longObject datapoints that match
        matchEnd = matchStart + timedelta(hours=case["shortDurn"])
        matchAvg = matchStart + (matchEnd - matchStart) / 2
        matchTimes.append((matchStart, matchEnd))

        if case["margin"] != 0:
            _startLong = matchStart - timedelta(hours=case["margin"] / 2)
            _endLong = matchStart + timedelta(hours=case["margin"] / 2)

        else:
            _startLong = None
            _endLong = None

        longTimes.append((_startLong, _endLong))

        # Get the specific case Name
        caseNames.append(case["caseName"])

        refLocations.append(
            _constructExpectedLocation(_times=(matchStart, matchEnd), **kwargs)
        )

    return shortTimes, longTimes, caseNames, refLocations


def caseCreation(
    shortTimes,
    longTimes,
    shortDuration,
    caseName,
    savePicklePath,
    shortDisplacement=None,
    forceCreate=False,
    firstRelevantLongTime=False,
    MARGIN=None,
):
    """Creates cases given times of short and long datasets

    Args:
        shortTimes (datetime): start SHORT end SHORT
        longTimes (datetime): start LONG end LONG
        shortDuration (int): duration of short WINDOW in hours
        caseName ([type]): CaseName  folder will take the format {caseName}_{startShort.day}_H{startShort.hour}
        savePicklePath ([type]): [description]
        shortDisplacement ([type], optional): [description]. Defaults to None.
        forceCreate (bool, optional): [description]. Defaults to False.
        firstRelevantLongTime (bool, optional): If needed to skip some time from long dataset. Defaults to False.

    Returns:
        [type]: [description]
    """
    import pickle

    # Attempt to load the file first
    if not forceCreate:
        try:
            with open(f"{savePicklePath}", "rb") as f:
                print("Loaded test cases.")
                cases = pickle.load(f)
                return cases

        except FileNotFoundError:
            pass

    from datetime import timedelta

    startSHORT, endSHORT = shortTimes
    startLONG, endLONG = longTimes

    baseLONG = endLONG - (endLONG - startLONG) / 2

    cases, i = [], 0
    _tShort = startSHORT

    shortDisplacement = (
        shortDuration if shortDisplacement == None else shortDisplacement
    )

    if shortDuration > shortDisplacement:
        refFactor = shortDuration
    else:
        refFactor = shortDisplacement
    # If displacement longer than duration it breaks
    while _tShort <= (endSHORT - timedelta(hours=refFactor)):
        _tShort = startSHORT + timedelta(hours=shortDisplacement) * i

        if firstRelevantLongTime != False:
            _matchTime = firstRelevantLongTime + \
                timedelta(hours=shortDisplacement) * i
        else:
            _matchTime = baseLONG

        cases.append(
            {
                "shortTime": _tShort,
                "matchTime": _matchTime,
                "shortDurn": shortDuration,
                "caseName": f"{caseName}_{_tShort.day}_T{_tShort.hour:02d}",
                "margin": MARGIN,  # +- some number of hours
            }
        )

        i += 1

    with open(f"{savePicklePath}", "wb") as f:
        pickle.dump(cases, f)
    return cases


def massFluxCalc(Vsw, Np):
    """
    Returns the mass flux
    Vsw: Solar wind speed in km/s
    Np: Solar wind proton density per cm3
    """
    import astropy.constants as cons
    import astropy.units as u

    mp = cons.m_p
    VxshortMS = (Vsw * (u.km / u.s)).to(u.m / u.s)
    NshortM3 = (Np * (u.cm ** (-3))).to(u.m ** (-3))
    return (NshortM3 * mp * VxshortMS).value


def getAvgRadius(SPCKernelName, times):
    """
    Find average radius for a time period
    """
    import heliopy.data.spice as spicedata
    import heliopy.spice as spice

    if SPCKernelName == "solo":
        spicedata.get_kernel("solo")
        sp_traj = spice.Trajectory("Solar Orbiter")

    elif SPCKernelName == "psp":
        spicedata.get_kernel("psp")
        sp_traj = spice.Trajectory("SPP")

    elif SPCKernelName == "sun":
        spicedata.get_kernel("helio_frames")
        sp_traj = spice.Trajectory("Sun")

    else:
        raise ValueError(
            f"{SPCKernelName} is not a valid spacecraft kernel, please choose one from ['solo'] "
        )
    sp_traj.generate_positions(times, "Sun", "IAU_SUN")  # Is in Km
    R = sp_traj.coords.radius

    mR = R.mean().to(u.AU).value

    return mR


def collect_dfs_npys(isDf, lcDic, region, base_folder, windDisp="60s", period="3 - 20"):
    def _find_corr_mat(
        region="chole",
        lcDic=lcDic,
        shortVar=193,
        insituDF=None,
        _base_folder="/home/diegodp/Documents/PhD/Paper_3/insituObject_SDO_EUI/unsafe/ISSI/SB_6789/",
        windDisp="60s",
        period="3 - 20",
    ):
        """Finds correlation matrices for all In-situ variables and a given wavelength

        Args:
            region (str): Which remote sensing region to explore
            shortVar (int): Which wavelength is relevant
            insituParams (list): In situ parameters to find correlation matrices for
        """
        resultingMatrices = {}
        matrixData = namedtuple(
            "data", "isData corrMatrix shortData shortTime")

        for isparam in insituDF.columns:
            # How do we get in situ data? From above function!
            if region != "":
                _subfolder = f"{_base_folder}*{shortVar}_{region}*/*{isparam}*/{windDisp}/{period[0]} - {period[1]}/"
            else:
                _subfolder = f"{_base_folder}*{shortVar}*/*{isparam}*/{windDisp}/{period[0]} - {period[1]}/"
            foundMatrix = glob(f"{_subfolder}IMF/Corr_matrix_all.npy")
            short_D = glob(f"{_subfolder}IMF/short*.npy")
            short_T = lcDic[shortVar].index
            resultingMatrices[f"{isparam}"] = matrixData(
                insituDF[f"{isparam}"],
                np.load(foundMatrix[0]),
                np.load(short_D[0]),
                short_T,
            )

        return resultingMatrices

    expandedWvlDic = {}
    # Select the correlation matrix for each insituObject variable, given region, given lcurve
    for shortVar in lcDic:
        # dataContainer should have all
        dataContainer = _find_corr_mat(
            shortVar=shortVar,
            lcDic=lcDic,
            insituDF=isDf,
            _base_folder=base_folder,
            region=region,
            windDisp=windDisp,
            period=period,
        )

        # namedtuple("data", "isData corrMatrix shortData shortTime")
        expandedWvlDic[f"{shortVar}"] = dataContainer
    return expandedWvlDic


def plot_imfs(time_array, vector, imfs, title, savePath, width, show):

    # Find optimal spacing
    spacing = 0  # Start  0 base spacing
    for curr_imf in imfs:
        # Update spacing
        if spacing < abs(curr_imf.max() - curr_imf.mean()):
            spacing = abs(curr_imf.max() - curr_imf.mean())
        if spacing < abs(curr_imf.mean() - curr_imf.min()):
            spacing = abs(curr_imf.mean() - curr_imf.min())

    ax = plt.figure(figsize=(10, 9))
    plt.suptitle(f"{title}")

    for index, curr_imf in enumerate(imfs):
        ax.add_subplot(len(imfs) + 1, 1, len(imfs) - index)
        ax_frame = plt.gca()

        if type(time_array) is np.ndarray:
            plt.plot(time_array, curr_imf, color="black")
        else:
            plt.plot(curr_imf, color="black")
        leg = plt.legend([f"IMF {index}"], loc=1)
        leg.get_frame().set_linewidth(0.0)
        plt.ylim(curr_imf.mean() - spacing - width,
                 curr_imf.mean() + spacing + width)
        ax_frame.axes.get_xaxis().set_visible(False)

    ax.add_subplot(len(imfs), 1, len(imfs))

    if type(time_array) is np.ndarray:
        plt.plot(time_array, vector, color="blue")
        plt.plot(time_array, imfs[-1], color="red")
    else:
        plt.plot(vector, color="blue")
        plt.plot(imfs[-1], color="red")

    plt.legend((["Signal", "Residual"]), frameon=False, loc=1, ncol=2)
    plt.savefig(f"{savePath}.pdf")

    if show:
        plt.show()

    plt.close("all")


def emdAndCompareCases(
    shortDFDic,
    longDFSimpleDic,
    saveFolder,
    PeriodMinMax=[5, 20],
    detrendBoxWidth=None,
    showFig=True,
    corrThrPlotList=np.arange(0.65, 1, 0.05),
    multiCPU=1,
    inKind=False,
    windDispParam=10,
):
    """
    Perform EMD and Compare a short and a Long dataframe
    Args:
    shortDFDic: A named tuple containing the shortDF class, including "name", "df", "regions", and "cases"
    longDF: The long dataframe within a named tuple, including "name", "df"
    saveFolder: Folder to save within (will then create a directory for each caseName)
    PeriodMinMax: Minimum and Maximum IMF periodicities to be considered
    speedLim: Upper and lower bounds on Speed for reference. Should be 2 numbers which do not change
    multiCPU: Either 'None' if not multiprocessing, or the number of CPUs to use
    """
    _dfLong = longDFSimpleDic.df
    _dfLong.columns = [f"{longDFSimpleDic.name}_{i}" for i in _dfLong.columns]
    cadLong = (_dfLong.index[1] - _dfLong.index[0]).seconds

    # Analyse multiple Cases at the same time
    def multiEMD(splitTimes,
                 splitIndices,
                 caseNamesList,
                 _dfShort,
                 longDF,
                 saveFolder,
                 cads,
                 corrThrPlotList,
                 PeriodMinMax,
                 detrendBoxWidth,
                 showFig,
                 splitLongTimes=None,
                 ):
        """EMD for a set of splitTimes and Indices
        Optimised for multiprocessing

        Args:
            splitTimes (np.array): short DF times that are selected
            splitIndices ([type]): indices used for dirNames
            caseNamesList ([type]): [description]
            _dfShort ([type]): [description]
            longDF ([type]): namedTuple with '.name' and '.df'
            saveFolder ([type]): [description]
            cads ([type]): [description]
            corrThrPlotList ([type]): [description]
            PeriodMinMax ([type]): [description]
            detrendBoxWidth ([type]): [description]
            showFig ([type]): [description]
            loSpeed ([type]): [description]
            hiSpeed ([type]): [description]
        """

        cadShort, cadLong = cads
        _dfLong = longDF.df.copy()

        for i, shortTimes in enumerate(splitTimes):
            dirName = f"{caseNamesList[splitIndices[i]]}"
            print(f"Starting {dirName} -- to save in {saveFolder}")
            _dfShortCut = _dfShort[shortTimes[0]: shortTimes[1]]
            if splitLongTimes is not None:
                _dfLongCut = _dfLong[splitLongTimes[i][0]:splitLongTimes[i][1]]
            else:
                _dfLongCut = _dfLong
            _specificFolder = f"{saveFolder}{dirName}/"

            _expectedLocationList = False

            compareTS(
                dfSelf=_dfShortCut,
                dfOther=_dfLongCut,
                cadSelf=cadShort,
                cadOther=cadLong,
                labelOther=longDF.name,
                winDispList=[cadShort * windDispParam],
                corrThrPlotList=corrThrPlotList,
                PeriodMinMax=PeriodMinMax,
                showLocationList=False,
                filterPeriods=True,
                savePath=_specificFolder,
                useRealTime=True,
                expectedLocationList=_expectedLocationList,
                detrend_box_width=detrendBoxWidth,
                showFig=showFig,
                renormalize=False,
                showSpeed=False,
                SPCKernelName=longDF.name.lower(),  # Should ensure that using SPC name
                inKind=inKind,
            )

    # For each of the "wavelengths" or "shortParameters"
    for dfCase in shortDFDic:
        (
            shortTimesList,
            longTimesList,
            caseNamesList,
            refLocations,
        ) = extractDiscreteExamples(
            dfCase.cases,
            shortDuration=dfCase.cases[0]["shortDurn"],
        )
        _dfShort = dfCase.df.copy()
        _dfShort.columns = [f"{dfCase.name}_{i}" for i in _dfShort.columns]
        cadShort = (_dfShort.index[1] - _dfShort.index[0]).seconds

        assert (
            cadLong == cadShort
        ), "Cadence of short object not equal to cad. of long Object"

        # Old functionality

        import multiprocessing
        # Take shortTimesList and transform to array?
        # shortTimesList
        indices = np.arange(len(shortTimesList))
        splitTimes = np.array_split(np.array(shortTimesList), multiCPU)
        splitLongTimes = np.array_split(np.array(longTimesList), multiCPU)
        splitIndices = np.array_split(indices, multiCPU)

        # Now create a process that handles a range of these

        procs = []

        for _splitTimes, _splitIndices, _splitLongTimes in zip(splitTimes, splitIndices, splitLongTimes):
            proc = multiprocessing.Process(
                target=multiEMD,
                args=(
                    _splitTimes,
                    _splitIndices,
                    caseNamesList,
                    _dfShort,
                    longDFSimpleDic,
                    saveFolder,
                    [cadShort, cadLong],
                    corrThrPlotList,
                    PeriodMinMax,
                    detrendBoxWidth,
                    showFig,
                    _splitLongTimes,
                ),
            )
            procs.append(proc)
            try:
                proc.start()
            except Exception as e:
                raise e

        for proc in procs:
            proc.join()


def superSummaryPlotGeneric(shortDFDic,
                            longDFDic,
                            unsafeEMDataPath,
                            PeriodMinMax,
                            corrThrPlotList,
                            showFig,
                            baseEMDObject,
                            showBox=None,
                            gridRegions=False,
                            shortName="",
                            longName="",
                            missingData=None,
                            inKind=False,
                            skipParams=[],
                            ):
    """
    Calculates and plots a superSummaryPlot (shows all short params,
    at all times, correlated against a single long param)

    # THESE DATAFRAMES ARE EXPECTED TO BE ON SAME CADENCE ALREADY
    shortDFDic: namedTuple (df, name, paramList, regionName, kernelName)
    longDFDic: namedTuple (df, kernelName, accelerated)
    """
    assert(shortDFDic[0].df.index[1] - shortDFDic[0].df.index[0]
           == longDFDic.df.index[1] - longDFDic.df.index[0])

    missingData = missingData or None

    matchingColumn = [s for s in longDFDic.df.columns if "V_R" in s]
    # If we have a column called V_R, get max, min, mean values
    if longDFDic.speedSet != None:
        HISPEED, LOSPEED, AVGSPEED = longDFDic.speedSet

    elif longDFDic.speedSet == None:
        HISPEED, LOSPEED, AVGSPEED = (
            int(longDFDic.df[matchingColumn].max() / longDFDic.accelerated),
            int(longDFDic.df[matchingColumn].min() / longDFDic.accelerated),
            int(longDFDic.df[matchingColumn].mean() / longDFDic.accelerated),
        )

        if HISPEED == None:
            raise ValueError("Could not extract Speeds")

    figName = f"{shortDFDic[0].cases[0]['caseName'].partition('/')[0]}/"
    figName += "accelerated" if longDFDic.accelerated != 1 else "constant"
    figName += f"_P{PeriodMinMax[0]}_{PeriodMinMax[1]}"

    (shortTimesList, longTimesList, caseNamesList, _) = extractDiscreteExamples(
        shortDFDic[0].cases, shortDuration=shortDFDic[0].cases[0]["shortDurn"])

    # Create a list with all cases that should be split up
    allCases = []
    for caseName, shortTimes, longTimes in zip(caseNamesList, shortTimesList, longTimesList):
        dirExtension = f"{caseName}"
        allCases.append(
            caseTuple(dirExtension, shortTimes, longTimes)
        )

    shortInfo = shortDFDic[0]
    shortParams = shortInfo.paramList
    shortRegion = shortInfo.regionName
    shortKernel = shortInfo.kernelName
    # Each of the parameters in long dataset
    if skipParams != []:
        for param in skipParams:
            longDFDic.df.pop(param)

    for longObjParam in longDFDic.df.columns:
        plot_super_summary(
            allCasesList=allCases,
            # Use all of the long dataset
            # longSpan=(
            #     longDFDic.df.index[0].to_pydatetime(),
            #     longDFDic.df.index[-1].to_pydatetime()
            # ),
            shortParamList=shortParams,
            longObjectParam=longObjParam,
            regions=shortRegion,
            unsafeEMDDataPath=unsafeEMDataPath,
            period=PeriodMinMax,
            SPCKernelName=shortKernel,
            otherObject=longDFDic.kernelName,
            speedSet=(HISPEED, LOSPEED, AVGSPEED),
            showFig=showFig,
            figName=figName,
            corrThrPlotList=corrThrPlotList,
            showBox=showBox,
            gridRegions=gridRegions,
            # Cadence of long dataset
            cadence=f"{(longDFDic.df.index[1] - longDFDic.df.index[0]).seconds}s",
            shortName=shortName,
            longName=longName,
            inKind=inKind,
            baseEMDObject=baseEMDObject,
        )


def compareTS(
    dfSelf,
    dfOther,
    cadSelf,
    cadOther,
    labelOther,
    winDispList=[60],
    PeriodMinMax=[1, 180],
    filterPeriods=True,
    savePath=None,
    useRealTime=False,
    expectedLocationList=False,
    showLocationList=False,
    detrend_box_width=200,
    showFig=True,
    renormalize=False,
    showSpeed=False,
    LOSPEED=255,
    HISPEED=285,
    SPCKernelName=None,
    corrThrPlotList=np.arange(0.75, 0.901, 0.05),
    inKind=False,
):
    """
    Takes two dataframes sampled at same cadence

    dfSelf is a dataframe
    dfOther is another dataFrame
    cadSelf, cadOther for cadence
    labelOther is the label to be shown on second dataset

    winDispList is a list of window displacements, in seconds
    corrThrPlotList is

    labelOther = Which label to show for other dataset
    """

    import random

    assert savePath != None, "Please set savePath to store relevant arrays"
    # For all of the lightcurves
    varList = list(dfOther)
    random.shuffle(varList)
    for varOther in varList:
        otherPath = f"{savePath}../{labelOther}/{varOther}/"
        makedirs(otherPath, exist_ok=True)

        nanPercentage = np.isnan(
            dfOther[varOther]).sum() / len(dfOther[varOther])
        ignoreNANs = True if nanPercentage < 0.15 else False
        signalOther = Signal(
            cadence=cadOther,
            custom_data=dfOther[varOther],
            name=varOther,
            time=dfOther.index,
            saveFolder=otherPath,
            ignoreNANs=ignoreNANs,
        )

        signalOther.detrend(box_width=detrend_box_width)

        # Add functionality, filter and generate IMFs
        otherSigFunc = SignalFunctions(
            signalOther,
            filterIMFs=True,
            PeriodMinMax=PeriodMinMax,
            corrThrPlotList=corrThrPlotList,
        )

        # TODO: Get rid of NAN values when under a certain percentage!

        # For each of the in-situ variables
        for varSelf in dfSelf:
            # If inKind is true, only compare equivalent values
            if not inKind or varSelf.split("_", 1)[1] == varOther.split("_", 1)[1]:
                selfPath = f"{savePath}{varSelf}/{varOther}/"
                makedirs(selfPath, exist_ok=True)

                dataSelf = dfSelf[varSelf]
                # calculate percentage of NANs
                nanPercentage = np.isnan(dataSelf).sum() / len(dataSelf)
                ignoreNANs = True if nanPercentage < 0.3 else False
                signalSelf = Signal(
                    cadence=cadSelf,
                    custom_data=dataSelf,
                    saveFolder=selfPath,
                    name=varSelf,
                    ignoreNANs=ignoreNANs,
                )
                signalSelf.detrend(box_width=detrend_box_width)
                selfSigFunc = SignalFunctions(
                    signalSelf,
                    filterIMFs=True,
                    PeriodMinMax=PeriodMinMax,
                    corrThrPlotList=corrThrPlotList,
                )

                for windowDisp in winDispList:
                    otherWinFolder = f"{otherPath}{windowDisp}s/"
                    selfWinFolder = (
                        f"{selfPath}{windowDisp}s/{PeriodMinMax[0]} - {PeriodMinMax[1]}/"
                    )
                    otherSigFunc.saveFolder = otherWinFolder
                    selfSigFunc.saveFolder = selfWinFolder

                    selfSigFunc.generate_windows(
                        other=otherSigFunc,
                        windowDisp=windowDisp,
                        useRealTime=useRealTime,
                        filterPeriods=filterPeriods,
                        renormalize=renormalize,
                    )
                    selfSigFunc.plot_all_results(
                        other=otherSigFunc,
                        Label_long_ts=varOther,
                        plot_heatmaps=False,
                        savePath=f"{otherSigFunc.saveFolder}corr_matrix/",
                        bar_width=None,
                        useRealTime=useRealTime,
                        expectedLocationList=expectedLocationList,
                        showLocationList=showLocationList,
                        showFig=showFig,
                        showSpeed=showSpeed,
                        SPCKernelName=SPCKernelName,
                        LOSPEED=LOSPEED,
                        HISPEED=HISPEED,
                        # ffactor=4 / 3,
                    )


def plot_super_summary(
    allCasesList,
    shortParamList,
    longObjectParam,
    regions,
    unsafeEMDDataPath,
    period,
    SPCKernelName,
    baseEMDObject,
    corrThrPlotList=np.arange(0.65, 1, 0.05),
    cadence="60s",
    speedSet=(300, 200, 250),
    showFig=False,
    figName="",
    gridRegions=True,
    insituArrayFreq="1min",
    otherObject="Sun",
    showBox=None,
    shortName="",
    longName="",
    inKind=False,
):
    """Plots a "super" summary with info about all selected regions
    Does not take dataframes as input, instead finds the data through


    Args:
        allCasesList (List): All the cases in a list, each index has some info
        longSpan (tuple): Start and end time of all possible LONG data
        shortParamList (tuple): List of wavelengths which are to be studied
        regions (List): List of regions that should be plotted. Good to be square
        unsafeEMDDataPath (String (Path)): The path under which all the numpy arrays are found
        period (Tuple): [description]
        SPCKernelName ([type], optional): SpacecraftKernel name for psp or solo. Defaults to None.
        showFig (bool, optional): [description]. Defaults to False.
        gridRegions = (nrows, ncols, sharex, sharey)
        showBox = ([X0, XF], [Y0, YF]) - in datetime
    """
    highSpeed, lowSpeed, _ = speedSet
    from matplotlib.lines import Line2D

    # Gapless subplots figure
    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
    # ax = [fig.add_subplot(2,2,i+1) for i in range(4)]

    # for a in ax:
    #     a.set_xticklabels([])
    #     a.set_yticklabels([])
    #     a.set_aspect('equal')

    # Makes Figure here
    if gridRegions == True:
        # This way makes sure there is space
        nrowsCols = np.sqrt(len(regions))
        if nrowsCols > int(nrowsCols):
            nrows = int(nrowsCols)
            ncols = int(nrowsCols) + 1
        else:
            nrows = int(nrowsCols)
            ncols = int(nrowsCols)
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(20, 10), sharex=True, sharey=True
        )
    else:
        # TODO: Add some text for each subplot in the grid
        nrows = gridRegions[0]
        ncols = gridRegions[1]
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=gridRegions[2],
            sharey=gridRegions[3],
            figsize=(18, 12),
        )

    used_ax = []
    for i, region in enumerate(regions):

        if type(axs) is np.ndarray:
            row, col = axDic[region]
            ax = axs[row, col]

        else:
            try:
                ax = axs[i]
            except TypeError:
                ax = axs

        used_ax.append(ax)

        # Take the starting point for AIA Times
        shortStartTimes, longTimes = [], []
        for case in allCasesList:
            shortStartTimes.append(case.shortTimes[0])
            longTimes.append(case.longTimes)

        listTimesSameSpeed = []
        listTimesSameSpeed_LOW = []

        # Define a function which allows for multiprocessing call
        # Use some of the shortStartTimes
        for index, (TshortDF) in enumerate(shortStartTimes):
            insituStTime, insituEndTime = longTimes[index]
            longARRAY = pd.date_range(
                insituStTime, insituEndTime, freq=insituArrayFreq)
            base_path = f"{unsafeEMDDataPath}{allCasesList[index].dirExtension}/"

            # Dataframe with all times, all dotSizes, for each wavelength
            dfDots = pd.DataFrame({})

            # shortParamList not necessarily equal to short df
            for _shortVar in shortParamList:
                if not inKind or _shortVar == longObjectParam.split("_", 1)[1]:
                    wvlPath = f"{base_path}{region}_{_shortVar}/"
                    try:
                        corr_matrix = np.load(
                            f"{wvlPath}{longObjectParam}/{cadence}/{period[0]} - {period[1]}/IMF/Corr_matrix_all.npy"
                        )
                    except FileNotFoundError:
                        try:
                            wvlPath = f"{base_path}{_shortVar}_{region}/"
                            corr_matrix = np.load(
                                f"{wvlPath}{longObjectParam}/{cadence}/{period[0]} - {period[1]}/IMF/Corr_matrix_all.npy"
                            )
                        except FileNotFoundError:
                            raise FileNotFoundError(
                                f"The correlation Matrix cannot be found at {wvlPath}{longObjectParam}/{cadence}/{period[0]} - {period[1]}/IMF/Corr_matrix_all.npy")

                    # List of how big dots should be in comparison.
                    # Smallest dot is 0, when correlation does not reach 0.70.
                    # Does not grow more if multiple matches at same corr thr.
                    dotSizeList = []
                    midpoint_time = insituStTime
                    midpointTimes = []

                    # Find time and necessary size of dots
                    for height in range(len(corr_matrix[0, 0, :, 0])):
                        if midpoint_time > insituEndTime:
                            break

                        pearson = corr_matrix[:, :, height, 0]
                        spearman = corr_matrix[:, :, height, 1]
                        spearman[spearman == 0] = np.nan
                        valid = corr_matrix[:, :, height, 2]

                        # Create the midpointTimes list to act as index
                        midpoint = corr_matrix[0, 0, height, 3]
                        midpoint_time = insituStTime + \
                            timedelta(seconds=midpoint)
                        midpointTimes.append(midpoint_time)

                        # Transform to real time
                        # Get rid of borders
                        pearson[pearson == 0] = np.nan
                        spearman[spearman == 0] = np.nan

                        # Pearson and Spearman Valid
                        pvalid = pearson[valid == 1]
                        rvalid = spearman[valid == 1]
                        # After getting rid of some pearson and spearman values
                        # Necessary to count how many are above each given threshold
                        dotSize = 0
                        for corrIndex, corr_thr in enumerate(corrThrPlotList):
                            _number_high_pe = len(
                                pvalid[np.abs(pvalid) >= corr_thr])
                            # corr_locations[height, index, 1] = _number_high_pe
                            _number_high_sp = len(
                                rvalid[np.abs(rvalid) >= corr_thr])
                            # corr_locations[height, index, 2] = _number_high_sp

                            if _number_high_pe > 0:
                                dotSize += 1

                        dotSizeList.append(dotSize)

                    dfDots[f"{_shortVar}"] = dotSizeList

            # After all shortVars Set the index of the Dots dataframe to midPoints
            dfDots.index = midpointTimes

            # Plot inside each of the squares
            ax.plot(
                longARRAY,
                np.repeat(TshortDF, len(longARRAY)),
                linewidth=1.2,
                color="black",
                alpha=0.5,
            )

            Vaxis = transformTimeAxistoVelocity(
                longARRAY,
                originTime=TshortDF,
                SPCKernelName=SPCKernelName,
                ObjBody=otherObject,
            )

            # Create lines to delineate highest speeds
            # Highest speeds
            closest_index = (np.abs(Vaxis - highSpeed)).argmin()
            closest_time = longARRAY[closest_index]
            listTimesSameSpeed.append(closest_time)

            # Lower speeds
            closest_index_LOW = (np.abs(Vaxis - lowSpeed)).argmin()
            closest_time_LOW = longARRAY[closest_index_LOW]
            listTimesSameSpeed_LOW.append(closest_time_LOW)

            # Set x axis to normal format
            longLocator = mdates.HourLocator([0])
            longFormatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(longLocator)
            ax.xaxis.set_major_formatter(longFormatter)

            # Locators for the y axis (short dframe)
            shortlocator = mdates.HourLocator([0, 12])
            shortformatter = mdates.ConciseDateFormatter(shortlocator)
            ax.yaxis.set_major_locator(shortlocator)
            ax.yaxis.set_major_formatter(shortformatter)

            # For each of the variables in the dots array
            for _shortVar in dfDots.columns:
                alphaList = [
                    alphaWVL[_shortVar] if x > 0 else 0
                    for x in dfDots[_shortVar].values
                ]
                _msize = 100 * (dfDots[_shortVar].values) ** 2
                if len(corrThrPlotList) == 1:
                    _msize = 100

                # TODO: Add crosshairs
                ax.scatter(
                    x=dfDots.index,
                    y=np.repeat(TshortDF, len(dfDots.index)),
                    s=_msize,
                    alpha=alphaList,
                    c=ColumnColours[f"{_shortVar}"],
                )

        # Compared to plotting lines, this does not plot outside of necessary area
        ax.fill_betweenx(
            shortStartTimes,
            listTimesSameSpeed,
            listTimesSameSpeed_LOW,
            color="orange",
            alpha=0.2,
        )
        ax.set_ylabel(f"{shortName}")
        ax.set_xlabel(f"{longName}")

        if showBox != None:
            box_X0, box_X1 = showBox[0]
            box_Y0, box_Y1 = showBox[1]

            rec = plt.Rectangle((box_X0, box_Y0), box_X1 - box_X0,
                                box_Y1 - box_Y0, fc="blue", ec=None, alpha=0.4)
            ax.add_patch(rec)

    # Custom legend to show circle sizes and colouring
    legend_elements = []
    for j, corrThr in enumerate(corrThrPlotList):
        _mkrsize = 12 + j * 8
        _legendElement = Line2D(
            [listTimesSameSpeed_LOW[0]],
            [TshortDF],
            marker="o",
            color="w",
            label=f"{corrThr:.02f}",
            markerfacecolor="k",
            markersize=_mkrsize,
        )

        legend_elements.append(_legendElement)

    # Plot all of the dots from the dataframe
    for j, var in enumerate(dfDots.columns):
        _mkrsize = 12
        _legendElement = Line2D(
            [listTimesSameSpeed_LOW[0]],
            [TshortDF],
            marker="o",
            color="w",
            label=f"{var}",
            markerfacecolor=ColumnColours[var],
            markersize=_mkrsize,
        )

        legend_elements.append(_legendElement)

    if type(axs) == np.ndarray:
        for row in axs:
            for _ax in row:
                if _ax not in used_ax:
                    fig.delaxes(_ax)
    # Or that the cadence of Short dataset is correct
    fig.legend(handles=legend_elements, prop={"size": 20})
    longParamLegible = longObjectParam if longObjectParam not in titleDic else titleDic[
        longObjectParam]
    fig.suptitle(
        f" Correlating against {longParamLegible} | in yellow {highSpeed} - {lowSpeed} km/s | Allowed IMF Periods: {period[0]} - {period[1]} min.")

    # Plot the datasets if inKind:
    if inKind:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # HERE
        shortDataSet = baseEMDObject.shortDFDics[0].df[longObjectParam.split('_', 1)[
            1]]
        longDataSet = baseEMDObject.longDFDic.df[longObjectParam]

        # inset axes along bottom
        x_axis_df = inset_axes(
            axs,
            width="100%",  # width = 30% of parent_bbox
            height=1,  # height : 1 inch
            loc="lower center",
        )
        plt.axis("off")

        y_axis_df = inset_axes(
            axs,
            width="5%",
            height="100%",
            loc="center left",)
        plt.axis("off")

        # duplicate x axis
        x_axis_df.plot(longDataSet.values,
                       color=ColumnColours[longObjectParam.split('_', 1)[1]])
        y_axis_df.plot(shortDataSet.values, shortDataSet.index,
                       color=ColumnColours[longObjectParam.split('_', 1)[1]])

        # TODO: Check that the axis limits are correct.

    plt.savefig(f"{unsafeEMDDataPath}{figName}_{longObjectParam}_Summary.png")

    if showFig:
        plt.show()

    plt.close()

    print(f"Saved {longObjectParam} to {unsafeEMDDataPath}")


def new_plot_format(
    dfInsitu,
    lcDic,
    regions,
    base_folder,
    period,
    corrThrPlotList=np.arange(0.65, 1, 0.05),
    addResidual=True,
    addEMDLcurves=True,
    SPCKernelName=None,
    spcSpeeds=(200, 300),
    showFig=False,
    windDisp="60s",
):
    """
    This new plot format requires rest of plots to have been made and correlations calculated!

    lcDic contains each of the relevant wavelengths and its dataframe
    """

    # Adds the bar plots
    def doBarplot(
        ax: plt.axis,
        RSDuration,
        ISTime: np.ndarray,
        corr_matrix: np.ndarray,
        barColour: str,
    ):

        # Prepare the array with matches
        corr_locations = np.ndarray(
            (len(corr_matrix[0, 0, :, 0]), len(corrThrPlotList), 3)
        )

        axBar = ax.twinx()
        for height in range(len(corr_matrix[0, 0, :, 0])):
            pearson = corr_matrix[:, :, height, 0]
            spearman = corr_matrix[:, :, height, 1]
            spearman[spearman == 0] = np.nan
            valid = corr_matrix[:, :, height, 2]

            midpoint = corr_matrix[0, 0, height, 3]
            midpoint_time = ISTime[0] + timedelta(seconds=midpoint)  # Ax xaxis
            barchartTime = midpoint_time.to_pydatetime()

            # Get rid of borders
            pearson[pearson == 0] = np.nan
            spearman[spearman == 0] = np.nan

            # Pearson and Spearman Valid
            pvalid = pearson[valid == 1]
            rvalid = spearman[valid == 1]

            for index, corr_thr in enumerate(corrThrPlotList):
                _number_high_pe = len(pvalid[np.abs(pvalid) >= corr_thr])
                corr_locations[height, index, 1] = _number_high_pe
                _number_high_sp = len(rvalid[np.abs(rvalid) >= corr_thr])
                corr_locations[height, index, 2] = _number_high_sp

            for index, corrLabel in enumerate(corrThrPlotList):
                pearson_array = corr_locations[height, :, 1]
                if pearson_array[index] > 0:
                    # Copy the axis
                    axBar.bar(
                        barchartTime,
                        corrLabel,
                        width=RSDuration,
                        color=barColour,
                        edgecolor=None,
                        alpha=0.35,
                        zorder=1,
                    )

        corrThrLimits = []
        for i, value in enumerate(corrThrPlotList):
            if np.mod(i, 2) == 0:
                corrThrLimits.append(value)

        axBar.set_yticks(corrThrLimits)  # Ensure ticks are good
        axBar.set_ylim(corrThrPlotList[0], 1.01)  # Allows for any correlation
        axBar.set_ylabel("Corr. value")
        axBar.grid("both")

        return corr_matrix[:, :, 0, 2]  # Return valid Arrays per height

    # Create expandedWvlDic for each region
    regionDic = {}
    for region in regions:
        expandedWvlDic = collect_dfs_npys(
            region=region,
            isDf=dfInsitu,
            lcDic=lcDic,
            base_folder=base_folder,
            period=period,
            windDisp=windDisp,
        )
        regionDic[region] = expandedWvlDic

    # Once opened up each of the regions, do plots
    for region in regionDic:
        # Below is how you pull vars. Can use "isData corrMatrix shortData shortTime"
        # regionDic[f"{region}"]["94"]["N"].isData
        r = regionDic[f"{region}"]

        # Extract the WVL and IS params
        shortParamList = list(r.keys())
        _nshortVar = len(shortParamList)
        insituList = list(r[f"{shortParamList[0]}"].keys())
        n_insitu = len(insituList)

        nplots = _nshortVar if _nshortVar >= n_insitu else n_insitu
        RSDuration = (
            lcDic[shortParamList[0]].index[-1] -
            lcDic[shortParamList[0]].index[0]
        )

        # Create one figure per region, per aiaTime
        fig, axs = plt.subplots(
            nrows=nplots,
            ncols=2,
            sharex="col",
            figsize=(4 * nplots, 8),
            constrained_layout=True,
        )
        fig.suptitle(
            f'SolO: {lcDic[shortParamList[0]].index[0].strftime(format="%Y-%m-%d %H:%M")}',
            size=25,
        )

        # Delete all axis. If used they are shown
        if len(axs) >= 4:
            for ax in axs:
                ax[0].set_axis_off()
                ax[1].set_axis_off()
        else:
            for ax in axs:
                ax.set_axis_off()

        i, j = 0, 0
        WVLValidity = {}

        # In situ plots
        for i, isVar in enumerate(insituList):
            # Open up the isInfo from e.g., first WVL
            isTuple = r[f"{shortParamList[0]}"][f"{isVar}"]
            axIS = axs[i, 0] if len(axs) >= 4 else axs[0]
            axIS.set_axis_on()
            fig.add_subplot(axIS)
            plt.plot(isTuple.isData, color="black")
            plt.ylabel(isVar, fontsize=20)

            # Add the speed information for first plot
            if isVar == "V_R" or isVar == "Vr":
                # Using start of AIA observations
                vSWAxis = transformTimeAxistoVelocity(
                    isTuple.isData.index,
                    originTime=isTuple.shortTime[0].to_pydatetime(),
                    SPCKernelName=SPCKernelName,
                )
                axV = axIS.twiny()
                axV.plot(vSWAxis, isTuple.isData, alpha=0)
                axV.invert_xaxis()

                # Add a span between low and high values
                axV.axvspan(
                    xmin=spcSpeeds[0],
                    xmax=spcSpeeds[1],
                    ymin=0,
                    ymax=1,
                    alpha=0.3,
                    color="orange",
                )
            # Plot bars within In situ chart and get IMF validity per WVL
            for shortVar in shortParamList:
                corrMatrix = r[f"{shortVar}"][f"{isVar}"].corrMatrix
                validIMFsMatrix = doBarplot(
                    axIS,
                    ISTime=isTuple.isData.index,
                    RSDuration=RSDuration,
                    corr_matrix=corrMatrix,
                    barColour=ColumnColours[f"{shortVar}"],
                )

                if i == n_insitu - 1:
                    WVLValidity[f"{shortVar}"] = validIMFsMatrix[:, 0]

            if i == n_insitu - 1:

                axIS.xaxis.set_major_locator(locator)
                axIS.xaxis.set_major_formatter(formatter)

        # Plot all lightcurves
        wvlDataLabel = "Det. "
        for j, shortVar in enumerate(shortParamList):
            wvlTime = r[f"{shortVar}"][f"{insituList[0]}"].shortTime
            wvlEMD = r[f"{shortVar}"][f"{insituList[0]}"].shortData
            axRE = axs[(nplots - _nshortVar) + j,
                       1] if len(axs) >= 4 else axs[1]
            axRE.set_axis_on()
            axRE.yaxis.set_visible(True)
            axRE.yaxis.tick_right()
            fig.add_subplot(axRE)

            if addResidual:
                # Plot residual, then add it to all cases
                wvlEMD[:-1] = wvlEMD[:-1] + wvlEMD[-1]
                plt.plot(
                    wvlTime,
                    wvlEMD[-1],
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    label="Residual",
                )
                wvlDataLabel = ""

            # Plot original data
            plt.plot(
                wvlTime,
                wvlEMD[0],
                color=ColumnColours[f"{shortVar}"],
                alpha=0.7,
            )

            plt.title(
                wvlDataLabel + f" {shortVar}",
                color=ColumnColours[f"{shortVar}"],
                fontsize=20,
            )

            if addEMDLcurves:
                for k, wvlemd in enumerate(wvlEMD[1:-1]):
                    if int(WVLValidity[f"{shortVar}"][k]) == 1:
                        plt.plot(
                            wvlTime,
                            wvlemd,
                            alpha=0.9,
                        )

            if j == _nshortVar - 1:
                axRE.xaxis.set_major_formatter(Hmin)
                axRE.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
                axRE.xaxis.set_major_locator(mdates.HourLocator(interval=1))

        saveFolder = f"{base_folder}0_Compressed/{period[0]} - {period[1]}/"
        makedirs(saveFolder, exist_ok=True)
        if showFig:
            plt.show()
        print(f"Region {region}")
        plt.savefig(
            f"{saveFolder}{region}{'' if not addResidual else 'with_residual'}.png"
        )
        plt.close()


def plot_variables(df, BOXWIDTH=200):
    """Plots detrended variables found within a dataframe"""
    from astropy.convolution import convolve, Box1DKernel

    plt.figure(figsize=(20, 20))
    for index, parameter in enumerate(list(df)):
        signal = df[parameter].copy()
        csignal = convolve(signal, Box1DKernel(BOXWIDTH), boundary="extend")
        smp_signal = 100 * (signal - csignal) / csignal
        plt.subplot(5, 1, index + 1)
        plt.plot(smp_signal, color="black")
        plt.ylabel(parameter)
    plt.show()
    plt.close()
