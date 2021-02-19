BASE_PATH = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/"

from sys import path

path.append(f"{BASE_PATH}Scripts/")

"""Main routine to compare remote and in-situ observations"""
from os import makedirs
from signalHelpers import compareTwoTS
import numpy as np

# TODO: Add signal comparing structure from ISSI work

# Import the following functions into the AnySpacecraft_data script
def comparePSPtoSolo(SolOSpc, PSPSpc, soloVars, pspVars):
    """
    Feed in the PSP Spacecraft and SolOSpc object
    """
    # Set header of directories
    general_directory = f"{BASE_PATH}unsafe/Resources/EMD_Data/"
    makedirs(general_directory, exist_ok=True)

    ### Directory structure
    # Specific folder to have all extracted datasets and plots
    mainDir = f"{general_directory}PSP_SolO/"
    makedirs(mainDir, exist_ok=True)

    # Set the Self and Other dataframe to those within the Spacecraft object
    dfSelf = SolOSpc.df[soloVars]
    dfSelf.columns = [f"Solo{i}" for i in soloVars]
    dfOther = PSPSpc.df[pspVars]
    dfOther.columns = [f"PSP{i}" for i in pspVars]

    dfSelf.fillna(method="pad")
    dfOther.fillna(method="pad")
    cadSelf = SolOSpc.obj_cad
    cadOther = PSPSpc.obj_cad

    labelSelf = SolOSpc.name

    compareTwoTS(
        dfSelf,
        dfOther,
        cadSelf,
        cadOther,
        labelSelf,
        winDispList=[60],
        corrThrPlotList=[np.arange(0.4, 1, 0.05)],
        PeriodMinMax=[1, 180],
        savePath=mainDir,
        useRealTime=True,
        expectedLocationList=False,
        detrend_box_width=200,
    )
