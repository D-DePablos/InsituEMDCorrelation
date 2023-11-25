import pandas as pd
import cdflib
from glob import glob


def extractDF(CDFfolder, dfVars, timeIndex="Epoch", info=False, resample=False):
    firstFile = True
    for fileName in sorted(glob(CDFfolder + "*.cdf")):
        _cdf = cdflib.CDF(fileName)

        if info and firstFile:
            raise ValueError(
                f"Set info to false \n \n Zvars = {_cdf.cdf_info()['zVariables']}"
            )

        _t = cdflib.cdfepoch().to_datetime(_cdf[timeIndex])

        varDic = {}

        for var in dfVars:
            if (
                var != "VEL"
                and var != "vp_moment_RTN"
                and var != "psp_fld_l2_mag_RTN_1min"
                and var != "V_RTN"
                and var != "B_RTN"
                and var != "BFIELDRTN"
                and var != "BFIELD"
            ):
                varDic[var] = _cdf[var]

            elif var == "VEL" or var == "vp_moment_RTN" or var == "V_RTN":
                for i, subVar in enumerate(["V_R", "V_T", "V_N"]):
                    varDic[subVar] = _cdf[var][:, i]

            elif (
                var == "psp_fld_l2_mag_RTN_1min"
                or var == "B_RTN"
                or var == "BFIELDRTN"
                or var == "BFIELD"
            ):
                for i, subVar in enumerate(["B_R", "B_T", "B_N"]):
                    # Breaks here with psp_fld taking name
                    varDic[subVar] = _cdf[var][:, i]

        if firstFile == True:
            _df = pd.DataFrame(varDic, index=_t)
        else:
            _df = _df.append(pd.DataFrame(varDic, index=_t))

        firstFile = False
    if resample != False:
        _df = _df.resample(resample).mean()
    return _df


# Functions to explore some specific cdfs


def openSDAplasma():
    path = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/unsafe/Resources/STA_Data/plasma/"
    """['Epoch', 'FILTER_VALUE', 'BFIELDRTN', 'BTOTAL', 'CART_COMPNO', 'B_RTN_LABL_1', 
    'HAE', 'HEE', 'HEEQ', 'CARR', 'RTN', 'R', 'CART_LABL_1', 'CART_LABL_2', 'CART_LABL_3', 
    'Np', 'Vp', 'Tp', 'Vth', 'Vr_Over_V_RTN', 'Vt_Over_V_RTN', 'Vn_Over_V_RTN', 'Vp_RTN', 'Entropy', 'Beta', 
    'Total_Pressure', 'Cone_Angle', 'Clock_Angle', 'Magnetic_Pressure', 'Dynamic_Pressure', 'HAE_LABL_1', 
    'HEE_LABL_1', 'HEEQ_LABL_1', 'CARR_LABL_1', 'RTN_LABL_1']
    """
    sdaVars = ["BFIELDRTN", "BTOTAL", "Np", "Tp", "Vp_RTN"]
    df = extractDF(CDFfolder=path, dfVars=sdaVars)
    print(df)


def openSDAMag():
    """['Epoch', 'BFIELD', 'MAGFLAGUC', 'CART_LABL_1', 'FILTER_VALUE']"""
    path = "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/unsafe/Resources/STA_Data/mag/"
    sdaVars = ["BFIELD"]
    df = extractDF(CDFfolder=path, dfVars=sdaVars)
    print(df)


def openSOLOMag():
    """['Epoch', 'BFIELD', 'MAGFLAGUC', 'CART_LABL_1', 'FILTER_VALUE']"""
    import matplotlib.pyplot as plt

    path = "/Users/ddp/Downloads/"
    sdaVars = ["B_RTN"]
    df = extractDF(CDFfolder=path, dfVars=sdaVars)
    print(df)
    df["B_R"].plot()
    plt.show()


if __name__ == "__main__":
    openSDAMag()
