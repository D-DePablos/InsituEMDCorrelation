import pandas as pd
import cdflib
from glob import glob


def extractDF(CDFfolder, vars, timeIndex="Epoch", info=False, resample=False):
    firstFile = True
    for fileName in sorted(glob(f"{CDFfolder}*.cdf")):
        _cdf = cdflib.CDF(fileName)

        if info and firstFile:
            raise ValueError(
                f"Set info to false \n \n Zvars = {_cdf.cdf_info()['zVariables']}"
            )

        _t = cdflib.cdfepoch().to_datetime(_cdf[timeIndex])

        varDic = {}

        for var in vars:
            if (
                var != "VEL"
                and var != "vp_moment_RTN"
                and var != "psp_fld_l2_mag_RTN_1min"
                and var != "B_RTN"
                and var != "V_RTN"
            ):
                varDic[var] = _cdf[var]

            elif var == "VEL" or var == "vp_moment_RTN" or var == "V_RTN":
                for i, subVar in enumerate(["V_R", "V_T", "V_N"]):
                    varDic[subVar] = _cdf[var][:, i]

            elif var == "psp_fld_l2_mag_RTN_1min" or var == "B_RTN":
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
