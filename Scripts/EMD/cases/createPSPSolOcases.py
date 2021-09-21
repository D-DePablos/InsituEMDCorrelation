"""
Generate the cases for all possible SolO - SHORT times (every hour)
"""

from datetime import datetime, timedelta
import pickle

MARGINHOURSLONG = 60

# According to Telleni et al., 2020 Sept 27 03:15-04:45 is best
# For SolO we have Oct 1st 21:34 -> 23:04 UT

# The approach cross-correlates 1.5 hour periods of Btotal and checks origin region
startLONG = datetime(2020, 9, 24, 12)
endLONG = datetime(2020, 10, 5)


def main():
    # The Solar Orbiter (short) data is complete between the two times used here
    SHORT_base = datetime(2020, 10, 1, 18, 30)
    SHORT_max = datetime(2020, 10, 3, 11, 53)
    SHORT_dt = 1.5  # How many hours to advance to create new case (should be length of short dataset)

    # Around the base case, MARGINHOURSLONG is utilised
    LONG_base = endLONG - (endLONG - startLONG) / 2

    cases = []
    i = 0
    tSHORT = SHORT_base
    while tSHORT <= (SHORT_max - timedelta(hours=1)):
        tSHORT = SHORT_base + timedelta(hours=SHORT_dt) * i
        tLONG = LONG_base

        cases.append(
            {
                "shortTime": tSHORT,  # Equivalent to AIA time
                "matchTime": tLONG,  # Time centered for match
                "shortDurn": 1,
                "caseName": f"SolO_{tSHORT.day}_T{tSHORT.hour:02d}",
                "MARGINHOURSLONG": MARGINHOURSLONG,
            }
        )

        i += 1

    with open(
        "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/EMD/cases/cases.pickle",
        "wb",
    ) as f:
        print("Saving cases")
        pickle.dump(cases, f)


def caseCreation(
    shortTimes,
    longTimes,
    shortDuration,
    caseName,
    shortDisplacement=None,
    MarginHours=24,
    savePicklePath=None,
):
    import pickle
    from datetime import timedelta

    startSHORT, endSHORT = shortTimes
    startLONG, endLONG = longTimes

    baseLONG = endLONG - (endLONG - startLONG) / 2

    cases, i = [], 0
    _tShort = startSHORT

    shortDisplacement = (
        shortDuration if shortDisplacement == None else shortDisplacement
    )
    while _tShort <= (endSHORT - timedelta(hours=shortDuration)):
        _tShort = startSHORT + timedelta(hours=shortDisplacement) * i

        cases.append(
            {
                "shortTime": _tShort,
                "matchTime": baseLONG,
                "shortDurn": shortDuration,
                "caseName": f"{caseName}_{_tShort.day}_T{_tShort.hour:02d}",
                "MARGINHOURSLONG": MarginHours,
            }
        )

        i += 1

    with open(f"{savePicklePath}", "wb") as f:
        pickle.dump(cases, f)
    return cases


def alterCases():
    """
    Cases like "hump" from dave
    """
    startLONG = datetime(2020, 9, 30)
    endLONG = datetime(2020, 10, 3)

    # The Solar Orbiter (short) data is complete between the two times used here
    SHORT_base = datetime(2020, 10, 4, 3)
    SHORT_max = datetime(2020, 10, 4, 12)
    SHORT_dt = 10  # How many hours to advance to create new case (should be length of short dataset)

    # Around the base case, MARGINHOURSLONG is utilised
    LONG_base = endLONG - (endLONG - startLONG) / 2

    cases = []
    i = 0
    tSHORT = SHORT_base
    while i < 1:
        tSHORT = SHORT_base + timedelta(hours=SHORT_dt) * i
        tLONG = LONG_base

        cases.append(
            {
                "shortTime": tSHORT,  # Equivalent to AIA time
                "matchTime": tLONG,  # Time centered for match
                "shortDurn": 9,
                "caseName": f"altCase_SolO_{tSHORT.day}_T{tSHORT.hour:02d}",
                "MARGINHOURSLONG": MARGINHOURSLONG,
            }
        )

        i += 1

    with open(
        "/home/diegodp/Documents/PhD/Paper_2/InsituEMDCorrelation/Scripts/EMD/cases/altCases.pickle",
        "wb",
    ) as f:
        print("Saving cases")
        pickle.dump(cases, f)


def testCaseCreation():
    """
    Tests that cases are created successfully and deletes evidence
    """
    testCase = {
        "shortTimes": (datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 23)),
        "longTimes": (datetime(2020, 2, 1), datetime(2020, 2, 10)),
        "shortDuration": 1,
        "caseName": "TEST",
        "shortDisplacement": 1,
        "MarginHours": 10,
        "savePicklePath": "/home/diegodp/Downloads/case.pickle",
    }

    cases = caseCreation(**testCase)
    assert len(cases) == 24, "Length not equal to expected"


if __name__ == "__main__":
    # main()
    # alterCases()
    testCaseCreation()
