"""
We get the Parker and Solar Orbiter orbital data
"""


import heliopy.data.spice as spicedata
import heliopy.spice as spice
from datetime import datetime, timedelta
import astropy.units as u
import numpy as np

# Load kernels
solo_kernel = spicedata.get_kernel("solo")
spice.furnish(solo_kernel)
solo = spice.Trajectory("Solar Orbiter")

psp_kernel = spicedata.get_kernel("psp")
spice.furnish(psp_kernel)
psp = spice.Trajectory("PSP")

# Times
starttime = datetime(2018, 12, 12)
endtime = datetime(2020, 12, 12)
times = []
while starttime < endtime:
    times.append(starttime)
    starttime += timedelta(days = 1)

solo.generate_positions(times, "Sun", "ECLIPJ2000")
psp.generate_positions(times, "Sun", "ECLIPJ2000")