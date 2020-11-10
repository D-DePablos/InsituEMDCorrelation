#%%
import astropy.units as u
import astropy.constants as const
import astropy.coordinates as coord
import sunpy.coordinates.frames as frames
from heliopy import spice
from heliopy.data import spice as spicedata
from datetime import datetime, timedelta
import astropy.units as u

for kernel in ["psp", "solo"]:
    k = spicedata.get_kernel(kernel)
    spice.furnish(k)


def loc(dtime, body):
    traj = spice.Trajectory(body)
    traj.generate_positions([dtime], "Sun", "IAU_SUN")
    traj_coord = traj.coords
    traj_coord.representation_type = "spherical"
    return traj_coord


def spiral_correction(psp_coord, vsw):
    omega_sun = 14.713 * u.deg / u.d

    def delta_long(r):
        return omega_sun * (r - 2.5 * const.R_sun) / vsw

    psp_solar_lon = psp_coord.lon + delta_long(psp_coord.radius)
    psp_solar_surface = coord.SkyCoord(
        radius=2.49 * const.R_sun,
        lat=psp_coord.lat,
        lon=psp_solar_lon,
        frame=frames.HeliographicCarrington,
        obstime=psp_coord.obstime,
        representation_type="spherical",
    )
    return psp_solar_surface


def psp_xyz(dtime):
    psp = spice.Trajectory("SPP")
    psp.generate_positions([dtime], "Sun", "IAU_SUN")
    psp.change_units(u.au)
    return np.array([psp.x.value, psp.y.value, psp.z.value])[:, 0]


if __name__ == "__main__":
    sdate = datetime(2020, 2, 11)
    date = datetime.now()

    print(loc(sdate, "SPP"))