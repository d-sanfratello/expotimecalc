import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from astropy.visualization import time_support
from astropy.visualization import quantity_support

from src.location import Location
from src.skylocation import SkyLocation
from src.time import Time
from src import Versor

from src import Tsidday
from src import eq2000


time_support(scale='utc', format='iso', simplify=True)
quantity_support()


class Observation:
    def __init__(self, location, obstime=None, target=None):
        if not isinstance(location, Location):
            raise TypeError("Must be of type `src.location.Location`.")
        if not isinstance(obstime, Time):
            raise TypeError("Must be of type `src.time.Time` or `astropy.time.Time`.")
        if not isinstance(target, SkyLocation):
            raise TypeError("Must be of type `src.skylocation.SkyLocation`.")

        self.location = location
        self.obstime = obstime
        self.target = target

        self.target.observe_at_date(self.obstime, copy=False)

        self.zenithJ2000 = Versor(ra=0., dec=self.location.lat.rad, unit='rad')\
            .rotate('z', self.sidereal_day(self.target.epoch) + eq2000.GMST.rad + self.location.lon, unit='rad')

        self.zenith = self.zenith_at_date(self.obstime)

        self.target_ha = self.calculate_ha(self.target, self.location, self.obstime)
        self.target_azimuth = self.calculate_az(self.target, self.location, self.obstime)

        self.target_alt = self.calculate_alt()

    def zenith_at_date(self, obstime):
        return self.zenithJ2000.rotate('z', self.sidereal_day(obstime), unit='rad', copy=True)

    def sidereal_day(self, obstime, epoch_time=None):
        return ((2*np.pi/Tsidday.value) * (obstime - self.target.epoch).jd) % (2*np.pi) * u.rad

    def LST(self, location, obstime):
        return (eq2000.GMST.deg + self.sidereal_day(obstime).to(u.deg) + location.lon) % (360*u.deg)

    def calculate_ha(self, target, location, obstime):
        return (self.LST(location, obstime).to(u.deg) - target.ra) % (360*u.deg)

    def calculate_az(self, target, location, obstime):
        return (self.calculate_ha(target, location, obstime) - 180 * u.deg) % (360*u.deg)

    def calculate_alt(self, zenith='default', target='default'):
        if zenith != 'default' and not isinstance(zenith, Versor):
            raise TypeError("Invalid zenith.")
        if target != 'default' and not isinstance(target, Versor):
            raise TypeError("Invalid target.")

        if zenith == 'default':
            zenith_vsr = self.zenith.vsr
        else:
            zenith_vsr = zenith.vsr

        if target == 'default':
            target_vsr = self.target.vector_obstime.vsr
        else:
            target_vsr = target.vsr

        if (ps := zenith_vsr.dot(target_vsr)) >= 0:
            return (90 - np.rad2deg(np.arccos(ps))) % 90 * u.deg
        else:
            return (90 - np.rad2deg(np.arccos(ps))) % -90 * u.deg

    def plot_altaz_onday(self, interval=15*u.min):
        interval = interval.to(u.hour)
        numpoints = int(24*u.hour/interval)

        s_time = Time(int(self.obstime.mjd), format='mjd')
        dt = 15*u.min * np.linspace(0, numpoints, num=numpoints + 1)
        times = np.array([s_time + delta for delta in dt])

        alt = np.empty(numpoints + 1, dtype=u.quantity.Quantity)
        az = np.empty(numpoints + 1, dtype=u.quantity.Quantity)
        index = 0

        for t in times:
            t_zenith = self.zenith_at_date(t)
            tgt = self.target.observe_at_date(t, copy=True)

            alt[index] = self.calculate_alt(t_zenith, tgt)
            az[index] = self.calculate_az(tgt, self.location, t)

            index += 1

        times = Time([t for t in times], scale='utc')
        times = times.to_value('iso', subfmt='date_hm')
        alt = u.quantity.Quantity([at for at in alt])
        az = u.quantity.Quantity([z for z in az])

        plt.close(1)
        fig = plt.figure(1)

        fig.suptitle(self.target.name)

        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax1.grid()
        ax1.plot(times, alt, 'k-')
        ax1.set_ylim(-90, 90)
        ax1.yaxis.set_ticks(np.linspace(-90, 90, 7))
        ax1.xaxis.set_major_locator(plt.MaxNLocator(25))
        ax1.set_xticklabels([])
        ax1.set_xlabel('')
        ax1.set_ylabel('Alt [deg]')

        ax2 = plt.subplot2grid((2, 1), (1, 0))
        ax2.grid()
        ax2.plot(times, az, 'k-')
        ax2.set_ylim(0, 360)
        ax2.yaxis.set_ticks(np.linspace(0, 360, 13))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(25))
        ax2.xaxis.set_tick_params(rotation=80)
        ax2.set_ylabel('Az [deg]')

        ax1.set_xlim(min(times), max(times))
        ax2.set_xlim(min(times), max(times))

        fig.tight_layout()
        fig.show()
