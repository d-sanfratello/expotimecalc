import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u

from src.location import Location
from src.skylocation import SkyLocation
from src.time import Time
from src import Versor

from src import Tsidday
from src import eq2000


class Observation:
    def __init__(self, location, obstime=None, target=None):
        if not isinstance(location, Location):
            raise TypeError("Must be of type `expotimecalc.location.Location`.")
        if not isinstance(obstime, Time):
            raise TypeError("Must be of type `expotimecalc.time.Time` or `astropy.time.Time`.")
        if not isinstance(target, SkyLocation):
            raise TypeError("Must be of type `expotimecalc.skylocation.SkyLocation`.")

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
        return (2*np.pi/Tsidday.value) * (obstime - self.target.epoch).jd * u.rad

    def LST(self, location, obstime):
        return eq2000.GMST.deg + self.sidereal_day(obstime).to(u.deg) + location.lon

    def calculate_ha(self, target, location, obstime):
        return self.LST(location, obstime).to(u.deg) - target.ra

    def calculate_az(self, target, location, obstime):
        return self.calculate_ha(target, location, obstime) - 180 * u.deg

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

        if ps := zenith_vsr.dot(target_vsr) >= 0:
            return np.rad2deg(np.arccos(ps)) * u.deg
        else:
            return -np.rad2deg(np.arccos(ps)) * u.deg

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

        plt.close(1)
        plt.figure(1)

        plt.subplot2grid(2,1,0,0)
        plt.grid()
        plt.plot(times, alt, 'k-')

        plt.subplot2grid(2,1,1,0)
        plt.grid()
        plt.plot(times, az, 'k-')

        plt.show()
