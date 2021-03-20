import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from astropy.visualization import time_support
from astropy.visualization import quantity_support

from src.location import Location
from src.skylocation import SkyLocation
from src.skylocation.sun import Sun
from src.time import Time
from src import Versor

from src import Tsidday
from src import Equinox2000

from src import errmsg
from src import warnmsg

import warnings


time_support(scale='utc', format='iso', simplify=True)
quantity_support()


# noinspection PyUnresolvedReferences
class Observation:
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, location, obstime=None, target=None):
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))

        self.location = location
        self.obstime = obstime
        self.target = target
        self.sun = Sun(self.obstime)

        self.zenithJ2000 = Versor(ra=self.lst(self.location, Equinox2000.time).rad,
                                  dec=self.location.lat.rad,
                                  unit='rad')

        self.zenith = None

        self.target_ha = None
        self.target_azimuth = None
        self.target_alt = None
        self.airmass = None

        self.culmination = None
        self.visibility = None

        self.rise_time = None
        self.rise_azimuth = None
        self.rise_ha = None

        self.set_time = None
        self.set_azimuth = None
        self.set_ha = None

        self.make_observation(self.obstime)

    def make_observation(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime

        self.target.precession_at_date(self.obstime, copy=False)

        self.zenith = self.zenith_at_date(self.obstime)

        self.target_ha = self.calculate_ha(self.target, self.location, self.obstime)
        self.target_azimuth = self.calculate_az(self.target, self.location, self.obstime)
        self.target_alt = self.calculate_alt(self.target, self.zenith)

        if (ps := self.zenith.vsr.dot(self.target.vector_obstime.vsr)) > 0:
            # airmass = sec(z) = 1/cos(z)
            self.airmass = 1 / ps
        else:
            self.airmass = 0
        warnings.warn(warnmsg.airmassWarning)

        self.culmination = self.calculate_culmination(self.target, self.location, self.obstime)

        self.set_time = self.calculate_set_time(self.target, self.location, self.obstime)
        self.rise_time = self.calculate_rise_time(self.target, self.location, self.obstime)

        self.visibility = self.calculate_visibility(self.target, self.location)

        self.rise_azimuth = self.calculate_az(self.target, self.location, self.rise_time)
        self.rise_ha = self.calculate_ha(self.target, self.location, self.rise_time).to(u.hourangle)

        self.set_azimuth = self.calculate_az(self.target, self.location, self.set_time)
        self.set_ha = self.calculate_ha(self.target, self.location, self.set_time).to(u.hourangle)

    def zenith_at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        return self.zenithJ2000.rotate('z', self.sidereal_day_rotation(obstime), copy=True)

    @classmethod
    def sidereal_day_rotation(cls, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        return ((2 * np.pi * u.rad / Tsidday.value) * (obstime - reference.time).jd)  # % (2*np.pi)

    @classmethod
    def lst(cls, location, obstime):
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        shift = Equinox2000.GMST.deg + cls.sidereal_day_rotation(obstime).to(u.deg) + location.lon
        return shift.to(u.hourangle) % (24 * u.hourangle)

    @classmethod
    def calculate_ha(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        return (cls.lst(location, obstime) - target.ra).to(u.hourangle) % (24 * u.hourangle)

    @classmethod
    def calculate_az(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        return (cls.calculate_ha(target, location, obstime).to(u.deg) - 180 * u.deg) % (360*u.deg)

    @classmethod
    def calculate_alt(cls, target, zenith):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(zenith, Versor):
            raise TypeError(errmsg.notTypeError.format('zenith', 'src.Versor'))

        target_vsr = target.vector_obstime.vsr
        zenith_vsr = zenith.vsr

        if (ps := zenith_vsr.dot(target_vsr)) >= 0:
            return (90 - np.rad2deg(np.arccos(ps))) % 90 * u.deg
        else:
            return (90 - np.rad2deg(np.arccos(ps))) % -90 * u.deg

    @classmethod
    def calculate_culmination(cls, target, location, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        if target.dec * location.lat <= 0 and abs(target.dec) >= abs(location.lat):
            return None
        else:
            time = (Tsidday.value/(2*np.pi)) * (target.ra - reference.GMST.rad - location.lon).rad * u.day
            if time >= 0:
                return time + obstime
            else:
                return time + obstime + 1 * u.day

    @classmethod
    def calculate_set_time(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        if target.dec >= location.lat or target.dec <= -location.lat:
            return None

        culm_t = cls.calculate_culmination(target, location, obstime)
        visibility_window = cls.calculate_visibility(target, location)
        return culm_t + visibility_window / 2

    @classmethod
    def calculate_rise_time(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        if target.dec >= location.lat or target.dec <= -location.lat:
            return None

        culm_t = cls.calculate_culmination(target, location, obstime)
        visibility_window = cls.calculate_visibility(target, location)
        return culm_t - visibility_window / 2

    @classmethod
    def calculate_visibility(cls, target, location):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        warnings.warn(warnmsg.visibilityWarning)

        if abs(target.dec) >= abs(location.lat):
            if target.dec * location.lat > 0:
                return 24 * u.hour
            elif target.dec * location.lat <= 0:
                return 0 * u.hour
        else:
            sidday_factor = Tsidday.to(u.hour) / (2 * np.pi * u.rad)
            return 2 * sidday_factor * np.arccos(- np.tan(target.dec.rad) * np.tan(location.lat))

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
            tgt = self.target.precession_at_date(t, copy=True)

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
