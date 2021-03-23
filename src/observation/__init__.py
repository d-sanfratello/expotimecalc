import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from astropy.coordinates import Angle
from astropy.units.quantity import Quantity
from astropy.visualization import time_support
from astropy.visualization import quantity_support

from src.location import Location
from src.skylocation import SkyLocation
from src.skylocation.sun import Sun
from src.time import Time
from src import Versor

from src import Tsidday
from src import Tsidyear
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

        self.ha = None
        self.az = None

        self.alt = None
        self.zenith_dist = None
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

        self.target.at_date(self.obstime)
        self.location.zenith_at_date(self.obstime, copy=False)

        self.ha = self.calculate_ha(self.target, self.location, self.obstime)
        self.az = self.calculate_az(self.target, self.location, self.obstime)

        self.alt = self.calculate_alt(self.target, self.location, self.obstime)
        self.zenith_dist = self.calculate_zenith_dist(self.target, self.location, self.obstime)
        self.airmass = self.calculate_airmass(self.target, self.location, self.obstime)

        self.culmination = self.calculate_culmination(self.target, self.location, self.obstime)

        self.set_time = self.calculate_set_time(self.target, self.location, self.obstime)
        self.rise_time = self.calculate_rise_time(self.target, self.location, self.obstime)

        self.visibility = self.calculate_visibility(self.target, self.location, self.obstime)

        self.rise_azimuth = self.calculate_az(self.target, self.location, self.rise_time)
        self.set_azimuth = self.calculate_az(self.target, self.location, self.set_time)

    @classmethod
    def calculate_ha(cls, target, location, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        target_vsr = target.precession_at_date(obstime)

        return (location.lst(obstime, epoch_eq) - target_vsr.ra).to(u.hourangle) % (24 * u.hourangle)

    @classmethod
    def calculate_az(cls, target, location, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        # z_dist = cls.calculate_zenith_dist(target, location, obstime)
        #
        # target = target.precession_at_date(obstime)
        # cos_az = (np.sin(target.dec) - np.cos(location.lat) * np.cos(z_dist)) / (np.sin(location.lat) * np.sin(z_dist))

        # return np.arccos(cos_az).to(u.deg)

        return 90 * u.deg - cls.calculate_ha(target, location, obstime, epoch_eq)  # serve comunque controllo su posizione rispetto a nord O est. Però 2/4 tornano, ora.

    @classmethod
    def calculate_alt(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        target_vsr = target.precession_at_date(obstime).vsr
        zenith_vsr = location.zenith_at_date(obstime, copy=True)[0].vsr

        if (ps := zenith_vsr.dot(target_vsr)) >= 0:
            return Angle((90 - np.rad2deg(np.arccos(ps))) % 90 * u.deg)
        else:
            return Angle((90 - np.rad2deg(np.arccos(ps))) % -90 * u.deg)

    @classmethod
    def estimate_quality(cls, target, location, obstime, parameter=1.5, interval=2*u.hour, par_type='airmass'):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(parameter, (Quantity, int)):
            raise TypeError(errmsg.notTwoTypesError.format('parameter', 'astropy.units.quantity.Quantity', 'int'))
        if not isinstance(interval, Quantity):
            raise TypeError(errmsg.notTypeError.format('interval', 'astropy.units.quanrtity.Quantity'))

        if not isinstance(par_type, str):
            raise TypeError(errmsg.notTypeError.format('par_type', 'string'))
        elif isinstance(parameter, Quantity) and par_type not in ['alt', 'z']:
            raise ValueError(errmsg.altZError)
        elif isinstance(parameter, int) and par_type not in ['airmass']:
            raise ValueError(errmsg.airmassError)

        if par_type == 'alt':
            z_max = 90 * u.deg - parameter
        elif par_type == 'z':
            z_max = parameter
        elif par_type == 'airmass':
            # arcsec(1/x) = arccos(x)
            z_max = np.arccos(1/parameter)

        if cls.calculate_zenith_dist(target, location, cls.calculate_culmination(target, location, obstime)) >= z_max:
            return False, 0*u.hour
        else:
            sidday_factor = Tsidday.to(u.hour) / (2 * np.pi * u.rad)

            time_above = np.cos(z_max) - np.sin(location.lat) * np.sin(target.dec)
            time_above /= np.cos(location.lat * np.cos(target.dec))
            time_above = 2 * sidday_factor * np.arccos(time_above)

            return (time_above >= interval), time_above

    @classmethod
    def calculate_zenith_dist(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        alt = cls.calculate_alt(target, location, obstime)

        return 90 * u.deg - alt

    @classmethod
    def calculate_airmass(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        warnings.warn(warnmsg.airmassWarning)

        target_vsr = target.precession_at_date(obstime).vsr
        zenith_vsr = location.zenith_at_date(obstime, copy=True)[0].vsr

        if (ps := zenith_vsr.dot(target_vsr)) > 0:
            # airmass = sec(z) = 1/cos(z)
            return 1 / ps
        else:
            return 0

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

        vector_obstime = target.precession_at_date(obstime)

        if vector_obstime.dec * location.lat <= 0 and abs(vector_obstime.dec) >= abs(location.lat):
            return None
        else:
            delta_time = (Tsidday.value/(2*np.pi))*(vector_obstime.ra - reference.GMST.rad - location.lon).rad * u.day
            delta_time -= location.timezone
            delta_time += 12 * u.hour

            if cls.calculate_alt(target, location, obstime) < 0 * u.deg:
                delta_time -= 12 * u.hour

            if delta_time >= 0:
                return delta_time + obstime
            else:
                return delta_time + obstime + 1 * u.day

    @classmethod
    def calculate_set_time(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        vector_obstime = target.precession_at_date(obstime)

        if vector_obstime.dec >= location.lat or vector_obstime.dec <= -location.lat:
            return None

        culm_t = cls.calculate_culmination(target, location, obstime)
        visibility_window = cls.calculate_visibility(target, location, obstime)
        return culm_t + visibility_window / 2

    @classmethod
    def calculate_rise_time(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        vector_obstime = target.precession_at_date(obstime)

        if vector_obstime.dec >= location.lat or vector_obstime.dec <= -location.lat:
            return None

        culm_t = cls.calculate_culmination(target, location, obstime)
        visibility_window = cls.calculate_visibility(target, location, obstime)
        return culm_t - visibility_window / 2

    @classmethod
    def calculate_visibility(cls, target, location, obstime):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        warnings.warn(warnmsg.visibilityWarning)

        vector_obstime = target.precession_at_date(obstime)

        if abs(vector_obstime.dec) >= abs(location.lat):
            if vector_obstime.dec * location.lat > 0:
                return 24 * u.hour
            elif vector_obstime.dec * location.lat <= 0:
                return 0 * u.hour
        else:
            sidday_factor = Tsidday.to(u.hour) / (2 * np.pi * u.rad)
            return 2 * sidday_factor * np.arccos(- np.tan(vector_obstime.dec.rad) * np.tan(location.lat))

    @classmethod
    def calculate_best_day(cls, target, location, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        vector_obstime = target.precession_at_date(obstime)

        if vector_obstime.dec * location.lat <= 0 and abs(vector_obstime.dec) >= abs(location.lat):
            return None
        else:
            sidyear_factor = (Tsidyear.value/(2*np.pi))
            time = vector_obstime.ra + 180 * u.deg - reference.GMST.rad - location.lon
            time -= location.timezone * 15 * u.deg/u.hour
            time = sidyear_factor * time.rad * u.day

            best_target = SkyLocation(ra=target.ra, dec=target.dec, obstime=time+obstime)

            return cls.calculate_culmination(best_target, location, time+obstime, epoch_eq)

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
            tgt = self.target.precession_at_date(t)

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
