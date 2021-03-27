import numpy as np

from astropy import units as u

from src.time import Time
from src.skylocation import SkyLocation

from src import Tsidyear
from src import Omegasidmoon
from src import Equinox2000

from src import errmsg
from src import warnmsg


class Moon(SkyLocation):
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # https://ssd.jpl.nasa.gov/horizons.cgi
        # Ephemeris Type [change] : 	OBSERVER
        # Target Body [change] : 	    Moon [Luna] [301]
        # Observer Location [change] : 	Greenwich [000] ( 0°00'00.0''E, 51°28'38.6''N, 65.8 m )
        # Time Span [change] : 	        Start=2000-03-20 07:35, Stop=2000-03-20 07:37, Intervals=120
        # Table Settings [change] : 	QUANTITIES=1,7,9,20,23,24
        # Display/Output [change] : 	default (formatted HTML)
        #
        # Selected RA-DEC for the time corresponding to `Equinox2000.time`.
        super(Moon, self).__init__(locstring=None,
                                   ra='12h10m9.25s', dec='2d39m54.9s', obstime=obstime,
                                   ra_unit='hms', dec_unit='dms', epoch='J2000', name='Moon')

        self.at_date(obstime)

    def observe_at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        vector_ecl = self.vector_epoch.rotate_inv('x', self.axial_tilt(obstime), copy=True)
        b_ecliptic_lat = vector_ecl.dec
        l_ecliptic_lon = vector_ecl.ra

        vector_obstime = self.vector_epoch.rotate_inv('x', self.axial_tilt(obstime), copy=True)\
            .rotate_inv('z', l_ecliptic_lon, copy=True).rotate('y', b_ecliptic_lat, copy=True)\
            .rotate('z', self.moon_revolution(obstime), copy=True) \
            .rotate_inv('y', b_ecliptic_lat, copy=True).rotate('z', l_ecliptic_lon, copy=True) \
            .rotate('z', self.equinox_prec(obstime), copy=True)\
            .rotate('x', self.axial_tilt(obstime), copy=True)

        return vector_obstime

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

    @classmethod
    def moon_revolution(cls, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        return (Omegasidmoon.value * (obstime - reference.time).jd) % (2 * np.pi) * u.rad

    @classmethod
    def sidereal_year_rotation(cls, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        return ((2 * np.pi / Tsidyear.value) * (obstime - reference.time).jd) % (2 * np.pi) * u.rad
