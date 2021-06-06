import logging
import numpy as np
import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from astropy import units as u
from astropy import constants as cts
from astropy.coordinates import Angle
from astropy.coordinates.angles import Longitude
from numbers import Number

from .. import Versor
from ...time import Time
from .. import SkyLocation

from .. import Equinox2000
from .. import tJ2000
from ... import Tsidyear

from .. import errmsg
from .. import logger


# noinspection PyUnresolvedReferences
class Planet(SkyLocation):
    __private_arguments = ['is_earth']

    def __init__(self, obstime, *, name=None, epoch='J2000', **kwargs):

        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if name is None:
            raise ValueError(errmsg.mustDeclareName)

        if not bool(kwargs):
            self.__is_earth = False
        else:
            k = kwargs.keys()
            for _ in k:
                if _ not in self.__private_arguments:
                    raise KeyError(errmsg.keyError)
                else:
                    self.__is_earth = kwargs[_]
        self.__logger = logging.getLogger('src.skylocation.planets.Planet')
        self.__logger.setLevel(logger.getEffectiveLevel())

        self.__logger.debug('Initializing `Planet` class')
        self.__logger.info(f'instance of {name}')
        self.__logger.debug(f'{name} is earth: {self.__is_earth}')

        self.__mass_sun = cts.M_sun
        self.__grav_const = cts.G * self.__mass_sun / (4 * np.pi ** 2)

        self.__semimaj = None
        self.__ecc = None
        self.__argument_pericenter = None
        self.__mean_anomaly = None
        self.__eccentric_anomaly = None
        self.__inclination = None
        self.__longitude_an = None

        self.at_date(obstime)

        super(Planet, self).__init__(locstring=None, ra=self.vector_obstime.ra, dec=self.vector_obstime.dec,
                                     distance=self.distance_from_sun, obstime=obstime,
                                     ra_unit='deg', dec_unit='deg', epoch=epoch, name=name)

    def observe_at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        self.__logger.debug(f'`obstime`: {obstime} call.')

        epoch = pk.epoch_from_string(obstime.iso)

        __semimaj, __ecc, __inclination, __ra_an, __peri_arg, __mean_anomaly = self.ephemeris.osculating_elements(epoch)
        __semimaj *= u.m
        self.__semimaj = __semimaj.to(u.AU)

        __inclination *= u.rad
        self.__inclination = Angle(__inclination.to(u.deg))

        __ra_an *= u.rad
        self.__longitude_an = Longitude(__ra_an.to(u.deg))

        __peri_arg *= u.rad
        self.__argument_pericenter = Angle(__peri_arg.to(u.deg))
        self.__ecc = np.float64(__ecc)

        __mean_anomaly *= u.rad
        self.__mean_anomaly = Angle(__mean_anomaly.to(u.deg))
        self.__eccentric_anomaly = self.__approx_ecc_anomaly()

        if self.__is_earth:
            self.__logger.info(f'It is Earth.')
            self.__logger.debug(f'reference observer is the Sun')
        else:
            self.__logger.info(f'It is NOT Earth. Defining an `Earth` instance as reference')
            from .earth import Earth
            reference_observer = Earth(obstime=obstime).vector_obstime
            self.__logger.debug(f'Earth reference defined as {reference_observer.vsr}, {reference_observer.radius}')

        self.__logger.info(f'Rotating body with osculating parameters from ephemeris')
        vector_obstime = Versor(vector=np.array([1, 0, 0]) * self.distance_from_sun) \
            .rotate(axis='z', angle=self.true_anomaly, copy=True) \
            .rotate(axis='z', angle=self.argument_pericenter, copy=True) \
            .rotate(axis='x', angle=self.inclination, copy=True) \
            .rotate(axis='z', angle=self.longitude_an, copy=True)

        if not self.__is_earth:
            vector_obstime = (vector_obstime - reference_observer) \
                .rotate(axis='z', angle=self.equinox_prec(obstime), copy=True) \
                .rotate(axis='x', angle=self.axial_tilt(obstime), copy=True)  # check for rotate or rotate_inv

        self.__logger.info(f'Body position estimated at {vector_obstime.ra.hms}, {vector_obstime.dec.deg}, '
                           f'{vector_obstime.radius}. Returning `vector_obstime` from `observe_at_date`.')
        return vector_obstime

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        self.__logger.debug(f'obstime: {obstime}')

        self.obstime = obstime

        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec
        self.__logger.debug(f'ra-dec set by `at_date` method at {self.ra.hms}, {self.dec.deg}')

    def __approx_ecc_anomaly(self):
        # Method:
        #  1998aalg.book.....M (p196)
        mean = self.mean_anomaly.to(u.rad)

        ecc_an = mean
        ecc_an_next = mean - self.eccentricity * np.sin(ecc_an) * u.rad
        while abs(ecc_an_next.to(u.deg) - ecc_an.to(u.deg)) > (1e-4 * u.arcsec).to(u.deg):
            ecc_an = ecc_an_next
            ecc_an_next = mean - self.eccentricity * np.sin(ecc_an) * u.rad

        return Angle(ecc_an_next.to(u.deg))

    @property
    def longitude_an(self):
        return self.__longitude_an

    @property
    def revolution_period(self):
        return np.sqrt(self.semimaj ** 3 / self.__grav_const).to(u.day)

    @property
    def semimaj(self):
        return self.__semimaj

    @property
    def eccentricity(self):
        return self.__ecc

    @property
    def semimin(self):
        return self.semimaj * np.sqrt(1 - self.eccentricity**2)

    @property
    def argument_pericenter(self):
        return self.__argument_pericenter

    @property
    def mean_anomaly(self):
        return self.__mean_anomaly

    @property
    def eccentric_anomaly(self):
        return self.__eccentric_anomaly

    @property
    def true_anomaly(self):
        return (2 * np.arctan(np.sqrt((1 + self.eccentricity) / (1 - self.eccentricity))
                              * np.tan(self.eccentric_anomaly / 2))).to(u.deg)

    @property
    def inclination(self):
        return self.__inclination

    @property
    def peri_dist(self):
        return self.semimaj * (1 - self.eccentricity)

    @property
    def apo_dist(self):
        return self.semimaj * (1 + self.eccentricity)

    @property
    def distance_from_sun(self):
        return self.semimaj * (1 - self.eccentricity * np.cos(self.eccentric_anomaly))


from .mercury import Mercury
from .venus import Venus
from .earth import Earth
from .moon import Moon
from .mars import Mars
from .jupiter import Jupiter
