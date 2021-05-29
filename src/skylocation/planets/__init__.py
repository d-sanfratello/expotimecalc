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

        self.__mass_sun = cts.M_sun
        self.__grav_const = cts.G * self.__mass_sun / (4 * np.pi ** 2)

        self.__semimaj = None
        self.__ecc = None
        self.__argument_pericenter = None
        self.__mean_anomaly = None
        self.__inclination = None
        self.__longitude_an = None

        self.at_date(obstime)

        super(Planet, self).__init__(locstring=None, ra=self.vector_obstime.ra, dec=self.vector_obstime.dec,
                                     distance=self.distance_from_sun, obstime=obstime,
                                     ra_unit='deg', dec_unit='deg', epoch=epoch, name=name)

    def observe_at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        epoch = pk.epoch_from_string(obstime.iso)

        __semimaj, __ecc, __inclination, __ra_an, __peri_arg, __mean_anomaly = self.ephemeris.osculating_elements(epoch)
        __semimaj *= u.m
        self.__semimaj = __semimaj.to(cts.au)

        __inclination *= u.rad
        self.__inclination = Angle(__inclination.to(u.deg))

        __ra_an *= u.rad
        self.__longitude_an = Longitude(__ra_an.to(u.deg))

        __peri_arg *= u.rad
        self.__argument_pericenter = Angle(__peri_arg.to(u.deg))
        self.__ecc = __ecc
        self.__mean_anomaly = __mean_anomaly

        if self.__is_earth:
            reference_observer = Versor(vector=np.zeros(3) * cts.au)
        else:
            from .earth import Earth
            reference_observer = Earth(obstime=obstime).vector_obstime

        vector_obstime = Versor(vector=np.array([1, 0, 0]) * self.distance_from_sun) \
            .rotate(axis='z', angle=self.mean_anomaly, copy=True) \
            .rotate(axis='z', angle=self.argument_pericenter, copy=True) \
            .rotate(axis='x', angle=self.inclination, copy=True) \
            .rotate(axis='z', angle=self.longitude_an, copy=True)

        vector_obstime = (vector_obstime - reference_observer) \
            .rotate(axis='x', angle=self.axial_tilt(obstime), copy=True)  # check for rotate or rotate_inv

        return vector_obstime

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

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

    # noinspection PyTypeChecker
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
    def inclination(self):
        return self.__inclination

    # noinspection PyTypeChecker
    @property
    def peri_dist(self):
        return self.semimaj * (1 - self.eccentricity)

    # noinspection PyTypeChecker
    @property
    def apo_dist(self):
        return self.semimaj * (1 + self.eccentricity)

    @property
    def distance_from_sun(self):
        return 0.5 * (self.peri_dist + self.apo_dist)


from .venus import Venus
from .earth import Earth
