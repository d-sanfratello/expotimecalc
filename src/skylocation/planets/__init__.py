import numpy as np

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
    def __init__(self, obstime, *, semimaj, eccentricity, longitude_an, inclination,
                 argument_perihelion, mean_anomaly,
                 name=None, epoch='J2000'):

        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(semimaj, u.Quantity):
            raise TypeError(errmsg.notTypeError('semimaj', 'astropy.units.Quantity'))
        if not isinstance(eccentricity, Number):
            raise TypeError(errmsg.notTypeError('eccentricity', 'Number'))
        if not isinstance(longitude_an, (Angle, Longitude)):
            raise TypeError(errmsg.notTwoTypesError.format('longitude_an', 'astropy.coordinates.Angle',
                                                           'astropy.coordinates.angles.Longitude'))
        if not isinstance(inclination, Angle):
            raise TypeError(errmsg.notTypeError('orbit_inclination', 'astropy.coordinates.Angle'))
        if not isinstance(argument_perihelion, Angle):
            raise TypeError(errmsg.notTypeError('argument_periapsis', 'astropy.coordinates.Angle'))
        if name is None:
            raise ValueError(errmsg.mustDeclareName)

        self.__mass_sun = cts.M_sun
        self.__grav_const = cts.G * self.__mass_sun / (4 * np.pi ** 2)

        self.__semimaj = semimaj
        self.__ecc = eccentricity
        self.__argument_perihelion = argument_perihelion
        self.__mean_anomaly = mean_anomaly
        self.__inclination = inclination
        self.__longitude_an = longitude_an

        self.__initialize()

        super(Planet, self).__init__(locstring=None, ra=self.vector_epoch.ra, dec=self.vector_epoch.dec,
                                     distance=self.distance_from_sun, obstime=obstime,
                                     ra_unit='deg', dec_unit='deg', epoch=epoch, name=name)

        self.at_date(obstime)

    def observe_at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Il pianeta, che è stato inizializzato alla sua posizione iniziale all'epoca J2000, ha il versore ruotato
        # per avere il nodo ascendente dell'orbita coincidente con l'asse 'x', intorno a cui l'orbita viene inclinata
        # rispetto al piano dell'eclittica. Dopo l'applicazione del moto di rivoluzione intorno al Sole, si riporta
        # il pianeta sul piano della sua orbita, rispetto all'eclittica. Quindi si sottrae il moto della Terra,
        # intorno al Sole. Al vettore ottenuto, si applica la precessione degli equinozi per la correzione delle
        # coordinate equatoriali alla data e si trasforma il vettore in coordinate equatoriali, applicando una
        # rotazione intorno all'asse 'x'.
        # vector_obstime = (self.heliocentric_planet_J2000.rotate_inv('z', angle=self.longitude_an, copy=True)
        #                       .rotate_inv('x', angle=self.orbit_inclination, copy=True)
        #                       .rotate('z', angle=self.revolution(obstime), copy=True)
        #                       .rotate('x', angle=self.orbit_inclination, copy=True)
        #                       .rotate('z', angle=self.longitude_an, copy=True)
        #                   - self.heliocentric_earth_J2000.rotate('z', angle=self.year_revolution(obstime), copy=True)) \
        #     .rotate('z', angle=self.equinox_prec(obstime), copy=True) \
        #     .rotate('x', angle=self.axial_tilt(obstime), copy=True)
        heliocentric = Versor(ra=0 * u.deg, dec=0 * u.deg, radius=self.distance_from_sun) \
            .rotate(axis='z', angle=self.revolution(obstime, reference=tJ2000), copy=True) \
            .rotate(axis='z', angle=self.mean_anomaly, copy=True) \
            .rotate(axis='z', angle=self.argument_perihelion, copy=True) \
            .rotate(axis='x', angle=self.inclination, copy=True) \
            .rotate(axis='z', angle=self.longitude_an, copy=True)

        vector_obstime = (heliocentric - self.heliocentric_earth_J2000) \
            .rotate_inv(axis='x', angle=self.axial_tilt(obstime), copy=True)  # check for rotate or rotate_inv

        return vector_obstime

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

    def revolution(self, obstime, epoch_eq='equinoxJ2000', reference=None):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)
        if reference is not None and not isinstance(reference, Time):
            raise TypeError(errmsg.notTypeError.format('reference', 'astropy.time.Time'))

        if reference is None:
            reference = self.equinoxes[epoch_eq]
            ref_time = reference.time
        else:
            ref_time = reference

        # Restituisce la fase accumulata, in radianti, in rapporto al periodo di rivoluzione del pianeta.
        return ((2 * np.pi / self.revolution_period.value) * (obstime - ref_time).jd) % (2 * np.pi) * u.rad

    def __initialize(self):
        self.helioc_eq2000 = Versor(ra=0 * u.deg, dec=0 * u.deg, radius=self.distance_from_sun) \
            .rotate(axis='z', angle=self.revolution(Equinox2000.time, reference=tJ2000), copy=True) \
            .rotate(axis='z', angle=self.mean_anomaly, copy=True) \
            .rotate(axis='z', angle=self.argument_perihelion, copy=True) \
            .rotate(axis='x', angle=self.inclination, copy=True) \
            .rotate(axis='z', angle=self.longitude_an, copy=True)

        self.vector_epoch = (self.helioc_eq2000 - self.heliocentric_earth_J2000) \
            .rotate_inv(axis='x', angle=self.axial_tilt(Equinox2000.time), copy=True)  # check for rotate or rotate_inv

    # # noinspection PyPep8Naming
    # @property
    # def heliocentric_planet_J2000(self):
    #     helioc_planet = self.vector_epoch.rotate_inv('z', angle=self.axial_tilt(Equinox2000.time), copy=True)\
    #                     + self.heliocentric_earth_J2000
    #
    #     return helioc_planet

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
    def semimin(self):
        return self.semimaj * np.sqrt(1 - self.eccentricity**2)

    @property
    def eccentricity(self):
        return self.__ecc

    @property
    def argument_perihelion(self):
        return self.__argument_perihelion

    @property
    def mean_anomaly(self):
        return self.__mean_anomaly

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
        return 0.5 * (self.peri_dist + self.apo_dist)


from .venus import Venus
