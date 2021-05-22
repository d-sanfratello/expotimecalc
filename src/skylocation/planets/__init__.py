import numpy as np

from astropy import units as u
from astropy import constants as cts
from astropy.coordinates import Angle
from astropy.coordinates.angles import Longitude

from ...time import Time
from .. import SkyLocation

from .. import Equinox2000
from ... import Tsidyear

from .. import errmsg


class Planet(SkyLocation):
    def __init__(self, ra, dec, distance, obstime,
                 longitude_an, orbit_inclination, revolution_period,
                 name=None, epoch='J2000'):

        if not isinstance(distance, u.Quantity):
            raise TypeError(errmsg.notTypeError('distance', 'astropy.units.Quantity'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(longitude_an, (Angle, Longitude)):
            raise TypeError(errmsg.notTwoTypesError.format('longitude_an', 'astropy.coordinates.Angle',
                                                           'astropy.coordinates.angles.Longitude'))
        if not isinstance(orbit_inclination, Angle):
            raise TypeError(errmsg.notTypeError('orbit_inclination', 'astropy.coordinates.Angle'))
        if not isinstance(revolution_period, u.Quantity):
            raise TypeError(errmsg.notTypeError('revolution_period', 'astropy.units.Quantity'))
        if name is None:
            raise ValueError(errmsg.mustDeclareName)

        self.__longitude_an = longitude_an
        self.__orbit_inclination = orbit_inclination
        self.__revolution_period = revolution_period

        super(Planet, self).__init__(locstring=None, ra=ra, dec=dec, distance=distance,
                                     obstime=obstime, ra_unit='deg', dec_unit='deg', epoch=epoch, name=name)

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
        vector_obstime = (self.heliocentric_planet_J2000.rotate_inv('z', angle=self.longitude_an, copy=True)
                              .rotate_inv('x', angle=self.orbit_inclination, copy=True)
                              .rotate('z', angle=self.revolution(obstime), copy=True)
                              .rotate('x', angle=self.orbit_inclination, copy=True)
                              .rotate('z', angle=self.longitude_an, copy=True)
                          - self.heliocentric_earth_J2000.rotate('z', angle=self.year_revolution(obstime), copy=True))\
            .rotate('z', angle=self.equinox_prec(obstime), copy=True)\
            .rotate('x', angle=self.axial_tilt(obstime), copy=True)

        return vector_obstime

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

    def revolution(self, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = self.equinoxes[epoch_eq]

        # Restituisce la fase accumulata, in radianti, in rapporto al periodo di rivoluzione del pianeta.
        return ((2 * np.pi / self.revolution_period.value) * (obstime - reference.time).jd) % (2 * np.pi) * u.rad

    # noinspection PyPep8Naming
    @property
    def heliocentric_planet_J2000(self):
        helioc_planet = self.vector_epoch.rotate_inv('z', angle=self.axial_tilt(Equinox2000.time), copy=True)\
                        + self.heliocentric_earth_J2000

        return helioc_planet

    @property
    def longitude_an(self):
        return self.__longitude_an

    @property
    def orbit_inclination(self):
        return self.__orbit_inclination

    @property
    def revolution_period(self):
        return self.__revolution_period


from .venus import Venus
