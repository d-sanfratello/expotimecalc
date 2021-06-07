import numpy as np

from astropy import units as u
from astropy import constants as cts
from astropy.coordinates.angles import Angle

from ...time import Time
from ... import Versor
from .. import SkyLocation
from .earth import Earth

from ... import Equinox2000

from .. import errmsg


# noinspection PyUnresolvedReferences
class Sun(SkyLocation):
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, obstime):
        """
        Classe che descrive il moto della Luna in funzione della data di osservazione.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.__mass_sun = cts.M_sun
        self.__grav_const = cts.G * self.__mass_sun / (4 * np.pi ** 2)

        self.__earth = Earth(obstime)

        self.at_date(obstime)

        super(Sun, self).__init__(locstring=None,
                                  ra=self.vector_obstime.ra, dec=self.vector_obstime.dec,
                                  distance=self.distance_from_earth, obstime=obstime,
                                  ra_unit='deg', dec_unit='deg', epoch='J2000', name='Sun')

    def observe_at_date(self, obstime, return_complete=False):
        """
        Metodo che calcola, senza salvarne il risultato, la posizione della Luna ad una specifica data.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(return_complete, bool):
            raise TypeError(errmsg.notTypeError.format('return_complete', 'bool'))

        self.__earth.at_date(obstime)

        vector_obstime = Versor(vector=np.zeros(3)) - self.__earth.vector_obstime

        if return_complete:
            return vector_obstime, None
        else:
            return vector_obstime

    def at_date(self, obstime):
        """
        Metodo che salva il risultato di `Moon.observe_at_date` nei relativi attributi di classe.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime

        self.vector_obstime = self.observe_at_date(obstime, return_complete=False)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

    @property
    def distance_from_earth(self):
        return self.__earth.distance_from_sun
