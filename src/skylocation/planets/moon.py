import numpy as np

from astropy import units as u
from astropy import constants as cts
from astropy.coordinates.angles import Angle
from skyfield.api import load
from skyfield.elementslib import osculating_elements_of

from ...time import Time
from ... import Versor
from .. import SkyLocation

from ... import Equinox2000

from .. import errmsg


# noinspection PyUnresolvedReferences
class Moon(SkyLocation):
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, obstime):
        """
        Classe che descrive il moto della Luna in funzione della data di osservazione.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.__mass_earth = cts.M_earth
        self.__grav_const = cts.G * self.__mass_earth / (4 * np.pi ** 2)

        self.__semimaj = None
        self.__ecc = None
        self.__argument_pericenter = None
        self.__mean_anomaly = None
        self.__eccentric_anomaly = None
        self.__inclination = None
        self.__longitude_an = None

        self.at_date(obstime)

        super(Moon, self).__init__(locstring=None,
                                   ra=self.vector_obstime.ra, dec=self.vector_obstime.dec,
                                   distance=self.distance_from_earth, obstime=obstime,
                                   ra_unit='deg', dec_unit='deg', epoch='J2000', name='Moon')

    def observe_at_date(self, obstime):
        """
        Metodo che calcola, senza salvarne il risultato, la posizione della Luna ad una specifica data.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        ts = load.timescale()
        t = ts.from_astropy(obstime)
        planets = load('de421.bsp')
        moon = planets['moon']
        earth = planets['earth']
        position = (moon - earth).at(t)
        elements = osculating_elements_of(position)

        self.__semimaj = elements.semi_major_axis.to(u.m)
        self.__ecc = elements.eccentricity
        self.__inclination = Angle(elements.inclination.to(u.deg))
        self.__longitude_an = Angle(elements.longitude_of_ascending_node.to(u.deg))
        self.__argument_pericenter = Angle(elements.argument_of_periapsis.to(u.deg))
        self.__mean_anomaly = Angle(elements.mean_anomaly.to(u.deg))
        self.__eccentric_anomaly = self.__approx_ecc_anomaly()

        vector_obstime = Versor(vector=np.array([1, 0, 0]) * self.distance_from_earth) \
            .rotate(axis='z', angle=self.true_anomaly, copy=True) \
            .rotate(axis='z', angle=self.argument_pericenter, copy=True) \
            .rotate(axis='x', angle=self.inclination, copy=True) \
            .rotate(axis='z', angle=self.longitude_an, copy=True)

        return vector_obstime

    def at_date(self, obstime):
        """
        Metodo che salva il risultato di `Moon.observe_at_date` nei relativi attributi di classe.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime

        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

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

    def calculate_moon_phase(self, sun, obstime):
        """
        Metodo che calcola la fase della Luna, dato il Sole. Notare che non vi è alcuna differenza tra fase crescente e
        calante. Il metodo restituisce `1` se la Luna è piena e `0` se è nuova.
        """
        if not isinstance(sun, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('sun', 'src.skylocation.sun.Sun'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        moon_obstime = self.observe_at_date(obstime)
        sun_obstime = sun.observe_at_date(obstime)

        # Calcolo del prodotto scalare dei vettori, rinormalizzato per dare il valore richiesto. Nel caso di Luna Nuova,
        # il prodotto scalare varrebbe `1`, e `-1` nel caso di Luna Piena. In questo modo si restituiscono i valori
        # dichiarati nella descrizione del metodo.
        return (- moon_obstime.vsr.dot(sun_obstime.vsr) + 1) / 2

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
        return self.semimaj * np.sqrt(1 - self.eccentricity ** 2)

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
    def distance_from_earth(self):
        return self.semimaj * (1 - self.eccentricity * np.cos(self.eccentric_anomaly))
