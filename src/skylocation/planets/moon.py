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

    def observe_at_date(self, obstime, return_complete=False):
        """
        Metodo che calcola, senza salvarne il risultato, la posizione della Luna ad una specifica data.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(return_complete, bool):
            raise TypeError(errmsg.notTypeError.format('return_complete', 'bool'))

        ts = load.timescale()
        t = ts.from_astropy(obstime)
        planets = load('de421.bsp')
        moon = planets['moon']
        earth = planets['earth']
        position = (moon - earth).at(t)
        elements = osculating_elements_of(position)

        semimaj = elements.semi_major_axis.to(u.m)
        inclination = Angle(elements.inclination.to(u.deg))
        argument_pericenter = Angle(elements.argument_of_periapsis.to(u.deg))
        longitude_an = Angle(elements.longitude_of_ascending_node.to(u.deg))
        eccentricity = elements.eccentricity
        mean_anomaly = Angle(elements.mean_anomaly.to(u.deg))
        eccentric_anomaly = self.__approx_ecc_anomaly(mean_anomaly=mean_anomaly, eccentricity=eccentricity)
        true_anomaly = self.__true_anomaly(eccentric_anomaly=eccentric_anomaly, eccentricity=eccentricity)
        distance_from_earth = self.__distance_from_center(semimaj=semimaj,
                                                          eccentric_anomaly=eccentric_anomaly,
                                                          eccentricity=eccentricity)

        vector_obstime = Versor(vector=np.array([1, 0, 0]) * distance_from_earth) \
            .rotate(axis='z', angle=true_anomaly, copy=True) \
            .rotate(axis='z', angle=argument_pericenter, copy=True) \
            .rotate(axis='x', angle=inclination, copy=True) \
            .rotate(axis='z', angle=longitude_an, copy=True)

        if return_complete:
            orb_pars = {'semimaj': semimaj,
                        'inclination': inclination,
                        'longitude an': longitude_an,
                        'argument pericenter': argument_pericenter,
                        'eccentricity': eccentricity,
                        'mean anomaly': mean_anomaly,
                        'eccentric anomaly': eccentric_anomaly,
                        'true anomaly': true_anomaly,
                        'distance from earth': distance_from_earth}
            return vector_obstime, orb_pars
        else:
            return vector_obstime

    def at_date(self, obstime):
        """
        Metodo che salva il risultato di `Moon.observe_at_date` nei relativi attributi di classe.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime

        self.vector_obstime, orb_pars = self.observe_at_date(obstime, return_complete=True)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

        self.__semimaj = orb_pars['semimaj']
        self.__inclination = orb_pars['inclination']
        self.__longitude_an = orb_pars['longitude an']
        self.__argument_pericenter = orb_pars['argument pericenter']
        self.__ecc = orb_pars['eccentricity']
        self.__mean_anomaly = orb_pars['mean anomaly']
        self.__eccentric_anomaly = orb_pars['eccentric anomaly']

    def __approx_ecc_anomaly(self, mean_anomaly=None, eccentricity=None):
        if mean_anomaly is not None and eccentricity is None:
            raise ValueError(errmsg.notAllDeclared)
        elif mean_anomaly is None and eccentricity is not None:
            raise ValueError(errmsg.notAllDeclared)

        if mean_anomaly is None and eccentricity is None:
            mean = self.mean_anomaly.to(u.rad)
            ecc = self.eccentricity
        else:
            mean = mean_anomaly.to(u.rad)
            ecc = eccentricity
        # Method:
        #  1998aalg.book.....M (p196)

        ecc_an = mean
        ecc_an_next = mean - ecc * np.sin(ecc_an) * u.rad
        while abs(ecc_an_next.to(u.deg) - ecc_an.to(u.deg)) > (1e-4 * u.arcsec).to(u.deg):
            ecc_an = ecc_an_next
            ecc_an_next = mean - ecc * np.sin(ecc_an) * u.rad

        return Angle(ecc_an_next.to(u.deg))

    def __true_anomaly(self, eccentric_anomaly=None, eccentricity=None):
        if eccentric_anomaly is not None and eccentricity is None:
            raise ValueError(errmsg.notAllDeclared)
        elif eccentric_anomaly is None and eccentricity is not None:
            raise ValueError(errmsg.notAllDeclared)

        if eccentric_anomaly is None and eccentricity is None:
            eccentric_anomaly = self.eccentric_anomaly
            eccentricity = self.eccentricity

        true_an = (2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity))  * np.tan(eccentric_anomaly / 2)))
        return true_an.to(u.deg)

    def __distance_from_center(self, semimaj=None, eccentric_anomaly=None, eccentricity=None):
        if semimaj is not None and (eccentric_anomaly is None or eccentricity is None):
            raise ValueError(errmsg.notAllDeclared)
        elif eccentric_anomaly is not None and (semimaj is None or eccentricity is None):
            raise ValueError(errmsg.notAllDeclared)
        elif eccentricity is not None and (semimaj is None or eccentric_anomaly is None):
            raise ValueError(errmsg.notAllDeclared)

        if semimaj is None and eccentric_anomaly is None and eccentricity is None:
            semimaj = self.semimaj
            eccentric_anomaly = self.eccentric_anomaly
            eccentricity = self.eccentricity

        return semimaj * (1 - eccentricity * np.cos(eccentric_anomaly))

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
        return self.__true_anomaly()

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
        return self.__distance_from_center()
