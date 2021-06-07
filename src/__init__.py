__version__ = "1.4.0-a.1"
__author__ = "Daniele Sanfratello"

import logging
import numpy as np

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.coordinates import Angle
from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude

from skyfield import api
from skyfield import almanac

from .time import Time

from . import warnmsg
from . import errmsg


Tsidday = (23.93447192 * u.hour).to(u.day)  # Expl. Suppl. p698 (Sidereal year in 1990)
Tsidyear = 365.256363004 * u.day  # https://hpiers.obspm.fr/eop-pc/models/constants.html
Jyear = 365.25 * u.day

# DOI:10.1016/j.pss.2006.06.003 and DOI:10.1051/0004-6361:20021912
Tprec = ((1/(50287.9226200 * u.arcsec / (1000 * u.year))).to(u.year/u.deg) * 360 * u.deg).to(u.day)

Omegasidmoon = (2.661699489e-6 * u.rad / u.s).to(u.rad / u.d)  # Expl. Suppl. p701 (Revolution frequency of Moon)

moon_incl_to_ecliptic = 5.145396 * u.deg  # Expl. Suppl. p701

tJ2000 = Time('J2000.0')
sidday_diff = 1 * u.day - Tsidday

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logger.getEffectiveLevel())
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def hms2deg(hms):
    """
    Funzione che prende un angolo orario, in formato "[hh]h[mm]m[ss]s", numerico, tupla/lista/np.ndarray o
     `astropy.units.quantity.Quantity` e la converte in gradi, restituendo un angolo.
    """
    if isinstance(hms, str):
        hour, minsec = hms.lower().split('h')
        mins, sec = minsec.lower().split('m')
        sec = sec[:-1]

        hour = float(hour)
        mins = float(mins) / 60
        sec = float(sec) / 3600

        return (hour + mins + sec) * 15 * u.deg
    elif isinstance(hms, (int, float)):
        return Angle(hms * 15 * u.deg)
    elif isinstance(hms, u.quantity.Quantity) and hms.unit == "h":
        return Angle(hms * 360*u.deg / (24 * u.hour))
    elif isinstance(hms, (np.ndarray, tuple, list)) and len(hms) == 3:
        return Angle((hms[0] + hms[1] + hms[2]).to(u.hour).value * 15 * u.deg)


def dms2deg(dms):
    """
    Funzione che prende un angolo, in formato "[ddd]d[mm]m[ss]s", e lo converte in gradi, restituendo un angolo.
    """
    if not isinstance(dms, str):
        raise TypeError(errmsg.notTypeError.format('dms', 'string'))

    deg, minsec = dms.lower().split('d')
    mins, sec = minsec.lower().split('m')
    sec = sec[:-1]

    deg = float(deg) * u.deg
    mins = float(mins) * u.arcmin
    sec = float(sec) * u.arcsec

    if deg.value >= 0:
        return Angle(deg + mins + sec)
    else:
        return Angle(deg - mins - sec)


def open_loc_file(obs_path, tgt_path):
    """
    Funzione che dato un percorso e un file, apre quest'ultimo per generare le stringhe che identificano località,
    orario di osservazione e target.
    """
    with open(obs_path, "r") as f:
        loc, obstime = f.readlines()

    with open(tgt_path, "r") as f:
        tgt = f.readlines()

    return loc, obstime, tgt


class GMSTeq2000:
    def __init__(self, obstime):
        """
        Classe che identifica il Greenwich Mean Sidereal Time all'equinozio vernale del 2000.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.hms = self.__set_time(obstime)
        self.deg = self.hms.to(u.deg)
        self.rad = self.hms.to(u.rad)

    @staticmethod
    def __set_time(obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        obstime.location = EarthLocation.of_site('greenwich')
        return obstime.sidereal_time('mean')


class Equinox2000:
    """
    Costante che definisce l'equinozio vernale del 2000, calcolandolo dalle effemeridi de421.
    """
    ts = api.load.timescale()
    eph = api.load('de421.bsp')
    t0 = ts.utc(2000, 3, 20)
    t1 = ts.utc(2000, 3, 21)
    t = almanac.find_discrete(t0, t1, almanac.seasons(eph))[0]
    eph.close()

    time = Time(t.utc_iso()[0][:-1], scale='utc')

    GMST = GMSTeq2000(time)
    hour = time.jd % 1
    rad = 2 * np.pi * hour


class Eclipse1999:
    """
    Costante che definisce l'eclissi di sole dell'agosto 1999, calcolandolo dalle effemeridi de421.
    """
    ts = api.load.timescale()
    eph = api.load('de421.bsp')
    t0 = ts.utc(1999, 8, 1)
    t1 = ts.utc(1999, 8, 15)
    t = almanac.find_discrete(t0, t1, almanac.moon_nodes(eph))[0]
    eph.close()

    time = Time(t.utc_iso()[0][:-1], scale='utc')


class Versor:
    def __init__(self, ra=None, dec=None, radius=None, vector=None, unit=None):
        """
        Classe che accetta delle coordinate equatoriali o un vettore in coordinate cartesiane e genera il versore che
        punta quelle coordinate.
        """
        self.__logger = logging.getLogger('src.Versor')
        self.__logger.setLevel(logging.DEBUG)

        if (ra is None and dec is None) and vector is None:
            raise ValueError(errmsg.versorError)

        if vector is None:
            if not isinstance(ra, u.quantity.Quantity) and not isinstance(dec, u.quantity.Quantity):
                if unit is None:
                    self.__logger.critical('Error!')
                    raise ValueError(errmsg.specifyUnitError)
                elif unit not in ['deg', 'rad', 'hmsdms']:
                    self.__logger.critical('Error!')
                    raise ValueError(errmsg.invalidUnitError)

        if ra is not None and dec is not None:
            # Se vengono fornite le coordinate equatoriali, vengono opportunamente salvate in un attributo della classe.
            if unit == 'deg':
                self.ra = ra * u.deg
                self.dec = dec * u.deg
            elif unit == 'rad':
                self.ra = np.rad2deg(ra) * u.deg
                self.dec = np.rad2deg(dec) * u.deg
            elif unit == 'hmsdms':
                self.ra = hms2deg(ra) * u.deg
                self.dec = dms2deg(dec) * u.deg
            else:
                self.ra = ra
                self.dec = dec

            # Se le coordinate equatoriali fornite Non sono istanze della classe `astropy.coordinates.angles.Latitude` o
            # `astropy.coordinates.angles.Longitude`, vengono convertite in queste coordinate. Tale conversione serve
            # a semplificare le operazioni nell'esecuzione delle operazioni.
            if not isinstance(dec, Latitude):
                self.dec = Latitude(self.dec)
            if not isinstance(ra, Longitude):
                self.ra = Longitude(self.ra)

            self.__logger.debug(f'Generating vector with ra-dec: {self.ra}, {self.dec}')

            ra = self.ra.rad
            dec = self.dec.rad

            # Se non viene fornito una dimensione radiale, si considera il versore come unitario.
            if radius is None:
                self.radius = 1
            else:
                self.radius = radius
            self.__logger.debug(f'Vector has radius: {self.radius}')

            # Date le coordinate equatoriali, viene generato il vettore che punti quelle coordinate. L'asse x è il punto
            # vernale, l'asse z indica il Polo Nord Celeste e l'asse y è definito per completare la terna ortogonale.
            self.vsr = np.array([np.cos(dec) * np.cos(ra),
                                 np.cos(dec) * np.sin(ra),
                                 np.sin(dec)], dtype=np.float64)
            self.__logger.debug(f'Vector has versor: {self.vsr}')
        else:
            # Se viene fornito il vettore, questo viene intanto normalizzato, salvando la norma dello stesso.
            self.radius = np.sqrt((vector**2).sum())
            self.__logger.debug(f'Generating vector with radius: {self.radius}')
            if self.radius != 0:
                self.vsr = np.copy(vector) / self.radius
                self.__logger.debug(f'Vector has versor: {self.vsr}')

                # Dalle coordinate si ricavano RA e DEC e si convertono in oggetti `Longitude` e `Latitude`,
                # rispettivamente.
                self.ra = np.arctan2(self.vsr[1], self.vsr[0])
                self.dec = np.arctan2(self.vsr[2], np.sqrt(self.vsr[0]**2 + self.vsr[1]**2))
            else:
                self.vsr = np.zeros(3)
                self.__logger.debug(f'Vector has versor: {self.vsr}')

                self.ra = 0 * u.deg
                self.dec = 0 * u.deg


        if isinstance(self.ra, u.Quantity):
            self.ra = self.ra.to(u.deg)
            self.ra = Longitude(self.ra)
        else:
            self.ra = Longitude(np.rad2deg(self.ra) * u.deg)
        if isinstance(self.dec, u.Quantity):
            self.dec = self.dec.to(u.deg)
            self.dec = Latitude(self.dec)
        else:
            self.dec = Latitude(np.rad2deg(self.dec) * u.deg)
        self.__logger.debug(f'Vector has ra-dec: {self.ra}, {self.dec}')

    def rotate(self, axis, angle, unit='rad', copy=False):
        """
        Metodo della classe `Versor` che applica una matrice di rotazione intorno all'asse cartesiano `axis`, di un
        angolo `angle` e, nel caso in cui `copy=True`, restituisce una copia del versore, altrimenti salva le modifiche
        nell'oggetto a cui tal modifica è stata applicata.
        """
        r_mat = RotationMatrix(axis, angle, unit)
        vsr = r_mat.mat.dot(self.vsr)

        vsr /= np.sqrt((vsr**2).sum())

        self.__logger.debug(f'Rotating {self.vsr} around {axis} by {angle}')
        if copy:
            self.__logger.debug('Returning new versor.')
            return Versor(vector=vsr * self.radius)
        else:
            self.vsr = vsr

    def rotate_inv(self, axis, angle, unit='rad', copy=False):
        """
        Metodo della classe `Versor` che applica una matrice di rotazione inversa intorno all'asse cartesiano `axis`,
        di un angolo `angle` e, nel caso in cui `copy=True`, restituisce una copia del versore, altrimenti salva le
        modifiche nell'oggetto a cui tal modifica è stata applicata.
        """
        r_mat = RotationMatrix(axis, angle, unit)
        vsr = r_mat.inv.dot(self.vsr)

        vsr /= np.sqrt((vsr**2).sum())

        self.__logger.debug(f'Rotating {self.vsr} around {axis} by {angle}')
        if copy:
            self.__logger.debug('Returning new versor.')
            return Versor(vector=vsr * self.radius)
        else:
            self.vsr = vsr

    def __add__(self, add_vcr):
        if not isinstance(add_vcr, Versor):
            raise TypeError(errmsg.notTypeError.format('add_vcr', 'Versor'))

        return Versor(vector=self.vsr * self.radius + add_vcr.vsr * add_vcr.radius)

    def __sub__(self, sub_vcr):
        if not isinstance(sub_vcr, Versor):
            raise TypeError(errmsg.notTypeError.format('sub_vcr', 'Versor'))

        return Versor(vector=self.vsr * self.radius - sub_vcr.vsr * sub_vcr.radius)


class RotationMatrix:
    def __init__(self, axis, angle, unit='deg'):
        """
        Classe che definisce una matrice di rotazione di angolo `angle` intorno all'asse `axis`, che deve essere uno dei
        tre assi coordinati.
        """
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError(errmsg.invalidAxisError)
        if unit not in ['deg', 'rad'] and\
                (not isinstance(angle, u.quantity.Quantity) and angle.unit not in ['deg', 'rad']):
            raise ValueError(errmsg.invalidUnitError)

        if isinstance(angle, u.quantity.Quantity):
            unit = angle.unit
        else:
            if unit == 'deg':
                angle *= u.deg
            elif unit == 'rad':
                angle *= u.rad

        self.axis = axis
        self.angle = angle
        self.unit = unit

        angle = self.angle.to(u.rad)

        self.mat = self.matrix(axis, angle)
        self.inv = self.matrix(axis, -angle)

    @staticmethod
    def matrix(axis, angle):
        """
        Metodo statico che definisce esplicitamente la matrice di rotazione, dato l'asse `axis`.
        """
        if axis == 'x':
            return np.array([[1,             0,              0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle),  np.cos(angle)]], dtype=np.float64)
        elif axis == 'y':
            return np.array([[ np.cos(angle), 0, np.sin(angle)],
                             [             0, 1,             0],
                             [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float64)
        elif axis == 'z':
            return np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle),  np.cos(angle), 0],
                             [             0,             0, 1]], dtype=np.float64)
        else:
            raise ValueError(errmsg.invalidAxisError)


from . import location
from . import observation
from . import time
from . import skylocation
from . import errmsg
from . import warnmsg
