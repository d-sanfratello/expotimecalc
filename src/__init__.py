__version__ = "1.2.0"
__author__ = "Daniele Sanfratello"

import logging
import numpy as np

from astropy import units as u
from astropy.units.quantity import Quantity
from astropy.coordinates import Angle
from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude

from .time import Time
from .matrix import RotationMatrix
from .matrix import Rx, Ry, Rz

from . import warnmsg
from . import errmsg

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
            if not isinstance(ra, Quantity) and not isinstance(dec, Quantity):
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

        if isinstance(self.ra, Quantity):
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
        if axis == 'x':
            r_mat = Rx
        elif axis == 'y':
            r_mat = Ry
        elif axis == 'z':
            r_mat = Rz
        else:
            raise ValueError(errmsg.invalidAxisError)

        r_mat.unit = unit
        r_mat.angle = angle

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
        if axis == 'x':
            r_mat = Rx
        elif axis == 'y':
            r_mat = Ry
        elif axis == 'z':
            r_mat = Rz
        else:
            raise ValueError(errmsg.invalidAxisError)

        r_mat.unit = unit
        r_mat.angle = angle

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


from . import constants
from . import location
from . import observation
from . import time
from . import skylocation
