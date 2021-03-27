import numpy as np

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.coordinates import Angle
from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude

from skyfield import api
from skyfield import almanac

from src.time import Time

import src.warnmsg as warnmsg
import src.errmsg as errmsg


Tsidday = (23.9345 * u.hour).to(u.day)
Tsidyear = 365.256363004 * u.day  # https://hpiers.obspm.fr/eop-pc/models/constants.html
Tprec = 25770 * Tsidyear  # DOI:10.1016/j.pss.2006.06.003 and DOI:10.1051/0004-6361:20021912
Omegasidmoon = (2.661699489e-6 * u.rad / u.s).to(u.rad / u.d)  # Expl. Suppl. p701 (Revolution frequency of Moon)

moon_incl_to_ecliptic = 5.145396 * u.deg  # Expl. Suppl. p701

tJ2000 = Time('J2000.0')
sidday_diff = 1 * u.day - Tsidday


def hms2deg(hms):
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
    with open(obs_path, "r") as f:
        loc, obstime = f.readlines()

    with open(tgt_path, "r") as f:
        tgt = f.readlines()

    return loc, obstime, tgt


class GMSTeq2000:
    def __init__(self, time):
        if not isinstance(time, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.hms = self.__set_time(time)
        self.deg = self.hms.to(u.deg)
        self.rad = self.hms.to(u.rad)

    @staticmethod
    def __set_time(time):
        if not isinstance(time, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        time.location = EarthLocation.of_site('greenwich')
        return time.sidereal_time('mean')


class Equinox2000:
    ts = api.load.timescale()
    eph = api.load('de421.bsp')
    t0 = ts.utc(2000, 3, 20)
    t1 = ts.utc(2000, 3, 21)
    t = almanac.find_discrete(t0, t1, almanac.seasons(eph))[0]

    time = Time(t.utc_iso()[0][:-1], scale='utc')

    GMST = GMSTeq2000(time)
    hour = time.jd % 1
    rad = 2 * np.pi * hour


class Eclipse1999:
    ts = api.load.timescale()
    eph = api.load('de421.bsp')
    t0 = ts.utc(1999, 8, 1)
    t1 = ts.utc(1999, 8, 15)
    t = almanac.find_discrete(t0, t1, almanac.moon_nodes(eph))[0]

    time = Time(t.utc_iso()[0][:-1], scale='utc')


class Versor:
    def __init__(self, ra=None, dec=None, vector=None, unit=None):
        if (ra is None and dec is None) and vector is None:
            raise ValueError(errmsg.versorError)

        if vector is None:
            if not isinstance(ra, u.quantity.Quantity) and not isinstance(dec, u.quantity.Quantity):
                if unit is None:
                    raise ValueError(errmsg.specifyUnitError)
                elif unit not in ['deg', 'rad', 'hmsdms']:
                    raise ValueError(errmsg.invalidUnitError)

        if ra is not None and dec is not None:
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

            if not isinstance(dec, Latitude):
                self.dec = Latitude(self.dec)
            if not isinstance(ra, Longitude):
                self.ra = Longitude(self.ra)

            ra = self.ra.rad
            dec = self.dec.rad

            self.vsr = np.array([np.cos(dec)*np.cos(ra),
                                 np.cos(dec)*np.sin(ra),
                                 np.sin(dec)], dtype=np.float64)
        else:
            self.vsr = np.copy(vector)/np.sqrt((vector**2).sum())

            self.ra = np.arctan2(self.vsr[1], self.vsr[0])
            self.dec = np.arctan2(self.vsr[2], np.sqrt(self.vsr[0]**2 + self.vsr[1]**2))

            self.ra = Longitude(np.rad2deg(self.ra) * u.deg)
            self.dec = Latitude(np.rad2deg(self.dec) * u.deg)

    def rotate(self, axis, angle, unit='rad', copy=False):
        r_mat = RotationMatrix(axis, angle, unit)
        vsr = r_mat.mat.dot(self.vsr)

        vsr /= np.sqrt((vsr**2).sum())

        if copy:
            return Versor(vector=vsr)
        else:
            self.vsr = vsr

    def rotate_inv(self, axis, angle, unit='rad', copy=False):
        r_mat = RotationMatrix(axis, angle, unit)
        vsr = r_mat.inv.dot(self.vsr)

        vsr /= np.sqrt((vsr**2).sum())

        if copy:
            return Versor(vector=vsr)
        else:
            self.vsr = vsr


class RotationMatrix:
    def __init__(self, axis, angle, unit='deg'):
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


import src.location
import src.observation
import src.time
import src.skylocation
import src.skylocation.sun
import src.errmsg
import src.warnmsg
