import numpy as np

from astropy import units as u
from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude

from .time import Time


Tsidday = (23.9345 * u.hour).to(u.day)
Tsidyear = 365.256363004 * u.day  # https://hpiers.obspm.fr/eop-pc/models/constants.html
Tprec = (26000 * u.year).to(u.day)
tJ2000 = Time('J2000.0')


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
        return hms * 15 * u.deg
    elif isinstance(hms, u.quantity.Quantity) and hms.unit == "h":
        return hms * 360*u.deg / (24 * u.hour)
    elif isinstance(hms, (np.ndarray, tuple, list)) and len(hms) == 3:
        return (hms[0] + hms[1] + hms[2]).to(u.hour).value * 15 * u.deg


def dms2deg(dms):
    if not isinstance(dms, str):
        raise TypeError("`dms` must be a string.")

    deg, minsec = dms.lower().split('d')
    mins, sec = minsec.lower().split('m')
    sec = sec[:-1]

    deg = float(deg) * u.deg
    mins = float(mins) / 60 * u.arcmin
    sec = float(sec) / 3600 * u.arcsec

    if deg.value >= 0:
        return deg + mins + sec
    else:
        return deg - mins - sec


def open_loc_file(obs_path, tgt_path):
    with open(obs_path, "r") as f:
        loc, obstime = f.readlines()

    with open(tgt_path, "r") as f:
        tgt = f.readlines()

    return loc, obstime, tgt


class GMSTeq2000:
    hms = 19 * u.hour + 17 * u.min + 57.3258 * u.s
    deg = hms2deg(hms)
    rad = deg.to(u.rad)


class Equinox2000:
    """
    ssd.jpl.nasa.gov/horizons.cgi

    [For Equinox time]
    ephemType=OBSERVER
    Target=Sun
    Location=Geocentric
    TimeSpan: 2000-03-20 07:25:00 - 2000-03-20 07:26:00, intervals=100
    settings=default
    display=default (HTML)

    [For GMST at equinox]
    ephemType=OBSERVER
    Target=Sun
    Location=Greenwich [000] ( 0°00'00.0''E, 51°28'38.6''N, 65.8 m )
    TimeSpan: 2000-03-20 07:25:00 - 2000-03-20 07:26:00, intervals=100
    settings=QUANTITIES=1,7,9,20,23,24
    display=default (HTML)
    """
    time = Time('2000-03-20T07:25:24.600', scale='utc', format='isot')
    GMST = GMSTeq2000
    hour = time.jd % 1
    rad = 2 * np.pi * hour


class Versor:
    def __init__(self, ra=None, dec=None, vector=None, unit=None):
        if (ra is None and dec is None) and vector is None:
            raise ValueError("Must give either a set of coordinates or a ra-dec position.")

        if vector is None:
            if not isinstance(ra, u.quantity.Quantity) and not isinstance(dec, u.quantity.Quantity):
                if unit is None:
                    raise ValueError("Must specify the unit of measure.")
                elif unit not in ['deg', 'rad', 'hmsdms']:
                    raise ValueError("Must use a valid unit of measure.")

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

        if copy:
            return Versor(vector=vsr)
        else:
            self.vsr = vsr
            return self

    def rotate_inv(self, axis, angle, unit='rad', copy=False):
        r_mat = RotationMatrix(axis, angle, unit)
        vsr = r_mat.inv.dot(self.vsr)

        if copy:
            return Versor(vector=vsr)
        else:
            self.vsr = vsr
            return self


class RotationMatrix:
    def __init__(self, axis, angle, unit='deg'):
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError("Not a valid rotation axis.")
        if unit not in ['deg', 'rad'] and\
                (not isinstance(angle, u.quantity.Quantity) and angle.unit not in ['deg', 'rad']):
            raise ValueError("Unknown angle unit.")

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
            raise ValueError("Invalid axis")


import src.location
import src.observation
import src.time
import src.skylocation
import src.skylocation.sun
