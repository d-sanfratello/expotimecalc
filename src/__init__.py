import numpy as np

from astropy import units as u
from .time import Time


Tsidday = (23.9345 * u.hour).to(u.day)
Tprec = (26000 * u.year).to(u.day)
tJ2000 = Time('J2000.0')


def str2dms(string):
    deg, minsec = string.lower().split('d')
    mins, sec = minsec.lower().split('m')
    sec = sec[:-1]

    deg = float(deg)
    mins = float(mins) / 60
    sec = float(sec) / 3600

    return deg + mins + sec


def hms2dms(hms):
    if isinstance(hms, str):
        hour, minsec = hms.lower().split('h')
        mins, sec = minsec.lower().split('m')
        sec = sec[:-1]

        hour = float(hour)
        mins = float(mins) / 60
        sec = float(sec) / 3600

        return (hour + mins + sec) * 15
    elif isinstance(hms, (int, float)):
        return hms * 15
    elif isinstance(hms, u.quantity.Quantity) and hms.unit == "h":
        return hms * 360*u.deg / (24 * u.hour)


def dms2dec(dms):
    if not isinstance(dms, str):
        raise TypeError("`dms` must be a string.")

    deg, minsec = dms.lower().split('d')
    mins, sec = minsec.lower().split('m')
    sec = sec[:-1]

    deg = float(deg)
    mins = float(mins) / 60
    sec = float(sec) / 3600

    return deg + mins + sec


def open_loc_file(obs_path, tgt_path):
    with open(obs_path, "r") as f:
        loc, obstime = f.readlines()

    with open(tgt_path, "r") as f:
        tgt = f.readlines()

    return loc, obstime, tgt


class GMSTeq2000:
    hms = 19 * u.hour + 17 * u.min + 57.3258 * u.s
    deg = hms2dms(hms)
    rad = deg.to(u.rad)


class eq2000:
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
    def __init__(self, ra=None, dec=None, vector=None, unit='deg'):
        if unit not in ['deg', 'rad', 'hmsdms']:
            raise ValueError("Must use a valid unit of measure.")

        if (ra is None and dec is None) and vector is None:
            raise ValueError("Must give either a set of coordinates or a ra-dec position.")

        if ra is not None and dec is not None:
            if unit == 'deg':
                self.ra = ra
                self.dec = dec
            elif unit == 'rad':
                self.ra = np.rad2deg(ra)
                self.dec = np.rad2deg(dec)
            else:
                self.ra = hms2dms(ra)
                self.dec = dms2dec(dec)

            ra = np.deg2rad(self.ra)
            dec = np.deg2rad(self.dec)

            self.vsr = np.array([np.cos(dec)*np.cos(ra),
                                 np.cos(dec)*np.sin(ra),
                                 np.sin(dec)], dtype=np.float64)
        else:
            self.vsr = np.copy(vector)/np.sqrt((vector**2).sum())

            self.ra = np.arctan2(self.vsr[1], self.vsr[0])
            self.dec = np.arctan2(self.vsr[2], np.sqrt(self.vsr[0]**2 + self.vsr[1]**2))

            self.ra = np.rad2deg(self.ra)
            self.dec = np.rad2deg(self.dec)

    def rotate(self, axis, angle, unit='rad', copy=False):
        r_mat = RotationMatrix(axis, angle, unit)
        self.vsr = r_mat.mat.dot(self.vsr)

        if copy:
            return Versor(vector=self.vsr)
        else:
            return self

    def rotate_inv(self, axis, angle, unit='rad', copy=False):
        r_mat = RotationMatrix(axis, angle, unit)
        self.vsr = r_mat.inv.dot(self.vsr)

        if copy:
            return Versor(vector=self.vsr)
        else:
            return self


class RotationMatrix:
    def __init__(self, axis, angle, unit='deg'):
        if axis not in ['x', 'y', 'z']:
            raise ValueError("Not a valid rotation axis.")
        if unit not in ['deg', 'rad']:
            raise ValueError("Unknown angle unit.")

        self.axis = axis
        self.angle = angle
        self.unit = unit

        if unit == 'deg':
            self.angle *= (2*np.pi/360)

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
