import numpy as np

from astropy import units as u


Tsidday = 23.9345 * u.hour
# GMST2000 =


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


def open_loc_file(obs_path, tgt_path):
    with open(obs_path, "r") as f:
        loc, obstime = f.readlines()

    with open(tgt_path, "r") as f:
        tgt = f.readlines()

    return loc, obstime, tgt


class Versor:
    def __init__(self, ra=None, dec=None, vector=None):
        if (ra is None and ra is None) and vector is None:
            raise ValueError("Must give either a set of coordinates or a ra-dec position.")

        if ra is not None and dec is not None:
            self.ra = ra
            self.dec = dec

            self.vsr = np.array([np.cos(dec)*np.cos(ra),
                                 np.cos(dec)*np.sin(ra),
                                 np.sin(dec)], dtype=np.float64)
        else:
            self.vsr = np.copy(vector/np.sqrt((vector**2).sum()))

            self.ra = np.arctan2(self.vsr[1]/self.vsr[0])
            self.dec = np.arctan2(self.vsr[2]/np.sqrt(self.vsr[0]**2 + self.vsr[1]**2))

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
