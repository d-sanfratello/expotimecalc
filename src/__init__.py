import numpy as np

# from astropy import units as u


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
    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec

        self.vsr = np.array([np.cos(dec)*np.cos(ra),
                             np.cos(dec)*np.sin(ra),
                             np.sin(dec)], dtype=np.float64)


class RotationMatrix:
    def __init__(self, axis, angle, unit='deg'):
        if axis is not in ['x', 'y', 'z']:
            raise ValueError("Not a valid rotation axis.")
        if unit is not in ['deg', 'rad']:
            raise ValueError("Unknown angle unit.")

        self.axis = axis
        self.angle = angle
        self.unit = unit

        if unit == 'deg':
            angle *= (2*np.pi/360)

        self.mat = self.matrix(axis, angle)

    def matrix(self, axis, angle):
        if axis == 'x':
            pass
        elif axis == 'y':
            pass
        else:
            return np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle),  np.cos(angle), 0],
                             [             0,             0, 1]])
