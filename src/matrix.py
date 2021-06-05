import numpy as np

from astropy import units as u
from astropy.units.quantity import Quantity

from . import errmsg


class RotationMatrix:
    def __init__(self, axis, angle=None, unit='deg'):
        """
        Classe che definisce una matrice di rotazione di angolo `angle` intorno all'asse `axis`, che deve essere uno dei
        tre assi coordinati.
        """
        if axis.lower() not in ['x', 'y', 'z']:
            raise ValueError(errmsg.invalidAxisError)
        if unit not in ['deg', 'rad'] and \
                (not isinstance(angle, u.quantity.Quantity) and angle.unit not in ['deg', 'rad']):
            raise ValueError(errmsg.invalidUnitError)

        if isinstance(angle, u.quantity.Quantity):
            unit = angle.unit
        elif angle is not None:
            if unit == 'deg':
                angle *= u.deg
            elif unit == 'rad':
                angle *= u.rad

        self.__axis = axis
        self.__angle = angle
        self.__unit = unit

    @property
    def axis(self):
        return self.__axis

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, new):
        if self.unit not in ['deg', 'rad']:
            raise ValueError(errmsg.invalidUnitError)

        self.__unit = new

    @property
    def angle(self):
        return self.__angle.to(u.rad)

    @angle.setter
    def angle(self, new):
        if isinstance(new, Quantity):
            self.unit = new.unit
            self.__angle = new
        else:
            if self.unit == 'deg':
                self.__angle *= u.deg
            elif self.unit == 'rad':
                self.__angle *= u.rad

    @property
    def mat(self):
        return self._matrix(self.axis, self.angle)

    @property
    def inv(self):
        return self._matrix(self.axis, -self.angle)

    @staticmethod
    def _matrix(axis, angle):
        """
        Metodo statico che definisce esplicitamente la matrice di rotazione, dato l'asse `axis`.
        """
        if angle is None:
            raise ValueError(errmsg.undefinedAngle)

        if axis == 'x':
            return np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]], dtype=np.float64)
        elif axis == 'y':
            return np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float64)
        elif axis == 'z':
            return np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]], dtype=np.float64)
        else:
            raise ValueError(errmsg.invalidAxisError)


Rx = RotationMatrix(axis='x')
Ry = RotationMatrix(axis='y')
Rz = RotationMatrix(axis='z')
