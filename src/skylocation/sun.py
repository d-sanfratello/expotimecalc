import numpy as np

from astropy import units as u

from src.time import Time
from src.skylocation import SkyLocation

from src import Tsidyear
from src import tJ2000


class Sun(SkyLocation):
    def __init__(self, obstime):
        super(Sun, self).__init__(locstring=None, ra=0*u.hour, dec=0*u.hour, obstime=obstime,
                                  ra_unit='hour', dec_unit='deg', epoch='J2000', name='Sun')

        self.vector_obstime = self.observe_at_date(obstime, copy=True)

    def observe_at_date(self, obstime, copy=True):
        if isinstance(obstime, Time):
            self.obstime = obstime.utc
        else:
            self.obstime = Time(obstime).utc

        vector_obstime = self.vector_epoch.rotate('x', self.nutation_corr(self.obstime), copy=True)\
            .rotate('z', self.sidereal_year(self.obstime), copy=True)\
            .rotate_inv('x', self.nutation_corr(self.obstime), copy=True)

        if copy:
            return vector_obstime
        else:
            self.vector_obstime = vector_obstime

    @staticmethod
    def sidereal_year(obstime):
        return ((2 * np.pi / Tsidyear.value) * (obstime - tJ2000).jd) % (2*np.pi) * u.rad
