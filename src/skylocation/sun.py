import numpy as np

from astropy import units as u

from src.time import Time
from src.skylocation import SkyLocation

from src import Tsidyear
from src import Equinox2000

from src import errmsg
from src import warnmsg


class Sun(SkyLocation):
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        super(Sun, self).__init__(locstring=None, ra=0*u.deg, dec=0*u.deg, obstime=obstime,
                                  ra_unit='deg', dec_unit='deg', epoch='J2000', name='Sun')

        self.vector_obstime = self.observe_at_date(obstime, copy=True)

    def observe_at_date(self, obstime, copy=True):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        vector_obstime = self.vector_epoch.rotate_inv('x', self.axial_tilt(self.obstime), copy=True)\
            .rotate('z', self.sidereal_year_rotation(self.obstime), copy=True)\
            .rotate('x', self.axial_tilt(self.obstime), copy=True)

        if copy:
            return vector_obstime
        else:
            self.vector_obstime = vector_obstime

    @classmethod
    def sidereal_year_rotation(cls, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        return ((2 * np.pi / Tsidyear.value) * (obstime - reference.time).jd) % (2*np.pi) * u.rad
