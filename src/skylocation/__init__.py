# import numpy as np

from astropy.time import Time

from .. import location
from .. import hms2dms
from .. import Versor


class SkyLocation(location.Location):
    def __init__(self, locstring=None, ra=None, dec=None, obstime=None, epoch='J2000'):
        if ra is not None:
            ra = hms2dms(ra)
        elif locstring is not None:
            ra = hms2dms(locstring.split()[0])

        super(SkyLocation, self).__init__(locstring, lat=dec, lon=ra)

        self.dec = self.__dict__.pop('lat')
        self.ra = self.__dict__.pop('lon')

        if obstime is None:
            self.obstime = Time(epoch).utc
        elif isinstance(obstime, Time):
            self.obstime = obstime.utc
        else:
            self.obstime = Time(obstime).utc
        self.epoch = epoch

        self.vector_epoch = Versor(self.ra, self.dec)
        self.vector_obstime = None
        self.observe_at_date(self.obstime)

    def convert_to_epoch(self, epoch='J2000'):
        if epoch not in ['J2000']:
            raise ValueError("`epoch` is not a valid epoch string.")

        self.epoch = epoch

        # defined for J2000. Needs revision for other epochs
        self.vector_epoch = self.vector_epoch.rotate_inv('x', self.nutation_corr(self.obstime))\
            .rotate_inv('z', self.equinox_prec_corr(self.obstime))\
            .rotate('x', self.nutation_corr(self.obstime))

    def observe_at_date(self, obstime):
        if isinstance(obstime, Time):
            self.obstime = obstime.utc
        else:
            self.obstime = Time(obstime).utc

        self.vector_obstime = self.vector_epoch.rotate_inv('x', self.nutation_corr(self.obstime), copy=True)\
            .rotate('z', self.equinox_prec_corr(self.obstime), copy=True)\
            .rotate_inv('x', self.nutation_corr(self.obstime), copy=True)
