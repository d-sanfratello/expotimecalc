from astropy.time import Time

from .. import location
from .. import hms2dms


class SkyLocation(location.Location):
    def __init__(self, locstring=None, ra=None, dec=None, obstime=None):
        if ra is not None:
            ra = hms2dms(ra)
        elif locstring is not None:
            ra = hms2dms(locstring.split()[0])

        super(SkyLocation, self).__init__(locstring, lat=dec, lon=ra)

        self.__dict__['dec'] = self.__dict__.pop('lat')
        self.__dict__['ra'] = self.__dict__.pop('lon')

        if obstime is None:
            self.obsepoch = Time.now().utc
        elif isinstance(obstime, Time):
            self.obsepoch = obstime.utc
        else:
            self.obsepoch = Time(obstime).utc

    def convert_to_epoch(self, epoch='J2000'):
        if epoch not in ['J2000']:
            raise ValueError("`epoch` is not a valid epoch string.")


