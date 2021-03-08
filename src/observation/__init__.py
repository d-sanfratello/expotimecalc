import numpy as np

from ..location import Location
from ..skylocation import SkyLocation
from ..time import Time
from .. import Versor

from .. import Tsidday
from .. import GMSTeq2000


class Observation:
    def __init__(self, location, obstime=None, target=None):
        if not isinstance(location, Location):
            raise TypeError("Must be of type `expotimecalc.location.Location`.")
        if not isinstance(obstime, Time):
            raise TypeError("Must be of type `expotimecalc.time.Time` or `astropy.time.Time`.")
        if not isinstance(target, SkyLocation):
            raise TypeError("Must be of type `expotimecalc.skylocation.SkyLocation`.")

        self.location = location
        self.obstime = obstime
        self.target = target

        self.target.observe_at_date(self.obstime)

        self.zenithJ2000 = Versor(ra=0., dec=self.location.lat.rad, unit='rad')\
            .rotate('z', self.sidereal_day(self.target.epoch) + GMSTeq2000.rad + self.location.lon.rad, unit='rad')

        self.zenith = self.zenith_at_date(self.obstime)

    def zenith_at_date(self, obstime):
        return self.zenithJ2000.rotate('z', self.sidereal_day(obstime), unit='rad')

    def sidereal_day(self, obstime, epoch_time=None):
        return (2*np.pi/Tsidday.value) * (obstime - self.target.epoch).jd
