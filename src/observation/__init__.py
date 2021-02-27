from ..location import Location
from ..skylocation import SkyLocation
from ..time import Time
from .. import Versor

class Observation:
    def __init__(self, location, obstime, target):
        if not isinstance(location, Location):
            raise TypeError("Must be of type `expotimecalc.location.Location`.")
        if not isinstance(obstime, Time):
            raise TypeError("Must be of type `expotimecalc.time.Time` or `astropy.time.Time`.")
        if not isinstance(target, SkyLocation):
            raise TypeError("Must be of type `expotimecalc.skylocation.SkyLocation`.")

        self.location = location
        self.obstime = obstime
        self.target = target
