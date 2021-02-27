from ..location import Location
from ..skylocation import SkyLocation
from ..time import Time
from .. import Versor

class Observation:
    def __init__(self, location, obstime, target):
        self.location = location
        self.obstime = obstime
        self.target = target
