import logging
import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from ...time import Time
from . import Planet

from . import errmsg
from ... import logger


class Venus(Planet):
    def __init__(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        self.__logger = logging.getLogger('src.skylocation.planets.venus.Venus')
        self.__logger.setLevel(logger.getEffectiveLevel())
        self.__logger.debug('Initializing Venus.')

        self.ephemeris = pk.planet.jpl_lp('venus')

        super(Venus, self).__init__(obstime, name='Venus', epoch=obstime.iso, is_earth=False)
