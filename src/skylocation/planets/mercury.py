import logging
import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from ...time import Time
from . import Planet

from . import errmsg
from ... import logger


class Mercury(Planet):
    def __init__(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        self.__logger = logging.getLogger('src.skylocation.planets.mercury.Mercury')
        self.__logger.setLevel(logger.getEffectiveLevel())
        self.__logger.debug('Initializing Mercury.')

        self.ephemeris = pk.planet.jpl_lp('mercury')

        super(Mercury, self).__init__(obstime, name='Mercury', epoch=obstime.iso, is_earth=False)
