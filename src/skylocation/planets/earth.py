import logging
import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from ...time import Time
from . import Planet

from . import errmsg
from ... import logger


class Earth(Planet):
    def __init__(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        self.__logger = logging.getLogger('src.skylocation.planets.earth.Earth')
        self.__logger.setLevel(logger.getEffectiveLevel())

        self.ephemeris = pk.planet.jpl_lp('earth')

        self.__logger.debug('Initializing Earth\'s superclass.')
        super(Earth, self).__init__(obstime, name='Earth', epoch=obstime.iso, is_earth=True)
        self.__logger.debug('Exiting `Earth` class initialization.')
