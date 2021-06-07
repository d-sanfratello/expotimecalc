import logging
import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from astropy import constants as cst

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
        self.__logger.debug('Initializing Earth.')

        self.ephemeris = pk.planet.jpl_lp('earth')

        super(Earth, self).__init__(obstime, name='Earth', epoch=obstime.iso, is_earth=True)

    @property
    def mass(self):
        return cst.M_earth
