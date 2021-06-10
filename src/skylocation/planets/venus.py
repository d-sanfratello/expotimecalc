import logging
import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from astropy import units as u

from ...time import Time
from . import Planet
from ...utils import century

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

        self.__standard_pars = {'a': 0.72333566 * u.AU, 'adot': 0.00000390 * u.AU/century,
                                'e': 0.00677672, 'edot': -0.00004107 / century,
                                'I': 3.39467605 * u.deg, 'Idot': -0.00078890 * u.deg / century,
                                'mean_long': 181.97909950 * u.deg, 'mean_long_dot': 58517.81538729 * u.deg / century,
                                'lon_peri': 131.60246718 * u.deg, 'lon_peri_dot': 0.00268329 * u.deg / century,
                                'lon_an': 76.67984255 * u.deg, 'lon_an_dot': -0.27769418 * u.deg / century}

        super(Venus, self).__init__(obstime, name='Venus', epoch=obstime.iso, is_earth=False)
