import logging
import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from astropy import constants as cst
from astropy import units as u

from ...time import Time
from . import Planet
from ...utils import century

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

        __standard_pars = {'a': 1.00000261 * u.AU, 'adot': 0.00000562 * u.AU / century,
                           'e': 0.01671123, 'edot': -0.00004392 / century,
                           'I': -0.00001531 * u.deg, 'Idot': -0.01294668 * u.deg / century,
                           'mean_long': 100.46457166 * u.deg, 'mean_long_dot': 35999.37244981 * u.deg / century,
                           'long_peri': 102.93768193 * u.deg, 'long_peri_dot': 0.32327364 * u.deg / century,
                           'long_an': 0.0 * u.deg, 'long_an_dot': 0.0 * u.deg / century}

        super(Earth, self).__init__(obstime, name='Earth', epoch=obstime.iso, is_earth=True,
                                    standard_pars=__standard_pars)

    @property
    def mass(self):
        return cst.M_earth
