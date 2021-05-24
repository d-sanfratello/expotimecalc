import pykep as pk
# https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from astropy import units as u
from astropy import constants as cts
from astropy.coordinates import Angle
from astropy.coordinates import Longitude

from ... import Equinox2000
from ...time import Time
from . import Planet

from . import errmsg


class Earth(Planet):
    def __init__(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        planet = pk.planet.jpl_lp('earth')
        epoch = pk.epoch_from_string(Equinox2000.time.iso)

        __semimaj, __ecc, __inclination, __ra_an, __peri_arg, __mean_anomaly = planet.osculating_elements(epoch)
        __semimaj *= u.m
        __semimaj = __semimaj.to(cts.au)

        __inclination *= u.rad
        __inclination = Angle(__inclination.to(u.deg))

        __ra_an *= u.rad
        __longitude_an = Longitude(__ra_an.to(u.deg))

        __peri_arg *= u.rad
        __peri_arg = Angle(__peri_arg.to(u.deg))

        super(Earth, self).__init__(obstime, semimaj=__semimaj, eccentricity=__ecc, longitude_an=__longitude_an,
                                    inclination=__inclination, argument_perihelion=__peri_arg,
                                    mean_anomaly=__mean_anomaly, name='Earth', epoch='J2000', is_earth=True)

