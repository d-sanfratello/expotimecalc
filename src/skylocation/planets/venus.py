import numpy as np

from astropy import units as u
from astropy import constants as cts
from astropy.coordinates import Angle

from skyfield import api

from ... import Equinox2000
from . import Planet


class Venus(Planet):
    def __init__(self, obstime):

        # From: https://nssdc.gsfc.nasa.gov/planetary/factsheet/venusfact.html
        __semimaj = 0.72333199 * cts.au
        __ecc = 0.00677323
        __semimin = __semimaj * np.sqrt(1 - __ecc ** 2)
        __distance = 0.5 * (__semimin + __semimaj)

        __longitude_an = Angle(76.68069 * u.deg)
        __orbit_inclination = Angle(3.39471 * u.deg)
        __revolution_period = 227.701 * u.d

        ts = api.load.timescale()
        time = ts.from_astropy(Equinox2000.time)
        planets = api.load('de421.bsp')
        earth = planets['earth']
        venus = planets['venus']
        astrometric = earth.at(time).observe(venus)
        ra, dec, dist = astrometric.radec()
        planets.close()

        super(Venus, self).__init__(ra.to(u.deg), dec.to(u.deg), __distance, obstime,
                                    __longitude_an, __orbit_inclination, __revolution_period,
                                    name='Venus', epoch='J2000')
