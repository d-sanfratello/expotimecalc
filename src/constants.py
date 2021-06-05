import numpy as np
import warnings

from astropy import units as u
from astropy.constants import Constant
from astropy.coordinates import EarthLocation

from skyfield import api
from skyfield import almanac

from .time import Time

from . import warnmsg
from . import errmsg


Tsidday = Constant(abbrev='T_sidday',
                   name='Sidereal day',
                   value=(23.93447192 * u.hour).to(u.day),
                   unit='day',
                   uncertainty=0,
                   reference='Expl. Suppl. p698 (Sidereal year in 1990)')

Tsidyear = Constant(abbrev='T_sidyear',
                    name='Sidereal year',
                    value=365.256363004 * u.day,
                    unit='day',
                    uncertainty=0,
                    reference='https://hpiers.obspm.fr/eop-pc/models/constants.html')

Jyear = Constant(abbrev='J_year',
                 name='Julian year',
                 value=365.25 * u.day,
                 unit='day',
                 uncertainty=0,
                 reference='')

Tprec = Constant(abbrev='T_prec',
                 name='Equinox precession period',
                 value=((1/(50287.9226200 * u.arcsec / (1000 * u.year))).to(u.year/u.deg) * 360 * u.deg).to(u.day),
                 unit='day',
                 uncertainty=0,
                 reference='DOI:10.1016/j.pss.2006.06.003 and DOI:10.1051/0004-6361:20021912')

Omegasidmoon = Constant(abbrev='Omega_sidMoon',
                        name='Rotational frequency of the Moon',
                        value=(2.661699489e-6 * u.rad / u.s).to(u.rad / u.d),
                        unit='rad / day',
                        uncertainty=0,
                        reference='Expl. Suppl. p701 (Revolution frequency of Moon)')

moon_incl_to_ecliptic = Constant(abbrev='moon_incl_to_ecliptic',
                                 name='Inclination of the orbit of the Moon with respect to the Ecliptic plane',
                                 value=5.145396 * u.deg,
                                 unit='deg',
                                 uncertainty=0,
                                 reference='Expl. Suppl. p701')

tJ2000 = Time('J2000.0')

sidday_diff = Constant(abbrev='sidday_diff',
                       name='Difference between one julian day and one sidereal day',
                       value=1 * u.day - Tsidday,
                       unit='day',
                       uncertainty=0,
                       reference='')


class GMSTeq2000:
    def __init__(self, obstime):
        """
        Classe che identifica il Greenwich Mean Sidereal Time all'equinozio vernale del 2000.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.__obstime = obstime
        self.hms = self.time
        self.deg = self.hms.to(u.deg)
        self.rad = self.hms.to(u.rad)

    @property
    def time(self):
        self.__obstime.location = EarthLocation.of_site('greenwich')
        return self.__obstime.sidereal_time('mean')


class Equinox2000:
    """
    Costante che definisce l'equinozio vernale del 2000, calcolandolo dalle effemeridi de421.
    """
    def __init__(self):
        ts = api.load.timescale()
        eph = api.load('de421.bsp')
        t0 = ts.utc(2000, 3, 20)
        t1 = ts.utc(2000, 3, 21)
        t = almanac.find_discrete(t0, t1, almanac.seasons(eph))[0]
        eph.close()

        self.__time = Time(t.utc_iso()[0][:-1], scale='utc')
        self.__gmst = GMSTeq2000(self.__time)

    @property
    def time(self):
        return self.__time

    @property
    def GMST(self):
        warnings.warn("Capital property is deprecated and will be removed. Please use `gmst`.", DeprecationWarning)
        return self.__gmst

    @property
    def gmst(self):
        return self.__gmst

    @property
    def hour(self):
        return self.__time.jd % 1

    @property
    def rad(self):
        return 2 * np.pi * self.hour * u.rad


class Eclipse1999:
    """
    Costante che definisce l'eclissi di sole dell'agosto 1999, calcolandolo dalle effemeridi de421.
    """
    def __init__(self):
        ts = api.load.timescale()
        eph = api.load('de421.bsp')
        t0 = ts.utc(1999, 8, 1)
        t1 = ts.utc(1999, 8, 15)
        t = almanac.find_discrete(t0, t1, almanac.moon_nodes(eph))[0]
        eph.close()

        self.__time = Time(t.utc_iso()[0][:-1], scale='utc')
        self.__gmst = GMSTeq2000(self.__time)

    @property
    def time(self):
        return self.__time

    @property
    def GMST(self):
        warnings.warn("Capital property is deprecated and will be removed. Please use `gmst`.", DeprecationWarning)
        return self.__gmst

    @property
    def gmst(self):
        return self.__gmst

    @property
    def hour(self):
        return self.__time.jd % 1

    @property
    def rad(self):
        return 2 * np.pi * self.hour * u.rad


Equinox2000 = Equinox2000()
Eclipse1999 = Eclipse1999()
