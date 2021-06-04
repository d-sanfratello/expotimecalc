from astropy import units as u
from astropy.constants import Constant

from .time import Time


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
