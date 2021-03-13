import numpy as np

from astropy import units as u

from src.time import Time
from src.location import Location
from src import Versor

from src import hms2deg
from src import dms2deg

from src import Tprec
from src import tJ2000


class SkyLocation(Location):
    def __init__(self, locstring=None, ra=None, dec=None, obstime=None, ra_unit='hour', dec_unit='deg', epoch='J2000'):
        if locstring is not None:
            ra, dec = locstring.split()

        if ra is not None and dec is not None:
            if isinstance(ra, str) and ra.lower().find("h") >= 0:
                ra = hms2deg(ra)
            elif ra_unit == 'hour':
                ra *= 15  # 1h = 15°
            elif ra_unit == 'deg':
                ra = ra
            elif ra_unit == 'rad':
                ra = np.rad2deg(ra)
            else:
                raise ValueError("Unknown unit of measure in Right Ascension.")

            if isinstance(dec, str) and dec.lower().find("d") >= 0:
                dec = dms2deg(dec)
            elif dec_unit == 'deg':
                dec = dec
            elif dec_unit == 'rad':
                dec = np.rad2deg(dec)
            else:
                raise ValueError("Unknown unit of measure in Declination.")
        else:
            raise ValueError("Must select at least one between `locstring` and the `ra`-`dec` couple.")

        super(SkyLocation, self).__init__(locstring, lat=dec, lon=ra)

        self.dec = self.__dict__.pop('lat')
        self.ra = self.__dict__.pop('lon')

        if obstime is None:
            self.obstime = Time(epoch).utc
        elif isinstance(obstime, Time):
            self.obstime = obstime.utc
        else:
            self.obstime = Time(obstime).utc
        self.epoch = Time(epoch).utc

        self.vector_epoch = Versor(self.ra, self.dec)
        self.vector_obstime = self.observe_at_date(self.obstime)

    def convert_to_epoch(self, epoch='J2000'):
        if epoch not in ['J2000']:
            raise ValueError("`epoch` is not a valid epoch string.")

        self.epoch = Time(epoch).utc

        # defined for J2000. Needs revision for other epochs
        self.vector_epoch = self.vector_epoch.rotate_inv('x', self.nutation_corr(self.obstime))\
            .rotate_inv('z', self.equinox_prec_corr(self.obstime))\
            .rotate('x', self.nutation_corr(self.obstime))

    def observe_at_date(self, obstime, copy=True):
        if isinstance(obstime, Time):
            self.obstime = obstime.utc
        else:
            self.obstime = Time(obstime).utc

        vector_obstime = self.vector_epoch.rotate_inv('x', self.nutation_corr(self.obstime), copy=True)\
            .rotate('z', self.equinox_prec_corr(self.obstime), copy=True)\
            .rotate('x', self.nutation_corr(self.obstime), copy=True)

        if copy:
            return vector_obstime
        else:
            self.vector_obstime = vector_obstime

    def __str__(self):
        ra, dec = self.__repr__().split()

        if dec.find('-') >= 0:
            return "RA:\t\t {0}\nDEC:\t{1}".format(ra, dec)
        else:
            return "RA:\t\t{0}\nDEC:\t{1}".format(ra, dec)

    def __repr__(self):
        ra = self.ra.hms
        dec = np.array(self.dec.dms)
        if dec[0] < 0:
            dec[1] *= -1
            dec[2] *= -1

        ra = "{0:d}h{1:d}m{2:.3f}s".format(int(ra[0]), int(ra[1]), ra[2])
        dec = "{0:d}d{1:d}m{2:.3f}s".format(int(dec[0]), int(dec[1]), dec[2])
        return ra + " " + dec

    @staticmethod
    def equinox_prec_corr(obstime):
        return (2*np.pi/Tprec.value) * (obstime - tJ2000).jd

    @staticmethod
    def nutation_corr(obstime):
        # check for Earth Fact Sheet at https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
        return 23.44 * u.deg
