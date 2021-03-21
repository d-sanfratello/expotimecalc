import numpy as np

from astropy import units as u
from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude
from astropy.units.quantity import Quantity

from src.time import Time
from src.location import Location
from src import Versor

from src import hms2deg
from src import dms2deg

from src import Tprec
from src import Equinox2000

from src import errmsg
from src import warnmsg


class SkyLocation(Location):
    epoch_names = {'J2000': 'J'}
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, locstring=None, ra=None, dec=None, obstime=None, ra_unit='hour', dec_unit='deg', epoch='J2000',
                 name=None):
        if locstring is not None:
            ra, dec = locstring.split()

        if ra is not None and dec is not None:
            if isinstance(ra, str) and ra.lower().find("h") >= 0:
                ra = hms2deg(ra)
            elif isinstance(ra, Quantity):
                ra = Longitude(ra)
            elif ra_unit == 'hour':
                # 1h = 15°
                ra *= 15
            elif ra_unit == 'deg':
                ra = ra
            elif ra_unit == 'rad':
                ra = np.rad2deg(ra)
            else:
                raise ValueError(errmsg.invalidUnitError)

            if isinstance(dec, str) and dec.lower().find("d") >= 0:
                dec = dms2deg(dec)
            elif isinstance(dec, Quantity):
                dec = Latitude(dec)
            elif dec_unit == 'deg':
                dec = dec
            elif dec_unit == 'rad':
                dec = np.rad2deg(dec)
            else:
                raise ValueError(errmsg.invalidUnitError)
        else:
            raise ValueError(errmsg.mustDeclareRaDecError)

        if obstime is not None and not isinstance(obstime, Time):
            raise TypeError(
                errmsg.notThreeTypesError.format('obstime', 'Nonetype', 'src.time.Time', 'astropy.time.Time'))

        if name is not None and not isinstance(name, str):
            raise TypeError(errmsg.notTwoTypesError.format('name', 'Nonetype', 'string'))

        super(SkyLocation, self).__init__(locstring, lat=dec, lon=ra)

        self.dec = self.__dict__.pop('lat')
        self.ra = self.__dict__.pop('lon')

        self.name = self.name_object(name, epoch)

        if obstime is None:
            self.obstime = Equinox2000.time
        else:
            self.obstime = obstime
        self.epoch = Time(epoch)

        self.vector_epoch = Versor(self.ra, self.dec)
        self.vector_obstime = None
        self.at_date(self.obstime)

    def convert_to_epoch(self, epoch='J2000'):
        if epoch not in ['J2000']:
            raise ValueError(errmsg.invalidEpoch)

        self.epoch = Time(epoch).utc

        # defined for J2000. Needs revision for other epochs
        self.vector_epoch = self.vector_epoch.rotate_inv('x', self.axial_tilt(self.obstime), copy=True)\
            .rotate_inv('z', self.equinox_prec(self.obstime), copy=True)\
            .rotate('x', self.axial_tilt(self.obstime), copy=True)

        self.ra = self.vector_epoch.ra
        self.dec = self.vector_epoch.dec

    def precession_at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        vector_obstime = self.vector_epoch.rotate_inv('x', self.axial_tilt(self.obstime), copy=True)\
            .rotate('z', self.equinox_prec(self.obstime), copy=True)\
            .rotate('x', self.axial_tilt(self.obstime), copy=True)

        return vector_obstime

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.precession_at_date(obstime)

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.precession_at_date(obstime, copy=False)

    def name_object(self, name, epoch):
        if name is None:
            coords = self.__repr__()
            if self.dec.deg >= 0:
                coords.replace(' ', '+')
            else:
                coords.replace(' ', '')
            return self.epoch_names[epoch] + coords
        else:
            return name

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

    @classmethod
    def equinox_prec(cls, obstime, epoch_eq='equinoxJ2000'):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        return ((2*np.pi/Tprec.value) * (obstime - reference.time).jd) % (2*np.pi) * u.rad

    @staticmethod
    def axial_tilt(obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # check for Kenneth Seidelmann, "Explanatory Supplement to the astronomical almanac" p. 114.
        # T = (obstime.jd - tJ2000.jd) / 36525
        # return 23*u.deg+26*u.arcmin+21.448*u.arcsec-46.8150*u.arcsec*T-0.00059*u.arcsec*T**2+0.001813*u.arcsec*T**3

        # check for Earth Fact Sheet at https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html [23.44]
        # check for Kenneth Seidelmann, "Explanatory Supplement to the astronomical almanac" p. 315.
        return 23.43929111 * u.deg


import src.skylocation.sun
