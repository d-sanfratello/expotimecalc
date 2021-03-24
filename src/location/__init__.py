from astropy import units as u
from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude
from astropy.units.quantity import Quantity
import numpy as np

from src.time import Time
from src import Versor

from src import dms2deg
from src import Equinox2000
from src import Tsidday

from src import errmsg
from src import warnmsg


class Location:
    valid_coord_types = (int, float, str, Latitude, Longitude, Quantity)
    equinoxes = {'equinoxJ2000': Equinox2000}

    @classmethod
    def parse_string(cls, coord_string, coord_letter_pos, coord_letter_neg):
        if (_ := coord_string.lower().find(coord_letter_pos.lower())) >= 0:
            return dms2deg(coord_string[:_])
        elif (_ := coord_string.lower().find(coord_letter_neg.lower())) >= 0:
            return -dms2deg(coord_string[:_])
        else:
            return float(coord_string)

    def __init__(self, locstring=None, lat=None, lon=None, timezone=None, obstime=None, in_sky=False):
        if locstring is None and (lat is None and lon is None):
            raise ValueError(errmsg.mustDeclareLocation)
        elif locstring is None and (lat is None or lon is None):
            raise ValueError(errmsg.mustDeclareLatLonError)

        if locstring is not None:
            if not isinstance(locstring, str):
                raise TypeError(errmsg.notTypeError.format('locstring', 'string'))
        else:
            if not isinstance(lat, self.valid_coord_types) or not isinstance(lon, self.valid_coord_types):
                raise TypeError(errmsg.latLonWrongTypeError)

        if timezone is not None and not isinstance(timezone, int):
            raise TypeError(errmsg.notTwoTypesError('timezone', 'Nonetype', 'int'))

        if obstime is not None and not isinstance(obstime, Time):
            raise TypeError(
                errmsg.notThreeTypesError.format('obstime', 'Nonetype', 'src.time.Time', 'astropy.time.Time'))

        if not isinstance(in_sky, bool):
            raise TypeError(errmsg.notTypeError('is_sky', 'bool'))

        self.__in_sky = in_sky

        if lat is None and lon is None:
            lat, lon = locstring.split()

        if isinstance(lat, str):
            lat = Latitude(self.parse_string(lat, 'N', 'S'), unit='deg')
        elif isinstance(lat, (int, float)):
            lat = Latitude(lat, unit='deg')
        elif isinstance(lat, Latitude):
            lat = lat
        elif isinstance(lat, Quantity):
            lat = Latitude(lat)
        else:
            raise TypeError(errmsg.latNotLonError)

        if isinstance(lon, str):
            lon = Longitude(self.parse_string(lon, 'E', 'W'), unit='deg')
        elif isinstance(lon, (int, float)):
            lon = Longitude(lon, unit='deg')
        elif isinstance(lon, Longitude):
            lon = lon
        elif isinstance(lon, Quantity):
            lon = Longitude(lon)
        else:
            raise TypeError(errmsg.lonNotLatError)

        self.lat = lat
        self.lon = lon

        if obstime is None:
            self.obstime = Equinox2000.time
        else:
            self.obstime = obstime

        if not self.__in_sky:
            if timezone is None:
                self.timezone = 0 * u.hour
            else:
                self.timezone = timezone * u.hour

            self.zenithJ2000 = Versor(ra=self.lst(Equinox2000.time).rad,
                                      dec=self.lat.rad,
                                      unit='rad')

            self.zenith_obstime = None
            self.north = Versor(ra=180 * u.deg + self.lst(Equinox2000.time),
                                dec=90 * u.deg - self.lat)
            self.east = Versor(ra=90 * u.deg + self.lst(Equinox2000.time),
                               dec=0 * u.deg)

            self.zenith_at_date(self.obstime, copy=False)

    def zenith_at_date(self, obstime, axis=None, copy=True):
        if self.__in_sky:
            raise TypeError(errmsg.cannotAccessError.format(self.zenith_at_date.__name__))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        if axis is not None and not isinstance(axis, str):
            raise TypeError(errmsg.notTwoTypesError.format('axis', 'Nonetype', 'string'))
        elif axis is not None and axis.lower() not in ['z', 'n', 'e', 'zenith', 'north', 'east']:
            raise ValueError(errmsg.invalidDirectionError)

        self.obstime = obstime

        if copy:
            if axis.lower() in ['z', 'zenith']:
                return self.zenithJ2000.rotate('z', self.sidereal_day_rotation(obstime), copy=True)
            elif axis.lower() in ['n', 'north']:
                return self.north.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True)
            elif axis.lower() in ['e', 'east']:
                return self.east.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True)
            else:
                return (self.zenithJ2000.rotate('z', self.sidereal_day_rotation(obstime), copy=True),
                        self.north.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True),
                        self.east.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True))
        else:
            self.zenith_obstime = self.zenithJ2000.rotate('z', self.sidereal_day_rotation(obstime), copy=True)
            self.north = self.north.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True)
            self.east = self.east.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True)

    def sidereal_day_rotation(self, obstime, epoch_eq='equinoxJ2000'):
        if self.__in_sky:
            raise TypeError(errmsg.cannotAccessError.format(self.sidereal_day_rotation.__name__))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = self.equinoxes[epoch_eq]

        return ((2 * np.pi * u.rad / Tsidday.value) * (obstime - reference.time).jd) % (2 * np.pi * u.rad)

    def lst(self, obstime, epoch_eq='equinoxJ2000'):
        if self.__in_sky:
            raise TypeError(errmsg.cannotAccessError.format(self.lst.__name__))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = self.equinoxes[epoch_eq]

        shift = reference.GMST.deg + self.sidereal_day_rotation(obstime).to(u.deg) + self.lon
        return shift.to(u.hourangle) % (24 * u.hourangle)

    def __str__(self):
        lon, lat = self.__repr__().split()

        if lat.find('-') >= 0:
            lat_string = "{}S".format(lat[1:])
        else:
            lat_string = "{}N".format(lat[1:])

        if lon.find('-') >= 0:
            lon_string = "{}W".format(lon[1:])
        else:
            lon_string = "{}E".format(lon[1:])

        return lat_string + " " + lon_string

    def __repr__(self):
        lon = self.lon

        if lon > 180 * u.deg:
            lon -= 360 * u.deg

        lon = np.array(lon.dms)
        if lon[0] < 0:
            lon[1] *= -1
            lon[2] *= -1

        lat = np.array(self.lat.dms)
        if lat[0] < 0:
            lat[1] *= -1
            lat[2] *= -1

        lon = "{0:d}d{1:d}m{2:.3f}s".format(int(lon[0]), int(lon[1]), lon[2])
        lat = "{0:d}d{1:d}m{2:.3f}s".format(int(lat[0]), int(lat[1]), lat[2])
        return lon + " " + lat
