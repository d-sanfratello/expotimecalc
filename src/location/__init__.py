from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude
from astropy.units.quantity import Quantity

from src import dms2deg


class Location:
    valid_coord_types = (int, float, str, Latitude, Longitude, Quantity)

    @classmethod
    def parse_string(cls, coord_string, coord_letter_pos, coord_letter_neg):
        if _ := coord_string.lower().find(coord_letter_pos.lower()) >= 0:
            return dms2deg(coord_string[:_])
        elif _ := coord_string.lower().find(coord_letter_neg.lower()) >= 0:
            return -dms2deg(coord_string[:_])
        else:
            return float(coord_string)

    def __init__(self, locstring=None, lat=None, lon=None):
        if locstring is None and (lat is None and lon is None):
            raise ValueError("Must declare location of observatory.")
        elif locstring is None and (lat is None or lon is None):
            raise ValueError("Must declare both latitude and longitude.")

        if locstring is not None:
            if not isinstance(locstring, str):
                raise TypeError("locstring must be of string type.")
        else:
            if not isinstance(lat, self.valid_coord_types) or not isinstance(lon, self.valid_coord_types):
                raise TypeError("`lat` or `lon` attributes are of wrong type.")

        if lat is None and lon is None:
            lat, lon = locstring.split()

        if isinstance(lat, str):
            lat = Latitude(type(self).parse_string(lat, 'N', 'S'), unit='deg')
        elif isinstance(lat, (int, float)):
            lat = Latitude(lat, unit='deg')
        elif isinstance(lat, Latitude):
            lat = lat
        elif isinstance(lat, Quantity):
            lat = Latitude(lat)
        else:
            raise TypeError("`lat` cannot be of type `Longitude`.")

        if isinstance(lon, str):
            lon = Longitude(type(self).parse_string(lon, 'E', 'W'), unit='deg')
        elif isinstance(lon, (int, float)):
            lon = Longitude(lon, unit='deg')
        elif isinstance(lon, Longitude):
            lon = lon
        elif isinstance(lon, Quantity):
            lon = Longitude(lon)
        else:
            raise TypeError("`lon` cannot be of type `Latitude`.")

        self.lat = lat
        self.lon = lon
