from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude


class Location:
    valid_coord_types = (int, float, str, Latitude, Longitude)

    @classmethod
    def parse_string(cls, coord_string, coord_letter_pos, coord_letter_neg):
        if coord_string.upper().find(coord_letter_pos.upper()) >= 0:
            return float(coord_string[:-1])
        elif coord_string.upper().find(coord_letter_neg.upper()) >= 0:
            return -float(coord_string[:-1])
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
            lat = type(self).parse_string(lat, 'N', 'S')
        elif isinstance(lat, (int, float)):
            lat = Latitude(lat, unit='deg')
        else:
            lat = lat

        if isinstance(lon, str):
            lon = type(self).parse_string(lon, 'E', 'W')
        elif isinstance(lon, (int, float)):
            lon = Longitude(lon, unit='deg')
        else:
            lon = lon

        self.lat = lat
        self.lon = lon
