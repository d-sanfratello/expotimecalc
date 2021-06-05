from astropy import units as u
from astropy.coordinates.angles import Latitude
from astropy.coordinates.angles import Longitude
from astropy.units.quantity import Quantity
import logging
import numpy as np

from ..time import Time
from .. import Versor

from .. import dms2deg
from ..constants import Equinox2000
from ..constants import Tsidday

from .. import errmsg
from .. import logger


# noinspection PyTypeChecker
class Location:
    valid_coord_types = (int, float, str, Latitude, Longitude, Quantity)
    equinoxes = {'equinoxJ2000': Equinox2000}

    @classmethod
    def parse_string(cls, coord_string, coord_letter_pos, coord_letter_neg):
        """
        Metodo di classe per effettuare il parsing di una stringa, che serve a determinare se si tratta di latitudine N
        o S o di longitudine E o W, e assegnare il segno opportuno. Viene indicata la stringa e quali lettere sono da
        considerarsi per le coordinate positive e per quelle negative.
        """
        if (_ := coord_string.lower().find(coord_letter_pos.lower())) >= 0:
            # Se trova la lettera per le coordinate positive
            return dms2deg(coord_string[:_])
        elif (_ := coord_string.lower().find(coord_letter_neg.lower())) >= 0:
            # Se trova la lettera per le coordinate negative
            return -dms2deg(coord_string[:_])
        else:
            # Se non trova lettere, la considera una stringa di un valore numerico.
            return float(coord_string)

    def __init__(self, locstring=None, lat=None, lon=None, timezone=None, obstime=None, in_sky=False):
        """
        Classe che definisce un vettore posizione data una stringa di coordinate o, direttamente, le coordinate
        assegnate agli argomenti `lat`, `lon`. Viene indicato anche il fuso orario e se l'oggetto è in cielo. L'ultimo
        parametro è utile in quanto questa classe ha caratteristiche ereditate dalla classe
        `src.skylocation.SkyLocation`.
        """
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

        self.__logger = logging.getLogger('src.location.Location')
        self.__logger.setLevel(logger.getEffectiveLevel())
        self.__logger.debug('Initializing `Location` class.')

        self.__in_sky = in_sky
        self.__logger.debug(f'`Location` is in sky: {self.__in_sky}')

        # Controlla le coordinate o le interpreta dalla stringa, assegnandole a degli oggetti
        # `astropy.coordinates.angles.Latitude` e `astropy.coordinates.angles.Longitude`.
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
        self.__logger.debug(f'lat and lon set to {self.lat}, {self.lon}')

        # Se non è assegnata alcuna data di osservazione, viene indicata la data dell'equinozio vernale del 2000.
        if obstime is None:
            self.obstime = Equinox2000.time
        else:
            self.obstime = obstime
        self.__logger.debug(f'obstime: {obstime}')

        if not self.__in_sky:
            # Se l'oggetto ha coordinate terrestri, viene salvato il fuso orario e vengono assegnate le posizioni, in
            # coordinate celesti, dello zenith all'equinozio 2000 e alla data. Inoltre, per comodità, anche se poi non
            # sono utilizzate altrove, vengono anche fornite le coordinate degli assi che identificano il nord e l'est
            # locali.
            self.__logger.debug('Location is on Earth. Defining location-specific quantities.')
            if timezone is None:
                self.timezone = 0 * u.hour
            else:
                self.timezone = timezone * u.hour

            self.zenithJ2000 = Versor(ra=self.lst(Equinox2000.time).rad, dec=self.lat.rad, unit='rad')

            self.zenith_obstime = None
            self.north = Versor(ra=180 * u.deg + self.lst(Equinox2000.time), dec=90 * u.deg - self.lat)
            self.east = Versor(ra=90 * u.deg + self.lst(Equinox2000.time), dec=0 * u.deg)

            self.zenith_at_date(self.obstime, copy=False)
        else:
            self.__logger.debug('Location is in the sky.')

    def zenith_at_date(self, obstime, axis=None, copy=True):
        """
        Metodo che calcola la posizione, sulla sfera celeste, del punto indicato dallo zenith alla data. Nel caso in cui
        l'oggetto che chiama questa funzione fosse già un oggetto celeste, questo metodo solleva un'eccezione.
        """
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
                # Se si chiede di restituire lo zenith.
                return self.zenithJ2000.rotate('z', self.sidereal_day_rotation(obstime), copy=True)
            elif axis.lower() in ['n', 'north']:
                # Se si chiede di restituire il nord.
                return self.north.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True)
            elif axis.lower() in ['e', 'east']:
                # Se si chiede di restituire l'est.
                return self.east.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True)
            else:
                # Se nessuna scelta è indicata, viene restituita una tupla contenente i tre vettori.
                return (self.zenithJ2000.rotate('z', self.sidereal_day_rotation(obstime), copy=True),
                        self.north.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True),
                        self.east.rotate('z', self.sidereal_day_rotation(self.obstime), copy=True))
        else:
            # Se si indica di NON volere una copia dei vettori, invece, il vettore ruotato viene assegnato
            # all'attributo corrispondente.
            self.zenith_obstime = self.zenithJ2000.rotate('z', self.sidereal_day_rotation(obstime), copy=True)
            self.north.rotate('z', self.sidereal_day_rotation(self.obstime), copy=False)
            self.east.rotate('z', self.sidereal_day_rotation(self.obstime), copy=False)

    def sidereal_day_rotation(self, obstime, epoch_eq='equinoxJ2000'):
        """
        Metodo che calcola la rotazione dello zenith locale dovuta alla rotazione della Terra intorno al proprio asse,
        secondo il giorno siderale.
        """
        if self.__in_sky:
            # Se è un'istanza di un oggetto celeste, viene sollevata un'eccezione.
            raise TypeError(errmsg.cannotAccessError.format(self.sidereal_day_rotation.__name__))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = self.equinoxes[epoch_eq]

        # Fase della rotazione di periodo dato dal giorno siderale, in funzione dell'equinozio di riferimento.
        return ((2 * np.pi * u.rad / Tsidday.value) * (obstime - reference.time).jd) % (2 * np.pi * u.rad)

    def lst(self, obstime, epoch_eq='equinoxJ2000'):
        """
        Metodo che calcola il tempo siderale locale, alla data. L'equinozio viene indicato perché serve alla chiamata
        del metodo `Location.sidereal_day_rotation`, anche se al momento il riferimento è inutilizzato. Inoltre serve
        per fornire il valore dell'LST a Greenwich, contenuta nell'equinozio.
        """
        if self.__in_sky:
            raise TypeError(errmsg.cannotAccessError.format(self.lst.__name__))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = self.equinoxes[epoch_eq]

        # Sfruttando il fatto che la posizione sulla sfera celeste dello zenith alla data indica il LST, si ricava il
        # GMST come:
        #
        #       deltaLST = GMST(at equinox date) + LST(at date) + longitude.
        #
        shift = reference.gmst.deg + self.sidereal_day_rotation(obstime).to(u.deg) + self.lon
        return shift.to(u.hourangle) % (24 * u.hourangle)

    def __str__(self):
        """
        Metodo 'magico' di python per gestire la chiamata `print(Location)`. Restituisce una stringa con le coordinate
        geografiche.

        """
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
        """
        Metodo 'magico' di python per gestire la chiamata `Location` nel terminale. Restituisce le coordinate
        geografiche.
        """
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
