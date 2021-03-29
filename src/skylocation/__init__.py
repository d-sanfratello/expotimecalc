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
from src import tJ2000

from src import errmsg
from src import warnmsg


class SkyLocation(Location):
    # For plot strings
    epoch_names = {'J2000': 'J'}
    equinoxes = {'equinoxJ2000': Equinox2000}
    epoch_rel_eq = {'J2000': 'equinoxJ2000'}

    def __init__(self, locstring=None, ra=None, dec=None, obstime=None, ra_unit='hour', dec_unit='deg', epoch='J2000',
                 name=None):
        """
        Classe che definisce un oggetto celeste e permette di calcolarne la posizione ad una data scelta dall'utente.
        Accetta le coordinate equatoriali in una stringa, oppure di averle espresse esplicitamente in `ra` e `dec`.
        Eredita alcune caratteristiche dalla classe `src.location.Location` anche se sono, di fatto, due classi
        differenti.
        """
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

        # Inizializzo la classe "genitore" `Location` per interpretare correttamente la stringa o le coordinate fornite
        # dall'utente.
        super(SkyLocation, self).__init__(locstring, lat=dec, lon=ra, in_sky=True)

        # Per evitare confusione elimino gli attributi `lat` e `lon` ereditati da `Location` e li definisco come `ra` e
        # `dec`, definiti all'epoca in cui ho fornito le coordinate.
        self.dec_epoch = self.__dict__.pop('lat')
        self.ra_epoch = self.__dict__.pop('lon')

        if obstime is None:
            # Se nessuna data di osservazione è fornita, viene usata la data dell'equinozio vernale del 2000.
            self.obstime = Equinox2000.time
        else:
            self.obstime = obstime
        self.epoch = Time(epoch)
        self.epoch_eq = self.equinoxes[self.epoch_rel_eq[epoch]]

        # Inizializzo il versore delle coordinate con le coordinate all'epoca indicata.
        self.vector_epoch = Versor(self.ra_epoch, self.dec_epoch)
        self.vector_obstime = None

        # Se è fornita una data, inizializzo le coordinate a quella data.
        if obstime is not None:
            self.ra = ra
            self.dec = dec

            self.convert_to_epoch(obstime)
        else:
            self.ra = None
            self.dec = None

            # Compio le correzioni per la data di osservazione.
            self.at_date(self.obstime)

        self.name = self.name_object(name, epoch)

    def convert_to_epoch(self, obstime, epoch='J2000'):
        """
        Metodo che converte il vettore che indica la posizione all'epoca iniziale ad un'epoca diversa. Per adesso non è
        possibile inserire epoche diverse da quella J2000.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch not in ['J2000']:
            raise ValueError(errmsg.invalidEpoch)

        old_eq = self.epoch_eq
        epoch_eq = self.equinoxes[self.epoch_rel_eq[epoch]]

        # Il vettore che contiene le coordinate all'epoca viene convertito in coordinate eclittiche e viene applicata la
        # precessione degli equinozi, riportando poi la posizione in coordinate equatoriali. Dato che non è considerata
        # la nutazione, la rotazione intorno all'asse x è, di fatto, un semplice cambio di base.
        self.vector_epoch = self.vector_epoch.rotate_inv('x', self.axial_tilt(old_eq.time), copy=True)\
            .rotate_inv('z', self.equinox_prec(obstime, self.epoch_rel_eq[epoch]), copy=True)\
            .rotate('x', self.axial_tilt(epoch_eq.time), copy=True)

        self.epoch = Time(epoch).utc
        self.epoch_eq = epoch_eq

        # Vengono aggiornate le coordinate che indicano la posizione all'epoca.
        self.ra_epoch = self.vector_epoch.ra
        self.dec_epoch = self.vector_epoch.dec

    def observe_at_date(self, obstime):
        """
        Metodo che applica l'effetto della precessione degli equinozi ad una certa data. Al momento non è possibile
        inserire epoche diverse.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # La posizione all'epoca viene portata in coordinate eclittiche, viene applicato l'effetto della precessione e
        # viene riportata in coordinate equatoriali.
        vector_obstime = self.vector_epoch.rotate_inv('x', self.axial_tilt(obstime), copy=True)\
            .rotate('z', self.equinox_prec(obstime), copy=True)\
            .rotate('x', self.axial_tilt(obstime), copy=True)

        # Questo metodo è pensato anche per calcoli estemporanei, pertando restituisce il versore opportunamente
        # ruotato, ma non salva nessun effetto della precessione negli attributi dell'oggetto.
        return vector_obstime

    def at_date(self, obstime):
        """
        Questo metodo applica gli effetti della precessione (vedere `SkyLocation.precession_at_date`), salvandone gli
        effetti negli specifici attributi dell'oggetto e modificando anche il valore salvato delle coordinate.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

    def name_object(self, name, epoch):
        """
        Semplice funzione, utile per i grafici. Se all'oggetto celeste è stato assegnato un nome, questo verrà
        restituito dalla chiamata, altrimenti verranno riportate le coordinate, con indicata l'opportuna epoca.
        """
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
        """
        Metodo 'magico' di python per indicare cosa viene restituito nel momento in cui si chiede di stampare l'oggetto,
        esempio:

        >   from src.skylocation import SkyLocation
        >
        >   skyobject = SkyLocation("0h0m0s 0d0m0s")
        >   print(skyobject)

        """
        ra_epoch = self.ra_epoch.hms
        dec_epoch = np.array(self.dec_epoch.dms)
        ra, dec = self.__repr__().split()

        if dec_epoch[0] < 0:
            dec_epoch[1] *= -1
            dec_epoch[2] *= -1

        # Le coordinate all'epoca vengono convertite in una stringa
        ra_epoch = "{0:d}h{1:d}m{2:.3f}s".format(int(ra_epoch[0]), int(ra_epoch[1]), ra_epoch[2])
        dec_epoch = "{0:d}d{1:d}m{2:.3f}s".format(int(dec_epoch[0]), int(dec_epoch[1]), dec_epoch[2])

        # Si aggiunge l'equinozio di riferimento prima delle coordianate all'epoca, riportando poi correttamente il
        # segno della declinazione.
        string = "\t{} (Epoch):\n".format(Equinox2000.time.iso)
        if dec_epoch.find('-') >= 0:
            string_epoch = "RA:\t\t {0}\nDEC:\t{1}\n\n".format(ra_epoch, dec_epoch)
        else:
            string_epoch = "RA:\t\t{0}\nDEC:\t{1}\n\n".format(ra_epoch, dec_epoch)

        # Quanto fatto prima viene ripetuto con le coordinate alla data.
        string += string_epoch + "\t{} (Date):\n".format(self.obstime.iso)
        if dec.find('-') >= 0:
            string += "RA:\t\t {0}\nDEC:\t{1}".format(ra, dec)
        else:
            string += "RA:\t\t{0}\nDEC:\t{1}".format(ra, dec)

        return string

    def __repr__(self):
        """
        Metodo 'magico' di python per indicare cosa restituisce una chiamata all'istanza della classe.
        """
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
        """
        Metodo di classe che calcola l'effetto cumulativo della precessione degli equinozi dall'equinozio di riferimento
        alla data indicata. Al momento è disponibile solo l'equinozio vernale del 2000.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        # Restituisce la fase accumulata, in radianti, in rapporto al tempo di precessione.
        return ((2*np.pi/Tprec.value) * (obstime - reference.time).jd) % (2*np.pi) * u.rad

    @staticmethod
    def axial_tilt(obstime):
        """
        Metodo di classe che calcola l'angolo dell'asse di rotazione terrestre alla data. Si noti che è presente anche
        una formula polinomiale, ricavata dall'"Explanatory Supplement to the Astronomical Almanac", ma questa
        richiederebbe di ridefinire vari parametri (tra cui il tempo dell'equinozio di riferimento) come tempo apparente
        e non come tempo medio. Ho scelto di ignorare gli effetti della nutazione, restituendo sempre un valore fisso,
        preso anche questo dall'"Explanatory Supplement to the Astronomical Almanac".
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # check for Kenneth Seidelmann, "Explanatory Supplement to the astronomical almanac" p. 114.
        # T = (obstime.jd - tJ2000.jd) / 36525
        # return 23 * u.deg + 26 * u.arcmin + 21.448 * u.arcsec - \
        #             46.8150 * u.arcsec * T - \
        #             0.00059 * u.arcsec * T**2 + \
        #             0.001813 * u.arcsec * T**3

        # check for Earth Fact Sheet at https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html [23.44]
        # check for Kenneth Seidelmann, "Explanatory Supplement to the astronomical almanac" p. 315.
        return 23.43929111 * u.deg


import src.skylocation.sun
import src.skylocation.moon
