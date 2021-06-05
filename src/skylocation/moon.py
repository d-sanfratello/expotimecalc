import numpy as np

from astropy import units as u

from ..time import Time
from ..skylocation import SkyLocation

from ..constants import Omegasidmoon
from ..constants import Equinox2000

from .. import errmsg


class Moon(SkyLocation):
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, obstime):
        """
        Classe che descrive il moto della Luna in funzione della data di osservazione.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # La posizione iniziale, in coordinate equatoriali, è fornita dalle Effemeridi ricavate dal sito del JPL,
        # secondo quanto riportato a seguire. L'equinozio di riferimento è quello del 2000.:
        #
        # https://ssd.jpl.nasa.gov/horizons.cgi
        # Ephemeris Type [change] : 	OBSERVER
        # Target Body [change] : 	    Moon [Luna] [301]
        # Observer Location [change] : 	Greenwich [000] ( 0°00'00.0''E, 51°28'38.6''N, 65.8 m )
        # Time Span [change] : 	        Start=2000-03-20 07:35, Stop=2000-03-20 07:37, Intervals=120
        # Table Settings [change] : 	QUANTITIES=1,7,9,20,23,24
        # Display/Output [change] : 	default (formatted HTML)
        #
        # Selected RA-DEC for the time corresponding to `Equinox2000.time`.
        #
        # Distance obtained from https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
        super(Moon, self).__init__(locstring=None,
                                   ra='12h10m9.25s', dec='2d39m54.9s', distance=378e6 * u.m,
                                   obstime=obstime, ra_unit='hms', dec_unit='dms', epoch='J2000', name='Moon')

        self.at_date(obstime)

    def observe_at_date(self, obstime):
        """
        Metodo che calcola, senza salvarne il risultato, la posizione della Luna ad una specifica data.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Per ricavare le coordinate eclittiche della luna, all'epoca, il versore iniziale viene ruotato intorno
        # all'asse x per ottenere le coordinate eclittiche. A questo punto, anche se il nome dell'attributo è sempre
        # 'ra' e 'dec', tali valori rappresentano le coordinate 'b' e 'l' in un sistema di coordinate eclittiche.
        vector_ecl = self.vector_epoch.rotate_inv('x', self.axial_tilt(obstime), copy=True)
        b_ecliptic_lat = vector_ecl.dec
        l_ecliptic_lon = vector_ecl.ra

        # La posizione della Luna viene convertita in coordinate eclittiche, con una rotazione intorno all'asse x.
        # Quindi viene applicata una rotazione per portare tale vettore in coordinate (0,0) (e poter lavorare in maniera
        # più comoda con le rotazioni). Essendo in coordinate del piano orbitale lunare, viene applicata una rotazione
        # intorno a z secondo il periodo di rivoluzione della Luna intorno alla Terra, quindi il vettore risultante
        # viene riportato in coordinate eclittiche con le opportune rotazioni inverse. Infine il vettore risultante è
        # riportato in coordinate equatoriali. Il risultato complessivo è un operatore applicato dopo un complesso
        # cambio di base.
        vector_obstime = self.vector_epoch.rotate_inv('x', self.axial_tilt(obstime), copy=True)\
            .rotate_inv('z', l_ecliptic_lon, copy=True).rotate('y', b_ecliptic_lat, copy=True)\
            .rotate('z', self.moon_revolution(obstime), copy=True) \
            .rotate_inv('y', b_ecliptic_lat, copy=True).rotate('z', l_ecliptic_lon, copy=True) \
            .rotate('z', self.equinox_prec(obstime), copy=True)\
            .rotate('x', self.axial_tilt(obstime), copy=True)

        return vector_obstime

    def at_date(self, obstime):
        """
        Metodo che salva il risultato di `Moon.observe_at_date` nei relativi attributi di classe.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

    @classmethod
    def moon_revolution(cls, obstime, epoch_eq='equinoxJ2000'):
        """
        Metodo di classe che calcola la fase data dalla rivoluzione della Luna intorno alla Terra, rispetto
        all'equinozio di riferimento.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        return (Omegasidmoon.value * (obstime - reference.time).jd) % (2 * np.pi) * u.rad

    def calculate_moon_phase(self, sun, obstime):
        """
        Metodo che calcola la fase della Luna, dato il Sole. Notare che non vi è alcuna differenza tra fase crescente e
        calante. Il metodo restituisce `1` se la Luna è piena e `0` se è nuova.
        """
        if not isinstance(sun, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('sun', 'src.skylocation.sun.Sun'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        moon_obstime = self.observe_at_date(obstime)
        sun_obstime = sun.observe_at_date(obstime)

        # Calcolo del prodotto scalare dei vettori, rinormalizzato per dare il valore richiesto. Nel caso di Luna Nuova,
        # il prodotto scalare varrebbe `1`, e `-1` nel caso di Luna Piena. In questo modo si restituiscono i valori
        # dichiarati nella descrizione del metodo.
        return (- moon_obstime.vsr.dot(sun_obstime.vsr) + 1) / 2
