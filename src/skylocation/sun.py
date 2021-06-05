import numpy as np

from astropy import units as u
from astropy import constants as cts

from ..time import Time
from ..skylocation import SkyLocation

from ..constants import Tsidyear
from ..constants import Equinox2000

from .. import errmsg


class Sun(SkyLocation):
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, obstime):
        """
        Classe che definisce la posizione del Sole alla data. Eredita dalla classe `src.skylocation.SkyLocation`.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Viene inizializzato alla posizione (0,0), che è quella del Sole alla data dell'equinozio di riferimento.
        super(Sun, self).__init__(locstring=None, ra=0*u.deg, dec=0*u.deg, distance=1 * cts.au,
                                  obstime=obstime, ra_unit='deg', dec_unit='deg', epoch='J2000', name='Sun')

        self.at_date(obstime)

    def observe_at_date(self, obstime):
        """
        Metodo che calcola, senza variare gli attributi di classe, la posizione del Sole alla data indicata.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Il Sole viene prima convertito in coordinate eclittiche (più per forma che per utilità, dato che ha coordinate
        # cartesiane (0,0,0)) e ruotato usando l'angolo calcolato con il metodo `Sun.sidereal_year_rotation`. Viene,
        # quindi, applicata la precessione e, infine, viene riportato in coordinate equatoriali.
        vector_obstime = self.vector_epoch.rotate_inv('x', self.axial_tilt(obstime), copy=True) \
            .rotate('z', self.sidereal_year_rotation(obstime), copy=True) \
            .rotate('z', self.equinox_prec(self.obstime), copy=True) \
            .rotate('x', self.axial_tilt(obstime), copy=True)

        return vector_obstime

    def at_date(self, obstime):
        """
        Metodo di classe che calcola la rotazione secondo il metodo `Sun.observe_at_date` e ne salva il risultato nel
        relativo attributo di classe.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec

    @classmethod
    def sidereal_year_rotation(cls, obstime, epoch_eq='equinoxJ2000'):
        """
        Metodo di classe che calcola la fase nella posizione apparente del sole dovuta alla rivoluzione della Terra
        intorno alla stella. Il calcolo è effettuato rispetto alla data dell'equinozio di riferimento, quello del 2000.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        # Si calcola la fase rispetto al periodo di rivoluzione di un anno siderale.
        return ((2 * np.pi / Tsidyear.value) * (obstime - reference.time).jd) % (2*np.pi) * u.rad
