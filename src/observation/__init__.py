import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from astropy.coordinates import Angle
from astropy.units.quantity import Quantity
from astropy.visualization import time_support
from astropy.visualization import quantity_support

from src.location import Location
from src.skylocation import SkyLocation
from src.skylocation.sun import Sun
from src.skylocation.moon import Moon
from src.time import Time

from src import Tsidday
from src import Tsidyear
from src import Equinox2000

from src import errmsg
from src import warnmsg

import warnings


time_support(scale='utc', format='iso', simplify=True)
quantity_support()


# noinspection PyUnresolvedReferences
class Observation:
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, location, obstime=None, target=None):
        """
        Classe che, data una località, una data di osservazione ed un target, è in grado di effettuare una serie di
        calcoli relativi all'osservazione del target, come la posizione in coordinate Alt-Az, i tempi di alba, tramonto
        e culminazione, la visibilità sopra l'orizzonte o una certa altezza. Inoltre è possibile ricavare l'angolo
        orario del target, l'airmass e la distanza dallo zenith. Infine è possibile, dato il target, determinare la
        migliore data di osservazione nell'anno successivo alla data, in funzione della posizione del Sole.
        """
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))

        self.location = location
        self.obstime = obstime
        self.target = target
        self.sun = Sun(self.obstime)
        self.moon = Moon(self.obstime)

        self.ha = None
        self.az = None

        self.alt = None
        self.zenith_dist = None
        self.airmass = None

        self.culmination = None
        self.visibility = None

        self.rise_time = None
        self.rise_azimuth = None
        self.rise_ha = None

        self.set_time = None
        self.set_azimuth = None
        self.set_ha = None

        self.make_observation(self.obstime)

    def make_observation(self, obstime):
        """
        Metodo che 'effettua' l'osservazione ad una determinata data, calcolando i parametri rilevanti del target.
        """
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime

        # Viene calcolato l'effetto della precessione al target, la posizione dello zenith alla data e le posizioni di
        # Sole e Luna, con i metodi definiti nei rispettivi moduli.
        self.target.at_date(self.obstime)
        self.location.zenith_at_date(self.obstime, copy=False)
        self.sun.at_date(self.obstime)
        self.moon.at_date(self.obstime)

        # Vengono calcolati angolo orario e azimuth del target.
        self.ha = self.calculate_ha(self.target, self.location, self.obstime)
        self.az = self.calculate_az(self.target, self.location, self.obstime)

        # Vengono calcolati altezza, distanza zenitale e airmass del target.
        self.alt = self.calculate_alt(self.target, self.location, self.obstime)
        self.zenith_dist = self.calculate_zenith_dist(self.target, self.location, self.obstime)
        self.airmass = self.calculate_airmass(self.target, self.location, self.obstime)

        # Vengono calcolati i tempi di culminazione, di alba e tramonto del target.
        self.culmination = self.calculate_culmination(self.target, self.location, self.obstime)
        self.set_time = self.calculate_set_time(self.target, self.location, self.obstime)
        self.rise_time = self.calculate_rise_time(self.target, self.location, self.obstime)

        # Viene calcolata la permanenza del target sopra l'orizzonte.
        self.visibility = self.calculate_visibility(self.target, self.location, self.obstime)

        # Vengono calcolati gli azimuth di alba e tramonto del target.
        self.rise_azimuth = self.calculate_az(self.target, self.location, self.rise_time)
        self.set_azimuth = self.calculate_az(self.target, self.location, self.set_time)

    @classmethod
    def calculate_ha(cls, target, location, obstime, epoch_eq='equinoxJ2000'):
        """
        Metodo di classe per il calcolo dell'angolo orario del target.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        # Viene calcolata la posizione del target alla data di osservazione
        target_obstime = target.precession_at_date(obstime)

        # Viene calcolato l'angolo orario, ricordando che:
        #
        #       HA = LST - RA
        #
        # Per il calcolo del LST, viene chiamato il metodo apposito della classe `src.location.Location`.
        return (location.lst(obstime, epoch_eq) - target_obstime.ra).to(u.hourangle) % (24 * u.hourangle)

    @classmethod
    def calculate_az(cls, target, location, obstime, epoch_eq='equinoxJ2000'):
        """
        Metodo di classe per il calcolo dell'azimuth del target.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        # Viene calcolata la posizione del target alla data di osservazione
        target_obstime = target.precession_at_date(obstime)

        # Viene calcolato l'angolo orario del target utilizzando il metodo di classe `Observation.calculate_ha` e la
        # distanza dallo zenith.
        hangle = cls.calculate_ha(target, location, obstime, epoch_eq)
        zdist = cls.calculate_zenith_dist(target, location, obstime)

        # Prima viene ridefinito l'angolo orario come positivo se ad ovest del Sud e negativo altrimenti, poi viene
        # effettuato lo stesso calcolo, cioè, dopo le seguenti definizioni:
        #   * N,E,S,W: i punti cardinali;
        #   * X: la posizione del target al tempo di osservazione;
        #   * L: il punto di culminazione del target;
        #   * Z, P: rispettivamente lo zenith e il polo nord celeste
        #
        # per argomenti di geometria sferica (vedere Smart, Green) è possibile scrivere che:
        #
        #       cos(XL) = cos(XZ)cos(ZL) + sin(XZ)sin(ZL)cos(XZL)
        #
        # dove XL, XZ e ZL sono gli archi di circonferenza definiti dai punti precedenti. XZL è l'angolo compreso tra i
        # tre punti definiti prima. Tali quantità corrispondono, rispettivamente a:
        #   * Arco associato all'angolo orario tra la posizione attuale e la culminazione, ignoto;
        #   * Distanza dallo zenith, z;
        #   * PL - PZ = 90° - DEC - (90° - Lat) = Lat - DEC;
        #   * Az - 180°.
        #
        # Per trovare la quantità ignota XL, si utilizza di nuovo la stessa relazione, ma applicata tra X, P ed L:
        #
        #       cos(XL) = cos(PX)cos(PL) + sin(PX)sin(PL)cos(XPL)
        #
        # dove XL, PX e PL sono gli archi di circonferenza definiti dai punti precedenti. XPL è l'angolo compreso tra i
        # tre punti indicati e corrispondono, rispettivamente, a:
        #   * Arco associato all'angolo orario tra la posizione attuale e la culminazione, ignoto;
        #   * 90° - DEC;
        #   * 90° - DEC;
        #   * Angolo orario, come definito precedentemente.
        #
        # Eguagliando le due espressioni per cos(XL) e dopo alcune manipolazioni si ottiene l'espressione, qui usata:
        #
        #                  /  sin^2(DEC) + cos(HA) cos^2(DEC) - cos(z) cos(Lat - DEC)\
        #       Az = arccos|- -------------------------------------------------------|
        #                  \                  sin(z) sin(Lat - DEC)                  /
        #
        # Dato che questa formula, in realtà, permette di ottenere l'angolo compreso tra il sud e l'azimut dell'oggetto,
        # nel caso di HA > 12h è necessario sottrarre il risultato a 360°.
        if hangle > 12 * u.hourangle:
            hangle -= 24 * u.hourangle

            cos_az = np.sin(target_obstime.dec)**2 + np.cos(target_obstime.dec)**2 * np.cos(hangle)
            cos_az -= np.cos(location.lat - target_obstime.dec) * np.cos(zdist)
            cos_az /= np.sin(zdist) * np.sin(location.lat - target_obstime.dec)

            return np.arccos(-cos_az).to(u.deg)
        else:
            cos_az = np.sin(target_obstime.dec) ** 2 + np.cos(target_obstime.dec) ** 2 * np.cos(hangle)
            cos_az -= np.cos(location.lat - target_obstime.dec) * np.cos(zdist)
            cos_az /= np.sin(zdist) * np.sin(location.lat - target_obstime.dec)

            return 360 * u.deg - np.arccos(-cos_az).to(u.deg)

    @classmethod
    def calculate_alt(cls, target, location, obstime):
        """
        Metodo di classe che calcola l'altezza sull'orizzonte del target.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Vengono ricavati i versori che indicano la posizione in cielo del target e dello zenith alla data, chiamando
        # gli opportuni metodi.
        target_vsr = target.precession_at_date(obstime).vsr
        zenith_vsr = location.zenith_at_date(obstime, axis='z', copy=True).vsr

        # Viene effettuata la proiezione del versore del target sullo zenith e viene calcolato l'arccos di questa
        # quantità. Risulta necessario dividere le funzione per ottenere la rappresentazione corretta di angoli negativi
        # (target sotto l'orizzonte). Non sono considerati effetti atmosferici.
        if (ps := zenith_vsr.dot(target_vsr)) >= 0:
            return Angle((90 - np.rad2deg(np.arccos(ps))) % 90 * u.deg)
        else:
            return Angle((90 - np.rad2deg(np.arccos(ps))) % -90 * u.deg)

    @classmethod
    def estimate_quality(cls, target, location, obstime, sun,
                         parameter=1.5, interval=2*u.hour, par_type='airmass'):
        """
        Metodo che stima la durata della permanenza di un oggetto al di sopra di una certa altezza dall'orizzonte, entro
        una certa distanza dallo zenith o una certa airmass. Inoltre riporta se tale oggetto permane sopra tale altezza
        minima per un tempo superiore ad un intervallo indicato dall'utente, fissato di default a 2h.
        """
        # https://www.weather.gov/fsd/twilight for twilight definitions
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(sun, Sun):
            raise TypeError(errmsg.notTypeError.format('sun', 'src.skylocation.sun.Sun'))
        if not isinstance(parameter, (Quantity, int)):
            raise TypeError(errmsg.notTwoTypesError.format('parameter', 'astropy.units.quantity.Quantity', 'int'))
        if not isinstance(interval, Quantity):
            raise TypeError(errmsg.notTypeError.format('interval', 'astropy.units.quanrtity.Quantity'))
        if not isinstance(par_type, str):
            raise TypeError(errmsg.notTypeError.format('par_type', 'string'))
        elif isinstance(parameter, Quantity) and par_type not in ['alt', 'z']:
            raise ValueError(errmsg.altZError)
        elif isinstance(parameter, int) and par_type not in ['airmass']:
            raise ValueError(errmsg.airmassError)

        # I calcoli nel codice sono effettuati in distanza zenitale, per cui parametri forniti in altri modi sono
        # convertiti in z.
        if par_type == 'alt':
            z_max = 90 * u.deg - parameter
        elif par_type == 'z':
            z_max = parameter
        elif par_type == 'airmass':
            # arcsec(1/x) = arccos(x)
            z_max = np.arccos(1/parameter)

        # Con il metodo di classe opportuno, viene calcolata la distanza dallo zenith alla culminazione, anche questa
        # ottenuta da uno specifico metodo di classe. Se la distanza alla culminazione è maggiore della distanza minima
        # accettata, il metodo restituisce `False` e `0h`, cioè il tempo di permanenza sopra l'altezza minima.
        if cls.calculate_zenith_dist(target, location, cls.calculate_culmination(target, location, obstime)) >= z_max:
            return False, 0*u.hour

        # Con una metodologia analoga a quella utilizzata (e spiegata) più avanti per il calcolo della visibilità, si
        # calcola la visibilità di un target entro una certa distanza zenitale. In un sistema di coordinate alt-az, si
        # definiscono:
        #   * Z, P: rispettivamente lo zenith e il Polo Nord Celeste;
        #   * F: il punto, ad est o ad ovest, in cui la distanza del target dallo zenith è quella massima indicata.
        #
        # Per argomenti di geometria sferica, si può scrivere che:
        #
        #       cos(ZF) = cos(PZ)cos(PF) + sin(PZ)sin(PF)cos(ZPF)
        #
        # dove:
        #   * ZF = z_max;
        #   * PZ = 90° - Lat;
        #   * PF = 90° - DEC;
        #   * ZPF: HA del target al momento del raggiungimento della distanza zenitale massima (nel caso F sia l'ovest).
        #
        # Risulta quindi, dopo alcune manipolazioni, scrivere:
        #
        #                  / cos(z_max) - sin(Lat) sin(DEC)\
        #       HA = arccos|-------------------------------|
        #                  \       cos(Lat) cos(DEC)       /
        #
        # Tuttavia, conoscendo la durata del giorno siderale, è identico al tempo percorso tra la culminazione e il
        # raggiungimento della z_max indicata e il doppio di questo valore indica la permanenza sopra tale altezza.
        else:
            sidday_factor = Tsidday.to(u.hour) / (2 * np.pi * u.rad)

            time_above = np.cos(z_max) - np.sin(location.lat) * np.sin(target.dec)
            time_above /= np.cos(location.lat * np.cos(target.dec))
            time_above = 2 * sidday_factor * np.arccos(time_above)

            return (time_above >= interval), time_above

    @classmethod
    def calculate_zenith_dist(cls, target, location, obstime):
        """
        Metodo di classe che calcola la distanza zenitale del target.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        alt = cls.calculate_alt(target, location, obstime)

        # Dopo aver calcolato l'altezza sull'orizzonte con il metodi di classe adatto, viene calcolata z come 90° - alt.
        return 90 * u.deg - alt

    @classmethod
    def calculate_airmass(cls, target, location, obstime):
        """
        Metodo di classe per il calcolo dell'airmass.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        warnings.warn(warnmsg.airmassWarning)

        # Calcolo dei vettori indicativi del target e dello zenith alla data.
        target_vsr = target.precession_at_date(obstime).vsr
        zenith_vsr = location.zenith_at_date(obstime, axis='z', copy=True).vsr

        if (ps := zenith_vsr.dot(target_vsr)) > 0:
            # airmass = sec(z) = 1/cos(z)
            return 1 / ps
        else:
            return 0

    @classmethod
    def calculate_culmination(cls, target, location, obstime):
        """
        Calcolo della culminazione del target, alla data.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.precession_at_date(obstime)

        # Se declinazione e latitudine sono discordi in segno (target sotto l'orizzonte) e |DEC| >= |Lat|, il target è
        # considerato invisibile e la funzione ritorna un NoneType.
        if target_obstime.dec * location.lat <= 0 and abs(target_obstime.dec) >= abs(location.lat):
            return None
        # Se il target è visibile (o circumpolare), viene calcolata la posizione del target alla mezzanotte della data
        # indicata, a cui si ricalcolano le posizioni del target e dello zenith.
        # Dato che, alla culminazione, la RA del target indica anche il LST della località (per definizione di transito
        # al meridiano) è possibile calcolare il tempo necessario ad arrivare al transito come, a meno del fattore di
        # conversione del giorno siderale, la distanza tra la RA del target alla mezzanotte e lo zenith alla mezzanotte.
        # Aggiungendo tale valore alla data, indicata alla mezzanotte, si ottiene la culminazione. Dato che il target
        # ha HA < 0 prima della culminazione, viene aggiunto un angolo di 360° per ottenere la culminazione alla data.
        else:
            obstime = Time(int(obstime.mjd), format='mjd', scale='utc')
            obstime.format = 'iso'

            target_obstime = target.precession_at_date(obstime)
            zenith_obstime = location.zenith_at_date(obstime, axis='z', copy=True)

            sidday_factor = Tsidday / (2 * np.pi * u.rad).to(u.deg)

            return obstime + (360*u.deg + target_obstime.ra - zenith_obstime.ra).to(u.deg) * sidday_factor
            # what if target is ahead of LST? Doesn't this return the lowest point on apparent motion? Check.

    @classmethod
    def calculate_set_time(cls, target, location, obstime):
        """
        Metodo di classe per il calcolo dell'orario di tramonto del target alla data.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.precession_at_date(obstime)

        # Se il target è circumpolare o non visibile, viene restituito un Nonetype.
        if target_obstime.dec >= location.lat or target_obstime.dec <= -location.lat:
            return None

        # Viene calcolato l'orario di culminazione dal metodo di classe apposito e viene aggiunto metà del periodo di
        # visibilità sopra l'orizzonte, per il target, anche questo calcolato con il metodo di classe apposito.
        culm_t = cls.calculate_culmination(target, location, obstime)
        visibility_window = cls.calculate_visibility(target, location, obstime)
        return culm_t + visibility_window / 2

    @classmethod
    def calculate_rise_time(cls, target, location, obstime):
        """
        Metodo di classe per il calcolo dell'orario dell'alba del target alla data.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.precession_at_date(obstime)

        # Se il target è circumpolare o non visibile, viene restituito un Nonetype.
        if target_obstime.dec >= location.lat or target_obstime.dec <= -location.lat:
            return None

        # Viene calcolato l'orario di culminazione dal metodo di classe apposito e viene sottratto metà del periodo di
        # visibilità sopra l'orizzonte, per il target, anche questo calcolato con il metodo di classe apposito.
        culm_t = cls.calculate_culmination(target, location, obstime)
        visibility_window = cls.calculate_visibility(target, location, obstime)
        return culm_t - visibility_window / 2

    @classmethod
    def calculate_visibility(cls, target, location, obstime):
        """
        Metodo che stima la durata della permanenza di un oggetto al di sopra dell'orizzonte.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        warnings.warn(warnmsg.visibilityWarning)

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.precession_at_date(obstime)

        # Se il target è circumpolare viene restituito il valore di 24h. Se invece non è visibile viene restituito 0h.
        if abs(target_obstime.dec) >= abs(location.lat):
            if target_obstime.dec * location.lat > 0:
                return 24 * u.hour
            elif target_obstime.dec * location.lat <= 0:
                return 0 * u.hour
        # Con una metodologia analoga a quella utilizzata (e spiegata) prima per la stima della qualità
        # dell'osservazione, si calcola la visibilità di un target entro una certa distanza zenitale. In un sistema di
        # coordinate alt-az, si definiscono:
        #   * Z, P: rispettivamente lo zenith e il Polo Nord Celeste;
        #   * F: il punto del tramonto dell'oggetto.
        #
        # Per argomenti di geometria sferica, si può scrivere che:
        #
        #       cos(ZF) = cos(PZ)cos(PF) + sin(PZ)sin(PF)cos(ZPF)
        #
        # dove:
        #   * ZF = 90°;
        #   * PZ = 90° - Lat;
        #   * PF = 90° - DEC;
        #   * ZPF: HA del target al tramonto.
        #
        # Risulta quindi, dopo alcune manipolazioni, scrivere:
        #
        #       HA = arccos(- tan(Lat) tan(DEC))
        #
        # Tuttavia, conoscendo la durata del giorno siderale, è identico al tempo percorso tra la culminazione e il
        # momento del tramonto e il doppio di questo valore indica la permanenza sopra l'orizzonte.
        else:
            sidday_factor = Tsidday.to(u.hour) / (2 * np.pi * u.rad)
            return 2 * sidday_factor * np.arccos(- np.tan(target_obstime.dec.rad) * np.tan(location.lat))

    @classmethod
    def calculate_best_day(cls, target, location, obstime, epoch_eq='equinoxJ2000'):
        """
        Metodo di classe per il calcolo del miglior periodo dell'anno per l'osservazione di un target.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if epoch_eq != 'equinoxJ2000':
            raise NotImplementedError(errmsg.epochNotImplemented)

        reference = cls.equinoxes[epoch_eq]

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.precession_at_date(obstime)

        # Se declinazione e latitudine sono discordi in segno (target sotto l'orizzonte) e |DEC| >= |Lat|, il target è
        # considerato invisibile e la funzione ritorna un NoneType.
        if target_obstime.dec * location.lat <= 0 and abs(target_obstime.dec) >= abs(location.lat):
            return None
        # Il giorno migliore per l'osservazione di un corpo celeste è quando il sole ha RA pari alla RA del target + 12h
        # Per ricavare tale data, si considera il sole in culminazione, con ascensione retta pari a RA + 12h. A questo
        # valore si sottrae il GMST all'equinozio di riferimento e la latitudine, oltre che il contributo, in RA, del
        # fuso orario. Per il giorno ricavato, si calcola la culminazione alla data, con il metodo di classe descritto
        # prima.
        else:
            sidyear_factor = (Tsidyear.value/(2*np.pi))
            time = target_obstime.ra + 180 * u.deg - reference.GMST.rad - location.lon
            time -= location.timezone * 15 * u.deg/u.hour
            time = sidyear_factor * time.rad * u.day

            best_target = SkyLocation(ra=target.ra, dec=target.dec, obstime=time+obstime)

            return cls.calculate_culmination(best_target, location, time + obstime)

    def plot_altaz_onday(self, interval=15*u.min):
        interval = interval.to(u.hour)
        numpoints = int(24*u.hour/interval)

        s_time = Time(int(self.obstime.mjd), format='mjd')
        dt = 15*u.min * np.linspace(0, numpoints, num=numpoints + 1)
        times = np.array([s_time + delta for delta in dt])

        alt = np.empty(numpoints + 1, dtype=u.quantity.Quantity)
        az = np.empty(numpoints + 1, dtype=u.quantity.Quantity)
        index = 0

        for t in times:
            alt[index] = self.calculate_alt(self.target, self.location, t)
            az[index] = self.calculate_az(self.target, self.location, t)

            index += 1

        times = Time([t for t in times], scale='utc')
        times = times.to_value('iso', subfmt='date_hm')
        alt = u.quantity.Quantity([at for at in alt])
        az = u.quantity.Quantity([z for z in az])

        plt.close(1)
        fig = plt.figure(1)

        fig.suptitle(self.target.name)

        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax1.grid()
        ax1.plot(times, alt, 'k-')
        ax1.set_ylim(-90, 90)
        ax1.yaxis.set_ticks(np.linspace(-90, 90, 7))
        ax1.xaxis.set_major_locator(plt.MaxNLocator(25))
        ax1.set_xticklabels([])
        ax1.set_xlabel('')
        ax1.set_ylabel('Alt [deg]')

        ax2 = plt.subplot2grid((2, 1), (1, 0))
        ax2.grid()
        ax2.plot(times, az, 'k-')
        ax2.set_ylim(0, 360)
        ax2.yaxis.set_ticks(np.linspace(0, 360, 13))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(25))
        ax2.xaxis.set_tick_params(rotation=80)
        ax2.set_ylabel('Az [deg]')

        ax1.set_xlim(min(times), max(times))
        ax2.set_xlim(min(times), max(times))

        fig.tight_layout()
        fig.show()
