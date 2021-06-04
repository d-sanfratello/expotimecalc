import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import warnings

from astropy import units as u
from astropy.coordinates import Angle
from astropy.units.quantity import Quantity
from astropy.visualization import time_support
from astropy.visualization import quantity_support
from matplotlib.colors import Normalize

from ..location import Location
from ..skylocation import SkyLocation
from ..skylocation.sun import Sun
from ..skylocation.moon import Moon
from ..time import Time

from .. import Tsidday
from .. import Tsidyear
from .. import Equinox2000

from .. import errmsg
from .. import warnmsg


time_support(scale='utc', format='iso', simplify=True)
quantity_support()


# noinspection PyUnresolvedReferences
class Observation:
    equinoxes = {'equinoxJ2000': Equinox2000}

    def __init__(self, location, obstime=None):
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

        self.location = location
        self.obstime = obstime
        self.sun = Sun(self.obstime)
        self.moon = Moon(self.obstime)

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

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Viene calcolata la posizione del target alla data di osservazione
        target_obstime = target.observe_at_date(obstime)

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

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Viene calcolata la posizione del target alla data di osservazione
        target_obstime = target.observe_at_date(obstime)

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

            return Angle(np.arccos(-cos_az).to(u.deg))
        else:
            cos_az = np.sin(target_obstime.dec) ** 2 + np.cos(target_obstime.dec) ** 2 * np.cos(hangle)
            cos_az -= np.cos(location.lat - target_obstime.dec) * np.cos(zdist)
            cos_az /= np.sin(zdist) * np.sin(location.lat - target_obstime.dec)

            return Angle(360 * u.deg - np.arccos(-cos_az).to(u.deg))

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

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Vengono ricavati i versori che indicano la posizione in cielo del target e dello zenith alla data, chiamando
        # gli opportuni metodi.
        target_vsr = target.observe_at_date(obstime).vsr
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
        if not isinstance(parameter, (Quantity, int, float)):
            raise TypeError(errmsg.notThreeTypesError.format('parameter',
                                                             'astropy.units.quantity.Quantity', 'int', 'float'))
        if not isinstance(interval, Quantity):
            raise TypeError(errmsg.notTypeError.format('interval', 'astropy.units.quanrtity.Quantity'))

        if not isinstance(par_type, str):
            raise TypeError(errmsg.notTypeError.format('par_type', 'string'))
        elif isinstance(parameter, Quantity) and par_type not in ['alt', 'z']:
            raise ValueError(errmsg.altZError)
        elif isinstance(parameter, int) and par_type not in ['airmass']:
            raise ValueError(errmsg.airmassError)

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)
        warnings.warn("`sun` parameter will be removed in a future version, by using the internal instance of `Sun` "
                      "class.",
                      DeprecationWarning)

        # I calcoli nel codice sono effettuati in distanza zenitale, per cui parametri forniti in altri modi sono
        # convertiti in z.
        if par_type == 'alt':
            z_max = 90 * u.deg - parameter
        elif par_type == 'z':
            z_max = parameter
        elif par_type == 'airmass':
            # arcsec(1/x) = arccos(x)
            z_max = (np.arccos(1/parameter) * u.rad).to(u.deg)

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
            target_obstime = target.observe_at_date(obstime)

            sidday_factor = Tsidday.to(u.hour) / (2 * np.pi * u.rad)

            time_above = np.cos(z_max) - np.sin(location.lat) * np.sin(target_obstime.dec)
            time_above /= (np.cos(location.lat) * np.cos(target_obstime.dec))
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

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)
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
        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Calcolo dei vettori indicativi del target e dello zenith alla data.
        target_vsr = target.observe_at_date(obstime).vsr
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

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.observe_at_date(obstime)

        # Se declinazione e latitudine sono discordi in segno (target sotto l'orizzonte) e |DEC| >= |Lat|, il target è
        # considerato invisibile e la funzione ritorna un NoneType.
        if target_obstime.dec * location.lat <= 0 and abs(target_obstime.dec) >= abs(location.lat):
            return None
        # Se il target è visibile (o circumpolare), viene calcolata la posizione del target alla data giuliana intera
        # più vicina a quella indicata, in cui ricalcoliamo sia le posizioni del target che dello zenith.
        # Dato che, alla culminazione, la RA del target indica anche il LST della località (per definizione di transito
        # al meridiano) è possibile calcolare il tempo necessario ad arrivare al transito come, a meno del fattore di
        # conversione del giorno siderale, la distanza tra la RA del target al mezzogiorno e lo zenith al mezzogiorno.
        # Aggiungendo tale valore alla data, indicata al mezzogiorno, si ottiene la culminazione. Se al momento della
        # culminazione il target è sotto l'orizzonte, si suppone di aver trovato il punto di altezza minima e si somma
        # mezzo giorno siderale. Se la culminazione così definita avviene prima della data indicata, viene aggiunto un
        # giorno siderale.
        else:
            obstime_midday = Time(int(np.round(obstime.jd)), format='jd', scale='utc')
            obstime_midday.format = 'iso'

            target_obstime = target.observe_at_date(obstime_midday)
            zenith_obstime = location.zenith_at_date(obstime_midday, axis='z', copy=True)

            sidday_factor = Tsidday / (2 * np.pi * u.rad).to(u.deg)

            delta_time = (target_obstime.ra - zenith_obstime.ra).to(u.deg) * sidday_factor

            culm = obstime_midday + delta_time

            if cls.calculate_alt(target, location, culm) < 0:
                culm += 0.5 * Tsidday
            if culm.jd <= obstime.jd:
                culm += Tsidday

            return culm

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

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.observe_at_date(obstime)

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

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.observe_at_date(obstime)

        # Se il target è circumpolare o non visibile, viene restituito un Nonetype.
        if target_obstime.dec >= location.lat or target_obstime.dec <= -location.lat:
            return None

        # Viene calcolato l'orario di culminazione dal metodo di classe apposito e viene sottratto metà del periodo di
        # visibilità sopra l'orizzonte, per il target, anche questo calcolato con il metodo di classe apposito.
        culm_t = cls.calculate_culmination(target, location, obstime)
        visibility_window = cls.calculate_visibility(target, location, obstime)
        return culm_t - visibility_window / 2

    @classmethod
    def calculate_twilight(cls, sun, location, obstime, twilight='nautical'):
        """
        Metodo di classe per il calcolo di inizio e fine del crepuscolo navale ed astronomico. Dati sulle altezze di
        definizione dei due crepuscoli ricavati da `https://www.weather.gov/fsd/twilight`
        """
        if not isinstance(sun, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('sun', 'src.skylocation.sun.Sun'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        if not isinstance(twilight, str):
            raise TypeError(errmsg.notTypeError.format('twilight', 'string'))
        elif twilight.lower() not in ['nautical', 'astronomical']:
            raise ValueError(errmsg.twilightError)

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)
        warnings.warn("`sun` parameter will be removed in a future version, by using the internal instance of `Sun` "
                      "class.",
                      DeprecationWarning)

        if twilight.lower() == 'nautical':
            alt = -12 * u.deg
        elif twilight.lower() == 'astronomical':
            alt = -18 * u.deg

        # La data di osservazione viene modificata per essere in data giuliana. Questo perché, essendo le osservazioni
        # notturne, permette di considerare la durata del giorno come dalle 12:00 alle 12:00 del giorno successivo. Se
        # l'orario impostato è già oltre le 12:00 UTC, si calcola per la data giuliana modificata, così da includere la
        # culminazione corretta.
        if obstime.jd % 1 >= 0.5:
            obstime = Time(int(obstime.jd), format='jd', scale='utc')
        else:
            obstime = Time(int(obstime.mjd), format='mjd', scale='utc')
        obstime.format = 'iso'

        # In maniera analoga a quanto calcolate nei metodi `calculate_set_time` e `calculate_rise_time`, viene stimato
        # l'orario della culminazione del Sole alla data.
        sun_culm = cls.calculate_culmination(sun, location, obstime)

        # Per migliorare la precisione, si reimposta l'orario con quello di culminazione.
        sun_obstime = sun.observe_at_date(sun_culm)

        # Analogamente al metodo `estimate_quality`, si calcola quanto tempo il Sole rimane al di sopra della soglia
        # impostata.
        sidday_factor = Tsidday.to(u.hour) / (2 * np.pi * u.rad)
        half_over_alt_time = np.sin(alt) - np.sin(location.lat) * np.sin(sun_obstime.dec)
        half_over_alt_time /= (np.cos(location.lat) * np.cos(sun_obstime.dec))
        half_over_alt_time = sidday_factor * np.arccos(half_over_alt_time)

        # Viene restituita la culminazione alla data, il tempo dell'inizio del prossimo crepuscolo e la sua fine.
        return sun_culm, sun_culm + half_over_alt_time, sun_culm + Tsidday - half_over_alt_time

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
        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.observe_at_date(obstime)

        # Se il target è circumpolare viene restituito il valore di 24h. Se invece non è visibile viene restituito 0h.
        if abs(target_obstime.dec) >= abs(location.lat):
            if target_obstime.dec * location.lat > 0:
                return 24 * u.hour
            elif target_obstime.dec * location.lat <= 0:
                return 0 * u.hour
        # Con una metodologia analoga a quella utilizzata (e spiegata) prima per la stima della qualità
        # dell'osservazione, si calcola la visibilità di un target entro una distanza zenitale di 90°. In un sistema di
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
    def calculate_best_day(cls, target, location, obstime):
        """
        Metodo di classe per il calcolo del miglior periodo dell'anno per l'osservazione di un target.
        """
        if not isinstance(target, SkyLocation):
            raise TypeError(errmsg.notTypeError.format('target', 'src.skylocation.SkyLocation'))
        if not isinstance(location, Location):
            raise TypeError(errmsg.notTypeError.format('location', 'src.location.Location'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)

        # Viene calcolata la posizione del target alla data.
        target_obstime = target.observe_at_date(obstime)

        # Se declinazione e latitudine sono discordi in segno (target sotto l'orizzonte) e |DEC| >= |Lat|, il target è
        # considerato invisibile e la funzione ritorna un NoneType.
        if target_obstime.dec * location.lat <= 0 and abs(target_obstime.dec) >= abs(location.lat):
            return None
        # Il giorno migliore per l'osservazione di un corpo celeste è quando il sole ha RA pari alla RA del target + 12h
        # Per ricavare tale data, si calcola la frazione di angolo giro (quindi di anno siderale) che manca per avere il
        # Sole e il target in opposizione. Il calcolo viene eseguito una seconda volta rispetto alla data ottenuta con
        # la prima iterazione per correggere per gli effetti della precessione durante l'anno (comunque inferiori a
        # 50"). Corretta la data, viene calcolato il momento esatto della culminazione del target alla data, con il
        # metodo di classe descritto prima.
        else:
            sun = Sun(obstime)

            factor_sidyear = Tsidyear / (360 * u.deg)

            if sun.ra >= target_obstime.ra:
                if (delta_phi := sun.ra - target_obstime.ra) <= 180 * u.deg:
                    delta_time = factor_sidyear * (180 * u.deg - delta_phi)
                else:
                    delta_time = factor_sidyear * (540 * u.deg - delta_phi)
            else:
                if (delta_phi := target_obstime.ra - sun.ra) <= 180 * u.deg:
                    delta_time = factor_sidyear * (180 * u.deg + delta_phi)
                else:
                    delta_time = factor_sidyear * (delta_phi - 180 * u.deg)

            sun_besttime = sun.observe_at_date(obstime + delta_time)
            target_besttime = target.observe_at_date(obstime + delta_time)

            if sun_besttime.ra >= target_besttime.ra:
                delta_phi = sun_besttime.ra - target_besttime.ra
                delta_time_1 = factor_sidyear * (180 * u.deg - delta_phi)
            else:
                delta_phi = target_besttime.ra - sun_besttime.ra
                delta_time_1 = factor_sidyear * (delta_phi - 180 * u.deg)

            return cls.calculate_culmination(target, location, obstime + delta_time + delta_time_1)

    def plot_altaz(self, target, location, obstime, sun, moon, interval=15*u.min):
        warnings.warn("`location` parameter will be removed in a future version, using the internal instance of "
                      "`Location` class.", DeprecationWarning)
        warnings.warn("`sun` and `moon` parameters will be removed in a future version, by using the internal "
                      "instances of `Sun` and `Moon` classes.",
                      DeprecationWarning)

        step_mjd = interval / (1 * u.d).to(interval.unit)

        # If obstime is before the day's sunrise, code would calculate data for the upcoming sunset. This allows the
        # correct data_night to be displayed.
        if obstime < self.calculate_rise_time(self.sun, location, Time(int(obstime.mjd), format='mjd')):
            obstime -= 1 * u.day

        sunset = self.calculate_set_time(sun, location, obstime)
        if int(sunset.mjd) > int(obstime.mjd):
            sunset = self.calculate_set_time(sun, location, obstime - 1 * u.day)
        sunrise = self.calculate_rise_time(sun, location, sunset)

        time_limits = [self.__find_edge_time(sunset - interval, 'lower'),
                       self.__find_edge_time(sunrise + interval, 'upper')]

        step_times = np.arange(time_limits[0].mjd, time_limits[1].mjd, step_mjd)
        times = Time(step_times, format='mjd')

        alt = [self.calculate_alt(target, location, t) for t in times] * u.deg
        az = [self.calculate_az(target, location, t) for t in times] * u.deg

        alt_moon = [self.calculate_alt(moon, location, t) for t in times] * u.deg
        az_moon = [self.calculate_az(moon, location, t) for t in times] * u.deg
        phase_moon = [moon.calculate_moon_phase(self.sun, t) for t in times]  #

        sun_naut_twilights_0 = self.calculate_twilight(sun, location, obstime, twilight='nautical')[1:]
        sun_astr_twilights_0 = self.calculate_twilight(sun, location, obstime, twilight='astronomical')[1:]

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(target.name + " between {} and {}".format(obstime.iso[:-13], times[-1].iso[:-13]))

        ## subplot1
        # plot dell'altezza sull'orizzonte
        ax1 = plt.subplot2grid((3, 1), (0, 0))
        ax1.grid()
        ax1.scatter(step_times, alt_moon, c=phase_moon, norm=Normalize(-1, 1), cmap='gray',
                    marker='o', s=40, edgecolors='black', linewidths=.8)
        ax1.scatter(step_times, alt, marker='*', s=10, color='black')

        # linee dei crepuscoli. I label sono definiti dopo.
        ax1.vlines(sun_naut_twilights_0[0], -90, 90, linestyles='dashed', colors='b')
        ax1.vlines(sun_naut_twilights_0[1], -90, 90, linestyles='dashed', colors='b')
        ax1.vlines(sun_astr_twilights_0[0], -90, 90, linestyles='solid', colors='b')
        ax1.vlines(sun_astr_twilights_0[1], -90, 90, linestyles='solid', colors='b')

        # plot dell'alba e del tramonto del Sole
        ax1.vlines(sunset, -90, 90, linestyles='dotted', colors='b', linewidth=1)
        ax1.vlines(sunrise, -90, 90, linestyles='dotted', colors='b', linewidth=1)

        # definiti i limiti come tra il crepuscolo astronomico e lo zenit.
        ax1.set_ylim(-18, 90)

        # gestione dei tick delle altezze sull'orizzonte.
        ax1.yaxis.set_ticks(np.array([-18, 0, 30, 60, 90]))
        ax1.hlines(-12, min(step_times), max(step_times), linestyles='dashed', colors='b', linewidth=1)
        ax1.hlines(0, min(step_times), max(step_times), linestyles='dotted', colors='b', linewidth=1)

        ax1.xaxis.set_major_locator(plt.MaxNLocator(len(times) - 1))
        ax1.xaxis.set_ticks(times[0::4])
        ax1.set_xticklabels([])
        ax1.set_xlabel('')
        ax1.set_ylabel('Alt [deg]')

        # aggiunta dei tick per le airmass
        ax1_ = ax1.twinx()
        ax1_.set_ylim(ax1.get_ylim())
        airmasses = np.concatenate((np.arange(1., 1.1, 0.05),
                                    np.arange(1.1, 1.3, 0.1),
                                    np.arange(1.3, 1.6, 0.2),
                                    np.arange(2, 3.5, 1)))
        airmasses_ticks = ['{:1.2f}'.format(tick) for tick in airmasses]
        alt_ticks = 90 - np.rad2deg(np.arccos(1 / airmasses))
        ax1_.yaxis.set_major_locator(mticker.FixedLocator(alt_ticks * u.deg))
        ax1_.set_yticks(alt_ticks)
        ax1_.set_yticklabels(airmasses_ticks)

        ax1_.set_ylabel('Airmass')

        ## subplot2
        # plot dell'azimut
        ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
        ax2.grid()
        moon_scatter = ax2.scatter(step_times, az_moon, c=phase_moon, norm=Normalize(-1, 1), cmap='gray',
                                   marker='o', s=40, edgecolors='black', linewidths=.8,
                                   label=r'Moon (color $\rightarrow$ phase)')
        cbar = fig.colorbar(moon_scatter, orientation='horizontal', pad=0.18)
        cbar.set_ticks([1, 0, -1])
        cbar.set_ticklabels(['New Moon', 'Quarter', 'Full Moon'])

        tgt_scatter = ax2.scatter(step_times, az, marker='*', s=10, color='black', label='{}'.format(target.name))

        # linee dei crepuscoli. I label sono definiti dopo.
        naut_twi = ax2.vlines(sun_naut_twilights_0[0], 0, 360, linestyles='dashed', colors='b', label='Naut. twilight')
        ax2.vlines(sun_naut_twilights_0[1], 0, 360, linestyles='dashed', colors='b')
        astr_twi = ax2.vlines(sun_astr_twilights_0[0], 0, 360, linestyles='solid', colors='b', label='Astr. twilight')
        ax2.vlines(sun_astr_twilights_0[1], 0, 360, linestyles='solid', colors='b')

        # plot dell'alba e del tramonto del Sole
        sun_riseset = ax2.vlines(sunset, 0, 360, linestyles='dotted', colors='b', linewidth=1, label='Sun set/rise')
        ax2.vlines(sunrise, 0, 360, linestyles='dotted', colors='b', linewidth=1)

        ax2.legend(handles=[tgt_scatter, moon_scatter, sun_riseset, naut_twi, astr_twi],
                   loc='best')

        # gestione dei tick degli azimut.
        ax2.set_ylim(0, 360)
        ax2.yaxis.set_ticks(np.linspace(0, 360, 13))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(len(step_times) - 1))
        ax2.xaxis.set_ticks(step_times[0::4])
        times_labs = times[0::4]
        labels = [l.iso[11:16] for l in times_labs]
        ax2.set_xticklabels(labels)
        ax2.xaxis.set_tick_params(rotation=80)
        ax2.set_ylabel('Az [deg]')

        # aggiunta dei tick per i punti cardinali.
        ax2_ = ax2.twinx()
        ax2_.set_ylim(ax2.get_ylim())
        ticks_loc = ax2.get_yticks().tolist()
        ax2_.yaxis.set_major_locator(mticker.FixedLocator([0, 90, 180, 270, 360]))
        ax2_ticks = []
        for tick in ticks_loc:
            if tick == 0 or tick == 360:
                ax2_ticks.append('N')
            elif tick == 90:
                ax2_ticks.append('E')
            elif tick == 180:
                ax2_ticks.append('S')
            elif tick == 270:
                ax2_ticks.append('W')
        ax2_.set_yticklabels(ax2_ticks)

        ax1.set_xlim(min(step_times), max(step_times))
        ax2.set_xlim(min(step_times), max(step_times))

        fig.tight_layout(h_pad=0.5)
        fig.show()

    @staticmethod
    def __find_edge_time(time, bound):
        str_time = time.iso
        mins = int(str_time[14:16]) + float(str_time[17:]) / 60

        if bound == 'lower':
            if 0 < mins <= 15:
                new_str = str_time[:14] + '00:00.000'
            elif 15 < mins <= 30:
                new_str = str_time[:14] + '15:00.000'
            elif 30 < mins <= 45:
                new_str = str_time[:14] + '30:00.000'
            elif 45 < mins <= (60 - 1 / 3600):
                new_str = str_time[:14] + '45:00.000'
        elif bound == 'upper':
            if 0 <= mins < 15:
                new_str = str_time[:14] + '15:00.000'
            elif 15 <= mins < 30:
                new_str = str_time[:14] + '30:00.000'
            elif 30 <= mins < 45:
                new_str = str_time[:14] + '45:00.000'
            elif 45 <= mins < (60 - 0.001 / 3600):
                hour = int(str_time[11:13])

                if hour == 23:
                    new_time = Time(np.ceil(time.mjd), format='mjd')
                    new_time.format = 'iso'
                    return new_time
                else:
                    new_str = str_time[:11] + '{:02d}:00:00.000'.format(int(str_time[11:13]) + 1)

        return Time(new_str, format='iso', scale='utc')
