import logging
import numpy as np
import pykep as pk  # https://ui.adsabs.harvard.edu/abs/2015arXiv151100821I/abstract

from astropy import units as u
from astropy import constants as cts
from astropy.coordinates import Angle
from astropy.coordinates.angles import Longitude

from .. import Versor
from ...time import Time
from .. import SkyLocation

from .. import errmsg
from .. import logger


# noinspection PyUnresolvedReferences
class Planet(SkyLocation):
    __private_arguments = ['is_earth']

    def __init__(self, obstime, *, name=None, epoch='J2000', **kwargs):

        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if name is None:
            raise ValueError(errmsg.mustDeclareName)

        if not bool(kwargs):
            self.__is_earth = False
        else:
            k = kwargs.keys()
            for _ in k:
                if _ not in self.__private_arguments:
                    raise KeyError(errmsg.keyError)
                else:
                    self.__is_earth = kwargs[_]
        self.__logger = logging.getLogger('src.skylocation.planets.Planet')
        self.__logger.setLevel(logger.getEffectiveLevel())

        self.__logger.debug('Initializing `Planet` class')
        self.__logger.info(f'instance of {name}')
        self.__logger.debug(f'{name} is earth: {self.__is_earth}')

        self.__mass_sun = cts.M_sun
        self.__grav_const = cts.G * self.__mass_sun / (4 * np.pi ** 2)

        self.__semimaj = None
        self.__ecc = None
        self.__argument_pericenter = None
        self.__mean_anomaly = None
        self.__eccentric_anomaly = None
        self.__inclination = None
        self.__longitude_an = None

        self.at_date(obstime)

        super(Planet, self).__init__(locstring=None, ra=self.vector_obstime.ra, dec=self.vector_obstime.dec,
                                     distance=self.distance_from_sun, obstime=obstime,
                                     ra_unit='deg', dec_unit='deg', epoch=epoch, name=name)

    def observe_at_date(self, obstime, return_complete=False):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        if not isinstance(return_complete, bool):
            raise TypeError(errmsg.notTypeError.format('return_complete', 'bool'))
        self.__logger.debug(f'`obstime`: {obstime} call.')

        epoch = pk.epoch_from_string(obstime.iso)

        __semimaj, __ecc, __inclination, __ra_an, __peri_arg, __mean_anomaly = self.ephemeris.osculating_elements(epoch)
        __semimaj *= u.m
        semimaj = __semimaj.to(u.AU)

        __inclination *= u.rad
        inclination = Angle(__inclination.to(u.deg))

        __ra_an *= u.rad
        longitude_an = Longitude(__ra_an.to(u.deg))

        __peri_arg *= u.rad
        argument_pericenter = Angle(__peri_arg.to(u.deg))
        eccentricity = np.float64(__ecc)

        __mean_anomaly *= u.rad
        mean_anomaly = Angle(__mean_anomaly.to(u.deg))
        eccentric_anomaly = self.__approx_ecc_anomaly(mean_anomaly=mean_anomaly, eccentricity=eccentricity)
        true_anomaly = self.__true_anomaly(eccentric_anomaly=eccentric_anomaly, eccentricity=eccentricity)

        distance_from_sun = self.__distance_from_center(semimaj=semimaj,
                                                        eccentric_anomaly=eccentric_anomaly,
                                                        eccentricity=eccentricity)

        if self.__is_earth:
            self.__logger.info(f'It is Earth.')
            self.__logger.debug(f'reference observer is the Sun')
        else:
            self.__logger.info(f'It is NOT Earth. Defining an `Earth` instance as reference')
            from .earth import Earth
            reference_observer = Earth(obstime=obstime).vector_obstime
            self.__logger.debug(f'Earth reference defined as {reference_observer.vsr}, {reference_observer.radius}')

        self.__logger.info(f'Rotating body with osculating parameters from ephemeris')
        vector_obstime = Versor(vector=np.array([1, 0, 0]) * distance_from_sun) \
            .rotate(axis='z', angle=true_anomaly, copy=True) \
            .rotate(axis='z', angle=argument_pericenter, copy=True) \
            .rotate(axis='x', angle=inclination, copy=True) \
            .rotate(axis='z', angle=longitude_an, copy=True)

        if not self.__is_earth:
            vector_obstime = (vector_obstime - reference_observer) \
                .rotate(axis='z', angle=self.equinox_prec(obstime), copy=True) \
                .rotate(axis='x', angle=self.axial_tilt(obstime), copy=True)  # check for rotate or rotate_inv

        self.__logger.info(f'Body position estimated at {vector_obstime.ra.hms}, {vector_obstime.dec.deg}, '
                           f'{vector_obstime.radius}. Returning `vector_obstime` from `observe_at_date`.')
        if return_complete:
            orb_pars = {'semimaj': semimaj,
                        'inclination': inclination,
                        'longitude an': longitude_an,
                        'argument pericenter': argument_pericenter,
                        'eccentricity': eccentricity,
                        'mean anomaly': mean_anomaly,
                        'eccentric anomaly': eccentric_anomaly,
                        'true anomaly': true_anomaly,
                        'distance from sun': distance_from_sun}
            return vector_obstime, orb_pars
        else:
            return vector_obstime

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))
        self.__logger.debug(f'obstime: {obstime}')

        self.obstime = obstime

        self.vector_obstime, orb_pars = self.observe_at_date(obstime, return_complete=True)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec
        self.__logger.debug(f'ra-dec set by `at_date` method at {self.ra.hms}, {self.dec.deg}')

        self.__semimaj = orb_pars['semimaj']
        self.__inclination = orb_pars['inclination']
        self.__longitude_an = orb_pars['longitude an']
        self.__argument_pericenter = orb_pars['argument pericenter']
        self.__ecc = orb_pars['eccentricity']
        self.__mean_anomaly = orb_pars['mean anomaly']
        self.__eccentric_anomaly = orb_pars['eccentric anomaly']

    def velocity(self, reference, obstime):
        from .sun import Sun
        from .earth import Earth
        if not isinstance(reference, (Earth, Sun)):
            raise TypeError(errmsg.notTwoTypesError.format('reference', 'Earth', 'planet.Sun'))
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        vector_obstime, orb_pars = self.observe_at_date(obstime, return_complete=True)

        e = orb_pars['eccentricity']
        f = orb_pars['true anomaly']
        a = orb_pars['semimaj']
        r = orb_pars['distance from sun']
        argument_pericenter = orb_pars['argument pericenter']
        inclination = orb_pars['inclination']
        longitude_an = orb_pars['longitude an']

        radial_v_mod = e * np.sin(f) * np.sqrt(cts.G * (self.mass + self.mass_central) / (a * (1 - e**2)))
        tangen_v_mod = np.sqrt(cts.G * (self.mass + self.mass_central) * a * (1 - e**2)) / r

        radial_v_mod = radial_v_mod.to(u.km / u.s)
        tangen_v_mod = tangen_v_mod.to(u.km / u.s)

        vel_radial = Versor(vector=np.array([np.cos(f), np.sin(f), 0]) * radial_v_mod)
        vel_tangen = Versor(vector=np.array([-np.sin(f), np.cos(f), 0]) * tangen_v_mod)

        velocity = (vel_radial + vel_tangen) \
            .rotate(axis='z', angle=argument_pericenter, copy=True) \
            .rotate(axis='x', angle=inclination, copy=True) \
            .rotate(axis='z', angle=longitude_an, copy=True)

        if isinstance(reference, Planet):
            reference_obstime, ref_orb_pars = reference.observe_at_date(obstime, return_complete=True)

            e_ref = ref_orb_pars['eccentricity']
            f_ref = ref_orb_pars['true anomaly']
            a_ref = ref_orb_pars['semimaj']
            r_ref = ref_orb_pars['distance from sun']
            argument_pericenter_ref = ref_orb_pars['argument pericenter']
            inclination_ref = ref_orb_pars['inclination']
            longitude_an_ref = ref_orb_pars['longitude an']

            radial_v_mod_ref = e_ref * np.sin(f_ref) * \
                np.sqrt(cts.G * (self.mass + self.mass_central) / (a_ref * (1 - e_ref ** 2)))
            tangen_v_mod_ref = np.sqrt(cts.G * (self.mass + self.mass_central) * a_ref * (1 - e_ref ** 2)) / r_ref

            radial_v_mod_ref = radial_v_mod_ref.to(u.km / u.s)
            tangen_v_mod_ref = tangen_v_mod_ref.to(u.km / u.s)

            vel_radial_ref = Versor(vector=np.array([np.cos(f), np.sin(f), 0]) * radial_v_mod_ref)
            vel_tangen_ref = Versor(vector=np.array([-np.sin(f), np.cos(f), 0]) * tangen_v_mod_ref)

            velocity_ref = (vel_radial_ref + vel_tangen_ref) \
                .rotate(axis='z', angle=argument_pericenter_ref, copy=True) \
                .rotate(axis='x', angle=inclination_ref, copy=True) \
                .rotate(axis='z', angle=longitude_an_ref, copy=True)

            rel_velocity = velocity - velocity_ref
            rel_velocity_mod = rel_velocity.vsr.dot(vector_obstime.vsr) * rel_velocity.radius

            return rel_velocity_mod
        else:
            return velocity

    def __approx_ecc_anomaly(self, mean_anomaly=None, eccentricity=None):
        if mean_anomaly is not None and eccentricity is None:
            raise ValueError(errmsg.notAllDeclared)
        elif mean_anomaly is None and eccentricity is not None:
            raise ValueError(errmsg.notAllDeclared)

        if mean_anomaly is None and eccentricity is None:
            mean = self.mean_anomaly.to(u.rad)
            ecc = self.eccentricity
        else:
            mean = mean_anomaly.to(u.rad)
            ecc = eccentricity
        # Method:
        #  1998aalg.book.....M (p196)

        ecc_an = mean
        ecc_an_next = mean - ecc * np.sin(ecc_an) * u.rad
        while abs(ecc_an_next.to(u.deg) - ecc_an.to(u.deg)) > (1e-4 * u.arcsec).to(u.deg):
            ecc_an = ecc_an_next
            ecc_an_next = mean - ecc * np.sin(ecc_an) * u.rad

        return Angle(ecc_an_next.to(u.deg))

    def __true_anomaly(self, eccentric_anomaly=None, eccentricity=None):
        if eccentric_anomaly is not None and eccentricity is None:
            raise ValueError(errmsg.notAllDeclared)
        elif eccentric_anomaly is None and eccentricity is not None:
            raise ValueError(errmsg.notAllDeclared)

        if eccentric_anomaly is None and eccentricity is None:
            eccentric_anomaly = self.eccentric_anomaly
            eccentricity = self.eccentricity

        true_an = (2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(eccentric_anomaly / 2)))
        return true_an.to(u.deg)

    def __distance_from_center(self, semimaj=None, eccentric_anomaly=None, eccentricity=None):
        if semimaj is not None and (eccentric_anomaly is None or eccentricity is None):
            raise ValueError(errmsg.notAllDeclared)
        elif eccentric_anomaly is not None and (semimaj is None or eccentricity is None):
            raise ValueError(errmsg.notAllDeclared)
        elif eccentricity is not None and (semimaj is None or eccentric_anomaly is None):
            raise ValueError(errmsg.notAllDeclared)

        if semimaj is None and eccentric_anomaly is None and eccentricity is None:
            semimaj = self.semimaj
            eccentric_anomaly = self.eccentric_anomaly
            eccentricity = self.eccentricity

        return semimaj * (1 - eccentricity * np.cos(eccentric_anomaly))

    @property
    def longitude_an(self):
        return self.__longitude_an

    @property
    def revolution_period(self):
        return np.sqrt(self.semimaj ** 3 / self.__grav_const).to(u.day)

    @property
    def semimaj(self):
        return self.__semimaj

    @property
    def eccentricity(self):
        return self.__ecc

    @property
    def semimin(self):
        return self.semimaj * np.sqrt(1 - self.eccentricity**2)

    @property
    def argument_pericenter(self):
        return self.__argument_pericenter

    @property
    def mean_anomaly(self):
        return self.__mean_anomaly

    @property
    def eccentric_anomaly(self):
        return self.__eccentric_anomaly

    @property
    def true_anomaly(self):
        return self.__true_anomaly()

    @property
    def inclination(self):
        return self.__inclination

    @property
    def peri_dist(self):
        return self.semimaj * (1 - self.eccentricity)

    @property
    def apo_dist(self):
        return self.semimaj * (1 + self.eccentricity)

    @property
    def distance_from_sun(self):
        return self.__distance_from_center()

    @property
    def mass(self):
        central_grav_par = self.ephemeris.mu_central_body
        self_grav_par = self.ephemeris.mu_self

        mass_ratio = self_grav_par / central_grav_par
        return mass_ratio * self.mass_central

    @property
    def mass_central(self):
        return self.__mass_sun


from .mercury import Mercury
from .venus import Venus
from .earth import Earth
from .moon import Moon
from .mars import Mars
from .jupiter import Jupiter
