import numpy as np

from astropy import units as u
from astropy.constants import au

from ...time import Time
from .. import SkyLocation

from .. import Equinox2000

from .. import errmsg


class Planet(SkyLocation):
    def __init__(self, ra, dec, distance, obstime, name=None, epoch='J2000'):
        super(Planet, self).__init__(locstring=None, ra=ra, dec=dec, distance=distance,
                                     obstime=obstime, ra_unit='deg', dec_unit='deg', epoch=epoch, name=name)

        self.at_date(obstime)

    def observe_at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        # # Il Sole viene prima convertito in coordinate eclittiche (più per forma che per utilità, dato che ha coordinate
        # # cartesiane (0,0,0)) e ruotato usando l'angolo calcolato con il metodo `Sun.sidereal_year_rotation`. Viene,
        # # quindi, applicata la precessione e, infine, viene riportato in coordinate equatoriali.
        # vector_obstime = self.vector_epoch.rotate_inv('x', self.axial_tilt(obstime), copy=True) \
        #     .rotate('z', self.sidereal_year_rotation(obstime), copy=True) \
        #     .rotate('z', self.equinox_prec(self.obstime), copy=True) \
        #     .rotate('x', self.axial_tilt(obstime), copy=True)

    def at_date(self, obstime):
        if not isinstance(obstime, Time):
            raise TypeError(errmsg.notTwoTypesError.format('obstime', 'src.time.Time', 'astropy.time.Time'))

        self.obstime = obstime
        self.vector_obstime = self.observe_at_date(obstime)
        self.ra = self.vector_obstime.ra
        self.dec = self.vector_obstime.dec
