# import src
from src.location import Location
from src.time import Time
from src.observation import Observation
from src.skylocation import SkyLocation

import numpy as np
from astropy import units as u

Pisa = Location(locstring='43.561667 10.589164')

targets = {'wPsc': SkyLocation(locstring="23h59m18.91s 6d51m45.5s", name='omega Pisces'),
           'bAnd': SkyLocation(locstring="1h9m42.41s 35d37m11.2s", name='Mirach'),
           'aPsc': SkyLocation(locstring="2h2m2.88s 2d45m49.3s", name='alpha Pisces'),
           'bPer': SkyLocation(locstring="3h8m10.14s 40d57m20.3s", name='Algol'),
           'lTau': SkyLocation(locstring="4h0m40.81s 12d29m25.0s", name='lambda Tauris'),
           'eAur': SkyLocation(locstring="5h1m58.14s 43d49m23.9s", name='epsilon Aurigae')}#,
           # 'aOri': SkyLocation(locstring='5h55m10.30536s 7d24m25.4304s', name='Betelgeuse'),
           # 'aCMa': SkyLocation(locstring='6h45m8.01s -16d43m25.1s', name='Sirius'),
           # 'aCmi': SkyLocation(locstring="7h39m17.09s 5d13m7.9s", name='Procione'),
           # 'iUMa': SkyLocation(locstring="8h59m11.89s 48d2m27.6s", name='iota Ursa Major'),
           # 'aLeo': SkyLocation(locstring="10h8m21.94s 11d58m3.1s", name='Regolo'),
           # 'bLeo': SkyLocation(locstring="11h2m19.76s 20d10m48.2s", name='b Leonis'),
           # 'BLeo': SkyLocation(locstring="11h49m2.86s 14d34m16.8s", name='beta Leonis'),
           # 'a2CVn': SkyLocation(locstring="12h56m1.43s 38d19m7.4s", name='Cor Caroli'),
           # 'aBoo': SkyLocation(locstring="14h15m38.22s 19d10m9.9s", name='Arturo'),
           # 'sLib': SkyLocation(locstring="15h4m4.11s -25d16m56.0s", name='sigma Librae'),
           # 'dOph': SkyLocation(locstring="16h14m20.7s -3d41m41.4s", name='delta Ophiuchi'),
           # 'eHer': SkyLocation(locstring="17h0m17.31s 30d55m35.9s", name='epsilon Herculis'),
           # 'nOph': SkyLocation(locstring="17h59m1.59s -9d46m27.6s", name='nu Ophiuchi'),
           # 'aLyr': SkyLocation(locstring="18h36m56.51s 38d47m8.5s", name='Vega'),
           # 'aAql': SkyLocation(locstring="19h50m47.79s 8d52m14.2s", name='Altair'),
           # 'aCyg': SkyLocation(locstring="20h41m25.92s 45d16m49.3s", name='Deneb'),
           # 'aAqr': SkyLocation(locstring="22h5m47.7s -0d19m11.7s", name='alpha Aquaris'),
           # 'aPsA': SkyLocation(locstring="22h57m39.56s -29d37m24.3s", name='Fomalhaut')}

obstime_base = Time('2021-03-08 12:00:00')

hour_steps = np.arange(0, 25, step=6) * u.hour
obstimes = np.empty(len(hour_steps), dtype=Time)

if __name__ == "__main__":
    for _ in range(len(hour_steps)):
        obstimes[_] = obstime_base + hour_steps[_]

    for obstime in obstimes:
        for target in targets.values():
            obs = Observation(Pisa, obstime, target)
            print(obs.target.name, obs.obstime)
