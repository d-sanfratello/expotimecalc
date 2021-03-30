# import src
from src.location import Location
from src.time import Time
from src.observation import Observation
from src.skylocation import SkyLocation

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from astropy.visualization import time_support
from astropy.visualization import quantity_support
time_support(scale='utc', format='iso', simplify=True)
quantity_support()


def init_times():
    obstime_base = Time('2021-03-30 20:00:00')

    hour_steps = np.arange(0, 25, step=12) * u.hour
    obstimes = np.empty(len(hour_steps), dtype=Time)

    for _ in range(len(hour_steps)):
        obstimes[_] = obstime_base + hour_steps[_]

    return obstimes


def observe_tgt_day(key, obstimes):
    target = targets[key]

    obs = Observation(location, obstimes[0], target)


def plot_altaz_onday(obs, target, location, obstimes):

    alt = np.array([obs.calculate_alt(target, location, t) for t in obstimes])
    az = np.array([obs.calculate_az(target, location, t) for t in obstimes])

    times = np.array([t.tp_value('iso', subfmt='date_hm') for t in obstimes])

    fig = plt.figure()

    fig.suptitle(target.name + " alla data del {}".format(times[0].iso[:-6]))

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


if __name__ == "__main__":
    location = Location(locstring='43.561667 10.589164')

    targets = {'iUMa': SkyLocation(locstring="8h59m11.89s 48d2m27.6s", name='iota Ursa Major')}  # ,
    # 'aLyr': SkyLocation(locstring="18h36m56.51s 38d47m8.5s", name='Vega'),
    # 'aCar': SkyLocation(locstring="6h23m57.16s -52d51m43.8s", name="Canopo")}

    obstimes = init_times()

    for key in targets.keys():
        observe_tgt_day(key, obstimes)

        plot_altaz_onday(targets[key], location, obstimes)
