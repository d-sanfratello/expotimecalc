import src
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

    hour_steps = np.arange(0, 25, step=0.5) * u.hour
    obstimes = np.empty(len(hour_steps), dtype=Time)

    for _ in range(len(hour_steps)):
        obstimes[_] = obstime_base + hour_steps[_]

    return obstimes


def plot_altaz_onday(targets, location, obstimes):
    times = Time([t.jd for t in obstimes], format='jd')
    times.format = 'iso'

    for key in targets.keys():
        target = targets[key]

        alt = [Observation.calculate_alt(target, location, t).deg for t in obstimes] * u.deg
        az = [Observation.calculate_az(target, location, t).deg for t in obstimes] * u.deg

        fig = plt.figure()

        fig.suptitle(targets[key].name + " alla data del {}".format(obstimes[0].iso[:-7]))

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

    plt.show()


if __name__ == "__main__":
    location = Location(locstring='43.561667 10.589164')

    targets = {'iUMa': SkyLocation(locstring="8h59m11.89s 48d2m27.6s", name='iota Ursa Major'),
               'aLyr': SkyLocation(locstring="18h36m56.51s 38d47m8.5s", name='Vega'),
               'aCar': SkyLocation(locstring="6h23m57.16s -52d51m43.8s", name="Canopo")}

    obstimes = init_times()

    plot_altaz_onday(targets, location, obstimes)
