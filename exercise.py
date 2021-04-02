import src
from src.location import Location
from src.time import Time
from src.observation import Observation
from src.skylocation import SkyLocation
from src.skylocation.sun import Sun
from src.skylocation.moon import Moon

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from astropy import units as u

from astropy.visualization import time_support
from astropy.visualization import quantity_support
time_support(scale='utc', format='iso', simplify=True)
quantity_support()


def init_times(time=None):
    if time is None:
        obstime_base = Time('2021-04-02 17:00:00')
        hour_steps = np.arange(0, 14, step=0.25) * u.hour
    else:
        obstime_base = time
        hour_steps = np.arange(-7, 7, step=0.25) * u.hour

    obstimes = np.empty(len(hour_steps), dtype=Time)
    for _ in range(len(hour_steps)):
        obstimes[_] = obstime_base + hour_steps[_]

    return obstimes


def plot_altaz_onday(targets, location, obstimes):
    times = Time([t.mjd for t in obstimes], format='mjd')

    for key in targets.keys():
        target = targets[key]

        alt = [Observation.calculate_alt(target, location, t).deg for t in obstimes] * u.deg
        az = [Observation.calculate_az(target, location, t).deg for t in obstimes] * u.deg

        sun = Sun(obstimes[0])
        moon = Moon(obstimes[0])

        alt_moon = [Observation.calculate_alt(moon, location, t).deg for t in obstimes] * u.deg
        az_moon = [Observation.calculate_az(moon, location, t).deg for t in obstimes] * u.deg

        sun_naut_twi_0 = Observation.calculate_twilight(sun, location, obstimes[0], twilight='nautical')[1:]
        sun_astr_twi_0 = Observation.calculate_twilight(sun, location, obstimes[0], twilight='astronomical')[1:]

        sun_set = Observation.calculate_set_time(sun, location, obstimes[0] - 6 * u.h)
        sun_rise = Observation.calculate_rise_time(sun, location, obstimes[-1])

        fig = plt.figure(figsize=(8, 8))

        fig.suptitle(targets[key].name + " alla data del {}".format(obstimes[0].iso[:-7]))

        # subplot1
        ax1 = plt.subplot2grid((3, 1), (0, 0))
        ax1.grid()
        ax1.plot(times, alt, 'k-')
        ax1.plot(times, alt_moon, 'k-.')

        plt.vlines(sun_naut_twi_0[0], -90, 90, linestyles='dashed', colors='b')
        plt.vlines(sun_naut_twi_0[1], -90, 90, linestyles='dashed', colors='b')
        plt.vlines(sun_astr_twi_0[0], -90, 90, linestyles='solid', colors='b')
        plt.vlines(sun_astr_twi_0[1], -90, 90, linestyles='solid', colors='b')

        plt.vlines(sun_set, -90, 90, linestyles='dotted', colors='b', linewidth=1)
        plt.vlines(sun_rise, -90, 90, linestyles='dotted', colors='b', linewidth=1)

        ax1.set_ylim(-18, 90)

        ax1.yaxis.set_ticks(np.array([-18, 0, 30, 60, 90]))
        ax1.hlines(-12, min(times), max(times), linestyles='dashed', colors='b', linewidth=1)
        ax1.hlines(0, min(times), max(times), linestyles='dotted', colors='b', linewidth=1)

        ax1.xaxis.set_major_locator(plt.MaxNLocator(len(times) - 1))
        ax1.xaxis.set_ticks(times[0::5])
        ax1.set_xticklabels([])
        ax1.set_xlabel('')
        ax1.set_ylabel('Alt [deg]')

        ax1_ = ax1.twinx()
        ax1_.set_ylim(ax1.get_ylim())
        ticks_loc = ax1.get_yticks().tolist()
        ax1_.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc[2:]))
        ax1_ticks = []
        for tick in ax1.get_yticks():
            if tick > 0:
                new_tick = 1 / np.cos(np.deg2rad(90 - tick))
                ax1_ticks.append('{:1.2f}'.format(new_tick))
        ax1_.set_yticklabels(ax1_ticks)
        ax1_.set_ylabel('Airmass')

        # subplot2
        ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
        ax2.grid()
        ax2.plot(times, az, 'k-')
        ax2.plot(times, az_moon, 'k-.',
                 label='Moon - phase = {:.2}$\\rightarrow${:.2}'.format(moon.calculate_moon_phase(sun, times[0]),
                                                                        moon.calculate_moon_phase(sun, times[-1])))

        plt.vlines(sun_naut_twi_0[0], 0, 360, linestyles='dashed', colors='b', label='Naut. twilight')
        plt.vlines(sun_naut_twi_0[1], 0, 360, linestyles='dashed', colors='b')
        plt.vlines(sun_astr_twi_0[0], 0, 360, linestyles='solid', colors='b', label='Astr. twilight')
        plt.vlines(sun_astr_twi_0[1], 0, 360, linestyles='solid', colors='b')

        plt.vlines(sun_set, 0, 360, linestyles='dotted', colors='b', linewidth=1, label='Sun set/rise')
        plt.vlines(sun_rise, 0, 360, linestyles='dotted', colors='b', linewidth=1)

        ax2.legend(loc='best')

        ax2.set_ylim(0, 360)
        ax2.yaxis.set_ticks(np.linspace(0, 360, 13))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(len(times) - 1))
        ax2.xaxis.set_ticks(times[0::5])
        ax2.xaxis.set_tick_params(rotation=80)
        ax2.set_ylabel('Az [deg]')

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

        ax1.set_xlim(min(times), max(times))
        ax2.set_xlim(min(times), max(times))

        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    location = Location(locstring='43.721045 10.407737')

    targets = {'iUMa': SkyLocation(locstring="8h59m12.45362s 48d2m30.5741s", name='$\iota$UMa'),
               'aLyr': SkyLocation(locstring="18h36m56.33635s 38d47m1.2802s", name='Vega'),
               'aOri': SkyLocation(locstring='5h55m10.30536s 7d24m25.4304s', name='Betelgeuse'),
               'aCar': SkyLocation(locstring="6h23m57.10988s -52d41m44.3810s", name="Canopus")}

    obstimes = init_times()

    plot_altaz_onday(targets, location, obstimes)

    best_iUma = Observation.calculate_best_day(targets['iUMa'], location, obstimes[0])
    best_aLyr = Observation.calculate_best_day(targets['aLyr'], location, obstimes[0])
    best_aOri = Observation.calculate_best_day(targets['aOri'], location, obstimes[0])

    print('best {}:{}'.format(targets['iUMa'].name, best_iUma))
    print('best {}:{}'.format(targets['aLyr'].name, best_aLyr))
    print('best {}:{}'.format(targets['aOri'].name, best_aOri))
    
    sun = Sun(best_iUma)
    print('quality {}:{}'.format(targets['iUMa'].name,
                                 Observation.estimate_quality(targets['iUMa'], location, best_iUma, sun,
                                                              parameter=1.5, interval=2*u.hour, par_type='airmass')))
    sun = Sun(best_aLyr)
    print('quality {}:{}'.format(targets['aLyr'].name,
                                 Observation.estimate_quality(targets['aLyr'], location, best_aLyr, sun,
                                                              parameter=1.5, interval=2 * u.hour, par_type='airmass')))
    sun = Sun(best_aOri)
    print('quality {}:{}'.format(targets['aOri'].name,
                                 Observation.estimate_quality(targets['aOri'], location, best_aOri, sun,
                                                              parameter=1.5, interval=2 * u.hour, par_type='airmass')))

    obstimes_iUma = init_times(best_iUma)
    obstimes_aLyr = init_times(best_aLyr)
    obstimes_aOri = init_times(best_aOri)

    plot_altaz_onday({'1': targets['iUMa']}, location, obstimes_iUma)
    plot_altaz_onday({'2': targets['aLyr']}, location, obstimes_aLyr)
    plot_altaz_onday({'3': targets['aOri']}, location, obstimes_aOri)

    from src import Omegasidmoon
    Tsidmoon = 1 / Omegasidmoon * 2 * np.pi * u.rad

    obstimes_aOri = init_times(best_aOri - Tsidmoon / 4 - 3 * u.hour)
    print('quality {}:{}'.format(targets['aOri'].name,
                                 Observation.estimate_quality(targets['aOri'], location,
                                                              obstimes_aOri[len(obstimes_aOri) // 2], sun,
                                                              parameter=1.5, interval=2 * u.hour, par_type='airmass')))
    print('culm: {}'.format(Observation.calculate_culmination(targets['aOri'], location,
                                                              obstimes_aOri[len(obstimes_aOri) // 2])))
    plot_altaz_onday({'3': targets['aOri']}, location, obstimes_aOri)

    obstimes_aOri = init_times(best_aOri + Tsidmoon / 4 + 5 * u.hour)
    print('quality {}:{}'.format(targets['aOri'].name,
                                 Observation.estimate_quality(targets['aOri'], location,
                                                              obstimes_aOri[len(obstimes_aOri) // 2], sun,
                                                              parameter=1.5, interval=2 * u.hour, par_type='airmass')))
    print('culm: {}'.format(Observation.calculate_culmination(targets['aOri'], location,
                                                              obstimes_aOri[0])))
    plot_altaz_onday({'3': targets['aOri']}, location, obstimes_aOri)

    obstimes_aOri = init_times(best_aOri - Tsidmoon / 2 - 8 * u.hour)
    print('quality {}:{}'.format(targets['aOri'].name,
                                 Observation.estimate_quality(targets['aOri'], location,
                                                              obstimes_aOri[len(obstimes_aOri) // 2], sun,
                                                              parameter=1.5, interval=2 * u.hour, par_type='airmass')))
    print('culm: {}'.format(Observation.calculate_culmination(targets['aOri'], location,
                                                              obstimes_aOri[len(obstimes_aOri) // 2])))
    plot_altaz_onday({'3': targets['aOri']}, location, obstimes_aOri)

    obstimes_aOri = init_times(best_aOri + Tsidmoon / 2 - 17 * u.hour)
    print('quality {}:{}'.format(targets['aOri'].name,
                                 Observation.estimate_quality(targets['aOri'], location,
                                                              obstimes_aOri[len(obstimes_aOri) // 2], sun,
                                                              parameter=1.5, interval=2 * u.hour, par_type='airmass')))
    print('culm: {}'.format(Observation.calculate_culmination(targets['aOri'], location,
                                                              obstimes_aOri[len(obstimes_aOri) // 2])))
    plot_altaz_onday({'3': targets['aOri']}, location, obstimes_aOri)
