from src import location as loc
from src import time as time
from src import skylocation as skyloc

from src import open_loc_file as olf

import sys


def main(argv):
    prompt = False

    if len(argv) <= 1:
        loc_geo = input("\tCoordinates of observatory:\t").upper()
        obstime = input("\tTime of observation:\t").upper()
        tgt = input("\tPosition of target:\t").upper()

        prompt = True
    else:
        loc_geo, obstime, tgt = olf(argv[1], argv[2])

    location = loc.Location(locstring=loc_geo)
    obsdate = time.Time(obstime, format='isot')
    target = skyloc.SkyLocation(locstring=tgt, obstime=obsdate)


if __name__ == '__main__':
    main(sys.argv)
