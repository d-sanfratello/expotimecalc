# import numpy as np

# from astropy import units as u


def str2dms(string):
    deg, minsec = string.lower().split('d')
    mins, sec = minsec.lower().split('m')
    sec = sec[:-1]

    deg = float(deg)
    mins = float(mins) / 60
    sec = float(sec) / 3600

    return deg + mins + sec


def hms2dms(hms):
    if isinstance(hms, str):
        hour, minsec = hms.lower().split('h')
        mins, sec = minsec.lower().split('m')
        sec = sec[:-1]

        hour = float(hour)
        mins = float(mins) / 60
        sec = float(sec) / 3600

        return (hour + mins + sec) * 15

    elif isinstance(hms, (int, float)):
        return hms * 15
