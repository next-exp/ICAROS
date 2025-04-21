import numpy as np
from invisible_cities.core.core_functions  import in_range
from invisible_cities.reco.corrections     import ASectorMap



class AbortingMapCreation(Exception):
    pass


def check_if_values_in_interval(values          : np.array,
                                low_lim         : float   ,
                                up_lim          : float   ,
                                raising_message : str = ''
                                )->None:
    """
    Raises exception, aborting kr map computation, if input
    values are not all inside the interval (low_lim, up_lim).
    Parameters
    ----------
    values : np.array
        Input array to check.
    low_lim: float
        Lower limit of the interval.
    up_lim: float
        Upper limit of the interval.
    raising_message: string
        Message to print if exception raises.
    Returns
    ----------
        None if values are in the interval. Otherwise, it raises an exception.
    """
    if in_range(values, low_lim, up_lim).all():
        return;
    else:
        raise AbortingMapCreation(raising_message)

def get_core(nbins,rmax, rfid):
    x        = np.linspace(-rmax, rmax, nbins)
    xx, yy   = np.meshgrid(x, x)
    r        = np.sqrt(xx**2 + yy**2)
    bin_size = 2. * rmax / (nbins -1)
    mask     = in_range(r, 0, rfid + bin_size)
    return mask

def check_failed_fits(maps      : ASectorMap,
                      maxFailed : float     ,
                      nbins     : int       ,
                      rmax      : float     ,
                      rfid      : float     ):
    inner_core_mask    = get_core(nbins=nbins, rmax=rmax, rfid=rfid)
    map_values_in_core = maps.lt.values[inner_core_mask]
    numFailed = np.count_nonzero(np.isnan(map_values_in_core))
    if numFailed > maxFailed:
        message = "Number of failed fits ({0}) ".format(numFailed)
        message += "exceeds max. allowed ({0}).".format(maxFailed)
        raise AbortingMapCreation(message)
    else:
        print("    Number of failing fits: {0}".format(numFailed))
        return
