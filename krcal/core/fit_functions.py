"""Module fit_functions.
This module includes general fit functions

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""
import numpy as np
import warnings
import logging
from typing      import Tuple
from typing      import Callable

import invisible_cities.core    .fit_functions  as     fitf
import invisible_cities.database.load_db        as     DB
from   invisible_cities.core    .core_functions import in_range
from   invisible_cities.core    .core_functions import shift_to_bin_centers
from   invisible_cities.core    .stat_functions import poisson_sigma
from   invisible_cities.evm     .ic_containers  import FitFunction

from . kr_types  import Measurement


log = logging.getLogger(__name__)

_FIT_EXCEPTIONS = AssertionError, RuntimeError


def chi2f(f   : Callable,
          nfp : int,        # number of function parameters
          x   : np.array,
          y   : np.array,
          yu  : np.array)->float:
    """
    Computes the chi2 of a function f applied over array x and compared
    with array y with error yu

    """

    assert len(x) == len(y) == len(yu)
    fitx  = f(x)
    chi2_ = np.sum(((y - fitx) / yu)**2)

    if len(x) > nfp:
        return chi2_/(len(x)-nfp)
    else:
        warnings.warn('nof = 0 in chi2 calculation, return chi2 = {chi2_}', UserWarning)
        return chi2_



def chi2(f : FitFunction,
         x : np.array,
         y : np.array,
         sy: np.array)->float:
    """
    Computes the chi2 of a function f applied over array x and compared
    with array y with error yu. The object f is of type FitFunction.

    """
    return chi2f(f.fn, len(f.values), x, y, sy)


def gauss_seed(x, y, sigma_rel=0.05):
    """
    Estimate the seed for a gaussian fit to the input data.
    """
    y_max  = np.argmax(y) # highest bin
    x_max  = x[y_max]
    sigma  = sigma_rel * x_max
    amp    = y_max * (2 * np.pi)**0.5 * sigma * np.diff(x)[0]
    seed   = amp, x_max, sigma
    return seed


def expo_seed(x, y, eps=1e-12):
    """
    Estimate the seed for a exponential fit to the input data.
    """
    x, y  = zip(*sorted(zip(x, y)))
    const = y[0]
    slope = (x[-1] - x[0]) / np.log(y[-1] / (y[0] + eps))
    seed  = const, slope
    return seed


def relative_errors(values, errors, default=0, percentual=False):
    """
    Compute relative errors from input values with safety checks.
    If the relative error cannot be computed, a default value is
    used. The errors can be in percent if the `percentual`
    argument is True.
    """
    ok         = values != 0
    scale      = 100 if percentual else 1
    rel_e      = np.empty_like(values)
    rel_e[ ok] = errors[ok] / np.abs(values[ok]) * scale
    rel_e[~ok] = default
    return rel_e


def to_relative(data, *args, **kwargs):
    """
    Produce another Measurement instance with relative instead of
    the absolute ones.
    """
    return Measurement(data.value, relative_errors(*data, *args, **kwargs))


def quick_gauss_fit(data, bins):
    """
    Histogram input data and fit it to a gaussian with the parameters
    automatically estimated.
    """
    y, x  = np.histogram(data, bins)
    x     = shift_to_bin_centers(x)
    seed  = gauss_seed(x, y)
    f     = fitf.fit(fitf.gauss, x, y, seed)
    assert np.all(f.values != seed)
    return f


def fit_profile_1d_expo(xdata, ydata, nbins, *args, **kwargs):
    """
    Make a profile of the input data and fit it to an exponential
    function with the parameters automatically estimated.
    """
    x, y, yu     = fitf.profileX(xdata, ydata, nbins, *args, **kwargs)
    valid_points = yu > 0

    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]
    seed = expo_seed(x, y)
    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
    assert np.all(f.values != seed)
    return f


def fit_slices_1d_gauss(xdata, ydata, xbins, ybins,
                        min_entries   = 1e2,
                        ignore_errors = _FIT_EXCEPTIONS):
    """
    Slice the data in x, histogram each slice, fit it to a gaussian
    and return the relevant values.

    Parameters
    ----------
    xdata, ydata: array_likes
        Values of each coordinate.
    xbins: array_like
        The bins in the x coordinate.
    ybins: array_like
        The bins in the y coordinate for histograming the data.
    min_entries: int (optional)
        Minimum amount of entries to perform the fit.

    Returns
    -------
    mean: Measurement(np.ndarray, np.ndarray)
        Values of mean with errors.
    sigma: Measurement(np.ndarray, np.ndarray)
        Values of sigma with errors.
    chi2: np.ndarray
        Chi2 from each fit.
    valid: boolean np.ndarray
        Where the fit has been succesfull.
    """
    nbins  = np.size (xbins) - 1
    mean   = np.zeros(nbins)
    sigma  = np.zeros(nbins)
    meanu  = np.zeros(nbins)
    sigmau = np.zeros(nbins)
    chi2   = np.zeros(nbins)
    valid  = np.zeros(nbins, dtype=bool)

    for i in range(nbins):
        sel = in_range(xdata, *xbins[i:i + 2])
        if np.count_nonzero(sel) < min_entries: continue

        try:
            f = quick_gauss_fit(ydata[sel], ybins)
            mean  [i] = f.values[1]
            meanu [i] = f.errors[1]
            sigma [i] = f.values[2]
            sigmau[i] = f.errors[2]
            chi2  [i] = f.chi2
            valid [i] = True
        except Exception as exc:
            if not isinstance(exc, ignore_errors):
                raise
    return Measurement(mean, meanu), Measurement(sigma, sigmau), chi2, valid


def fit_slices_2d_gauss(xdata, ydata, zdata, xbins, ybins, zbins,
                        min_entries   = 1e2,
                        ignore_errors = _FIT_EXCEPTIONS):
    """
    Slice the data in x and y, histogram each slice, fit it to a gaussian
    and return the relevant values.

    Parameters
    ----------
    xdata, ydata, zdata: array_likes
        Values of each coordinate.
    xbins, ybins: array_likes
        The bins in the x and y coordinates.
    zbins: array_like
        The bins in the z coordinate for histograming the data.
    min_entries: int (optional)
        Minimum amount of entries to perform the fit.

    Returns
    -------
    mean: Measurement(np.ndarray, np.ndarray)
        Values of mean with errors.
    sigma: Measurement(np.ndarray, np.ndarray)
        Values of sigma with errors.
    chi2: np.ndarray
        Chi2 from each fit.
    valid: boolean np.ndarray
        Where the fit has been succesfull.
    """
    nbins_x = np.size (xbins) - 1
    nbins_y = np.size (ybins) - 1
    nbins   = nbins_x, nbins_y
    mean    = np.zeros(nbins)
    sigma   = np.zeros(nbins)
    meanu   = np.zeros(nbins)
    sigmau  = np.zeros(nbins)
    chi2    = np.zeros(nbins)
    valid   = np.zeros(nbins, dtype=bool)

    for i in range(nbins_x):
        sel_x = in_range(xdata, *xbins[i:i + 2])
        for j in range(nbins_y):
            sel_y = in_range(ydata, *ybins[j:j + 2])
            sel   = sel_x & sel_y
            if np.count_nonzero(sel) < min_entries: continue

            try:
                f = quick_gauss_fit(zdata[sel], zbins)
                mean  [i, j] = f.values[1]
                meanu [i, j] = f.errors[1]
                sigma [i, j] = f.values[2]
                sigmau[i, j] = f.errors[2]
                chi2  [i, j] = f.chi2
                valid [i, j] = True
            except Exception as exc:
                if not isinstance(exc, ignore_errors):
                    raise

    return Measurement(mean, meanu), Measurement(sigma, sigmau), chi2, valid


def fit_slices_2d_expo(xdata, ydata, zdata, tdata,
                       xbins, ybins, nbins_z, zrange=None,
                       min_entries   = 1e2,
                       ignore_errors = _FIT_EXCEPTIONS):
    """
    Slice the data in x and y, make the profile in z of t,
    fit it to a exponential and return the relevant values.

    Parameters
    ----------
    xdata, ydata, zdata, tdata: array_likes
        Values of each coordinate.
    xbins, ybins: array_like
        The bins in the x coordinate.
    nbins_z: int
        The number of bins in the z coordinate for the profile.
    zrange: length-2 tuple (optional)
        Fix the range in z. Default is computed from min and max
        of the input data.
    min_entries: int (optional)
        Minimum amount of entries to perform the fit.

    Returns
    -------
    const: Measurement(np.ndarray, np.ndarray)
        Values of const with errors.
    slope: Measurement(np.ndarray, np.ndarray)
        Values of slope with errors.
    chi2: np.ndarray
        Chi2 from each fit.
    valid: boolean np.ndarray
        Where the fit has been succesfull.
    """
    nbins_x = np.size (xbins) - 1
    nbins_y = np.size (ybins) - 1
    nbins   = nbins_x, nbins_y
    const   = np.zeros(nbins)
    slope   = np.zeros(nbins)
    constu  = np.zeros(nbins)
    slopeu  = np.zeros(nbins)
    chi2    = np.zeros(nbins)
    valid   = np.zeros(nbins, dtype=bool)

    if zrange is None:
        zrange = np.min(zdata), np.max(zdata)
    for i in range(nbins_x):
        sel_x = in_range(xdata, *xbins[i:i + 2])
        for j in range(nbins_y):
            sel_y = in_range(ydata, *ybins[j:j + 2])
            sel   = sel_x & sel_y
            if np.count_nonzero(sel) < min_entries: continue

            try:
                f = fit_profile_1d_expo(zdata[sel], tdata[sel], nbins_z, xrange=zrange)
                const [i, j] = f.values[0]
                constu[i, j] = f.errors[0]
                slope [i, j] = f.values[1]
                slopeu[i, j] = f.errors[1]
                chi2  [i, j] = f.chi2
                valid [i, j] = True
            except Exception as exc:
                if not isinstance(exc, ignore_errors):
                    raise

    return Measurement(const, constu), Measurement(slope, slopeu), chi2, valid


def sigmoid(x          : np.array,
            scale      : float,
            inflection : float,
            slope      : float,
            offset     : float)->np.array:

    return scale / ( 1 + np.exp( - slope * ( x - inflection ) ) ) + offset


def compute_drift_v(zdata    : np.array,
                    nbins    : int,
                    zrange   : Tuple[float, float],
                    detector : str,
                    seed     : Tuple[float, float, float, float] = None,
                    )->Tuple[float, float]:
    """
    Computes the drift velocity for a given distribution
    using the sigmoid function to get the cathode edge.

    Parameters
    ----------
    zdata: array_like
        Values of Z coordinate.
    nbins: int (optional)
        The number of bins in the z coordinate for the binned fit.
    zrange: length-2 tuple (optional)
        Fix the range in z.
    seed: length-4 tuple (optional)
        Seed for the fit.
    detector: string (optional)
        Used to get the cathode position from DB.
    plot_fit: boolean (optional)
        Flag for plotting the results.

    Returns
    -------
    dv: float
        Drift velocity.
    dvu: float
        Drift velocity uncertainty.
    """

    y, x = np.histogram(zdata, nbins, zrange)
    x    = shift_to_bin_centers(x)

    if seed is None: seed = np.max(y), np.mean(zrange), 0.5, np.min(y)

    if detector == "new":
        z_cathode = DB.DetectorGeo(detector).ZMAX[0]
    elif detector == "next100":
        z_cathode = 1187 # TEMPORARY
    try:
        f = fitf.fit(sigmoid, x, y, seed, sigma=poisson_sigma(y), fit_range=zrange)
        dv  = z_cathode/f.values[1]
        dvu = dv / f.values[1] * f.errors[1]
    except RuntimeError:
        print("WARNING: Sigmoid fit for dv computation fails. NaN value will be set in its place.")
        dv, dvu = np.nan, np.nan

    return dv, dvu
