"""Module fit_lt_functions.
This module includes the fit functions for the lifetime.

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

from typing  import List
from typing  import Tuple
from typing  import Iterable

from numpy           .linalg                 import LinAlgError

from invisible_cities.core   .fit_functions  import fit
from invisible_cities.core   .fit_functions  import expo
from invisible_cities.core   .core_functions import in_range

from . fit_functions    import expo_seed
from . fit_functions    import chi2f
from . histo_functions  import profile1d
from . core_functions   import NN

from . kr_types import FitPar
from . kr_types import FitResult
from . kr_types import HistoPar2
from . kr_types import FitCollection
from . kr_types import FitCollection2
from . kr_types import FitType
from . kr_types import Measurement

import logging
log = logging.getLogger(__name__)

def fit_lifetime(z       : np.array,
                 e       : np.array,
                 nbins_z : int,
                 nbins_e : int,
                 range_z : Tuple[float,float],
                 range_e : Tuple[float,float],
                 fit     : FitType = FitType.unbined)->FitCollection2:
    """
    Fits the lifetime using a profile (FitType.profile) or an unbined
    fit (FitType.unbined).

    Parameters
    ----------
        z
            Array of z values.
        e
            Array of energy values.
        nbins_z
            Number of bins in Z for the profile fit.
        nbins_e
            Number of bins in energy.
        range_z
            Range in Z for fit.
        range_e
            Range in energy.
        fit
            Selects fit type.


    Returns
    -------
        A FitCollection2, which allows passing two fit parameters, histo parameters and fitResult.

    @dataclass
    class FitCollection:
        fp   : FitPar
        hp   : HistoPar
        fr   : FitResult


    @dataclass
    class FitCollection2(FitCollection):
        fp2   : FitPar

    """
    logging.debug(' fit_liftime ')
    logging.debug(f' len (z) ={len(z)}, len (e) ={len(e)} ')
    logging.debug(f' nbins_z ={nbins_z}, nbins_e ={nbins_e} range_z ={range_z} range_e ={range_e} ')

    hp = HistoPar2(var = z,
                   nbins = nbins_z,
                   range = range_z,
                   var2 = e,
                   nbins2 = nbins_e,
                   range2 = range_e)

    if fit == FitType.profile:
        fp, fp2, fr = fit_lifetime_profile(z, e, nbins_z, range_z)
    else:
        fp, fp2, fr = fit_lifetime_unbined(z, e, nbins_z, range_z)

    return FitCollection2(fp = fp, fp2 = fp2, hp = hp, fr = fr)
    logging.debug(f' fp ={fp}, fp2 ={fp2} ')

def fit_lifetime_profile(z : np.array,
                         e : np.array,
                         nbins_z : int,
                         range_z : Tuple[float,float])->Tuple[FitPar, FitPar, FitResult]:
    """
    Make a profile of the input data and fit it to an exponential
    function.
    Parameters
    ----------
        z
            Array of z values.
        e
            Array of energy values.
        nbins_z
            Number of bins in Z for the profile fit.
        range_z
            Range in Z for fit.


    Returns
    -------
        A Tuple with:
            FitPar : Fit parameters (arrays of fitted values and errors, fit function)
            FitPar : Fit parameters (duplicated to make it compatible with fit_liftime_unbined)
            FirResults: Fit results (lt, e0, errors, chi2)

    @dataclass
    class ProfilePar:
        x  : np.array
        y  : np.array
        xu : np.array
        yu : np.array

    @dataclass
    class FitPar(ProfilePar):
        f     : FitFunction

    @dataclass
    class FitResult:
        par  : np.array
        err  : np.array
        chi2 : float
        valid : bool

    """

    logging.debug(' fit_liftime_profile')
    logging.debug(f' len (z) ={len(z)}, len (e) ={len(e)} ')
    logging.debug(f' nbins_z ={nbins_z}, range_z ={range_z} ')
    fp    = None
    valid = True
    c2    = NN
    par   = NN  * np.ones(2)
    err   = NN  * np.ones(2)

    x, y, yu  = profile1d(z, e, nbins_z, range_z)
    xu        = np.diff(x) * 0.5
    seed      = expo_seed(x, y)

    logging.debug(f' after profile: len (x) ={len(x)}, len (y) ={len(y)} ')
    try:
        f      = fit(expo, x, y, seed, sigma=yu)
        c2     = f.chi2
        par    = np.array(f.values)
        par[1] = - par[1]
        err    = np.array(f.errors)

        logging.debug(f' e0z ={par[0]} +- {err[0]} ')
        logging.debug(f' lt ={par[1]} +- {err[1]} ')
        logging.debug(f' c2 ={c2} ')
        fp = FitPar(x  = x,
                    y  = y,
                    xu = xu,
                    yu = yu,
                    f  = f.fn)
    except:
        warnings.warn(f' fit failed for seed  = {seed} in fit_lifetime_profile', UserWarning)
        valid = False
        raise

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    return fp, fp, fr


def fit_lifetime_unbined(z       : np.array,
                         e       : np.array,
                         nbins_z : int,
                         range_z : Tuple[float,float])->Tuple[FitPar, FitPar, FitResult]:
    """
    Based on

    numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
    Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y).
    Returns a vector of coefficients p that minimises the squared error.

    E = E0 exp(-z/lt)
    y = -log(E) = (z/lt) - log(E)

    Parameters
    ----------
        z
            Array of z values.
        e
            Array of energy values.
        nbins_z
            Number of bins in Z for the profile fit.
        range_z
            Range in Z for fit.


    Returns
    -------
        A Tuple with:
            FitPar : Fit parameters (arrays of fitted values and errors, fit function)
            FitPar : Fit parameters
            FirResults: Fit results (lt, e0, errors, chi2)

    """

    logging.debug(' fit_liftime_unbined')
    logging.debug(f' len (z) ={len(z)}, len (e) ={len(e)} ')
    logging.debug(f' nbins_z ={nbins_z}, range_z ={range_z} ')

    fp    = None
    fp2   = None
    valid = True
    c2    = NN
    par   = NN  * np.ones(2)
    err   = NN  * np.ones(2)
    try:
        el = - np.log(e)
        z_sel = in_range(z, *range_z)
        cc, cov = np.polyfit(z[z_sel], el[z_sel], deg=1, full=False, w=None, cov=True)
        a, b = cc[0], cc[1]

        lt   = 1/a
        par[1] = lt
        err[1] = lt**2 * np.sqrt(cov[0, 0])

        e0     = np.exp(-b)
        par[0] = e0
        err[0] = e0    * np.sqrt(cov[1, 1])

        x, y, yu     = profile1d(z, e, nbins_z, range_z)
        xs, ys, yus  = profile1d(z, el, nbins_z, range_z)
        xu           = np.diff(x) * 0.5
        xus          = np.diff(xs) * 0.5

        logging.debug(f' after profile: len (x) ={len(x)}, len (y) ={len(y)} ')

        c2 = chi2f(lambda z: a * xs + b, 2, xs, ys, yus)

        logging.debug(f' e0z ={par[0]} +- {err[0]} ')
        logging.debug(f' lt ={par[1]} +- {err[1]} ')
        logging.debug(f' c2 ={c2} ')

        fp  = FitPar(x  = x,  y  = y,  xu = xu,  yu = yu,  f  = lambda z: e0 * np.exp(-z/lt))
        fp2 = FitPar(x  = xs, y  = ys, xu = xus, yu = yus, f  = lambda z: a * xs + b)

    except ValueError:
        logging.warn(f'Value Error found in fit_lifetime_unbined: not enough events for fit')
        valid = False

    except TypeError:
        logging.warn(f'Type error found in fit_lifetime_unbined: not enough events for fit')
        valid = False

    except SystemError:
        logging.warn(f'SystemError error found in fit_lifetime_unbined: not enough events for fit')
        valid = False

    except LinAlgError:
        logging.warn(f'LinAlgError error found in fit_lifetime_unbined: not enough events for fit')
        valid = False

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    return fp, fp2, fr


def pars_from_fcs(fcs : List[FitCollection])->Tuple[List[Measurement],
                                                    List[Measurement],
                                                    np.array]:
    E=[]
    LT = []
    C2 = []
    for fc in fcs:
        if fc.fr.valid:
            par  = fc.fr.par
            err  = fc.fr.err
            E.append(Measurement(par[0], err[0]))
            LT.append(Measurement(par[1], err[1]))
            C2.append(fc.fr.chi2)
        else:
            warnings.warn(f' fit did not succeed, returning NaN ', UserWarning)
            E.append(Measurement(NN, NN))
            LT.append(Measurement(NN, NN))
            C2.append(NN)
    return E, LT, np.array(C2)

def lt_params_from_fcs(fcs : Iterable[FitCollection])->Iterable[float]:
    e0s   = np.array([fc.fr.par[0] for fc in fcs])
    ue0s  = np.array([fc.fr.err[0] for fc in fcs])
    lts   = np.array([fc.fr.par[1] for fc in fcs])
    ults  = np.array([fc.fr.err[1] for fc in fcs])
    chi2s = np.array([fc.fr.chi2   for fc in fcs])
    return e0s, ue0s, lts, ults, chi2s
