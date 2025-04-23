from . fit_lt_functions     import fit_lifetime_profile
from . correction_functions import e0_xy_correction
from . fit_functions        import compute_drift_v
from . fit_functions        import quick_gauss_fit
from . kr_types             import ASectorMap
from . kr_types             import masks_container
from . core_functions       import resolution

from invisible_cities.reco .corrections import apply_all_correction
from invisible_cities.types.symbols     import NormStrategy

from typing import List
from typing import Tuple

import pandas as pd
import numpy  as np


def get_number_of_time_bins(nStimeprofile: int,
                            tstart       : int,
                            tfinal       : int )->int:
    """
    Computes the number of time bins to compute for a given time step
    in seconds.

    Parameters
    ----------
    nStimeprofile: int
        Time step in seconds
    tstart: int
        Initial timestamp for the data set
    tfinal: int
        Final timestamp for the data set

    Returns
    -------
    ntimebins: int
        Number of bins
    """
    ntimebins = int( np.floor( ( tfinal - tstart) / nStimeprofile) )
    ntimebins = np.max([ntimebins, 1])

    return ntimebins

def computing_kr_parameters(data       : pd.DataFrame,
                            ts         : float,
                            emaps      : ASectorMap,
                            zslices_lt : int,
                            zrange_lt  : Tuple[float,float],
                            nbins_dv   : int,
                            zrange_dv  : Tuple[float, float],
                            detector   : str)->pd.DataFrame:

    """
    Computes some average parameters (e0, lt, drift v, energy
    resolution, S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution. Returns a DataFrame.

    Parameters
    ----------
    data: DataFrame
        Kdst distribution to analyze.
    ts: float
        Central time of the distribution.
    emaps: correction map
        Allows geometrical correction of the energy.
    xr_map, yr_map: length-2 tuple
        Set the X/Y-coordinate range of the correction map.
    nx_map, ny_map: int
        Set the number of X/Y-coordinate bins for the correction map.
    zslices_lt: int
        Number of Z-coordinate bins for doing the exponential fit to compute
        the lifetime.
    zrange_lt: length-2 tuple (optional)
        Number of Z-coordinate range for doing the exponential fit to compute
        the lifetime.
    nbins_dv: int (optional)
        Number of bins in Z-coordinate for doing the histogram to compute
        the drift velocity.
    zrange_dv: int (optional)
        Range in Z-coordinate for doing the histogram to compute the drift
        velocity.
    detector: string (optional)
        Used to get the cathode position from DB for the drift velocity
        computation.
    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value of a different parameter.
    """

    ## lt and e0
    geo_correction_factor = e0_xy_correction(map           =  emaps,
                                             norm_strategy = NormStrategy.max)

    _, _, fr = fit_lifetime_profile(data.Z,
                                    data.S2e.values*geo_correction_factor(
                                        data.X.values,
                                        data.Y.values),
                                    zslices_lt, zrange_lt)
    e0,  lt  = fr.par
    e0u, ltu = fr.err

    ## compute drift_v
    dv, dvu  = compute_drift_v(data.Z, nbins=nbins_dv,
                               zrange=zrange_dv, detector=detector)

  ## energy resolution and error
    tot_corr_factor = apply_all_correction(maps = emaps,
                                           apply_temp=False)
    nbins = int((len(data.S2e))**0.5)
    f, _  = quick_gauss_fit(data.S2e .values*tot_corr_factor(data.X   .values,
                                                             data.Y   .values,
                                                             data.Z   .values,
                                                             data.time.values),
                            bins=nbins)
    R = resolution(f.values, f.errors, 41.5)
    resol, err_resol = R[0][0], R[0][1]
    ## average values
    parameters = ['S1w', 'S1h', 'S1e',
                  'S2w', 'S2h', 'S2e', 'S2q',
                  'Nsipm', 'Xrms', 'Yrms']
    mean_d, var_d = {}, {}
    for parameter in parameters:
        data_value           = getattr(data, parameter)
        mean_d[parameter] = np.mean(data_value)
        var_d [parameter] = (np.var(data_value)/len(data_value))**0.5

    ## saving as pd.DataFrame
    pars = pd.DataFrame({'ts'   : [ts]             ,
                         'e0'   : [e0]             , 'e0u'   : [e0u]           ,
                         'lt'   : [lt]             , 'ltu'   : [ltu]           ,
                         'dv'   : [dv]             , 'dvu'   : [dvu]           ,
                         'resol': [resol]          , 'resolu': [err_resol]     ,
                         's1w'  : [mean_d['S1w']]  , 's1wu'  : [var_d['S1w']]  ,
                         's1h'  : [mean_d['S1h']]  , 's1hu'  : [var_d['S1h']]  ,
                         's1e'  : [mean_d['S1e']]  , 's1eu'  : [var_d['S1e']]  ,
                         's2w'  : [mean_d['S2w']]  , 's2wu'  : [var_d['S2w']]  ,
                         's2h'  : [mean_d['S2h']]  , 's2hu'  : [var_d['S2h']]  ,
                         's2e'  : [mean_d['S2e']]  , 's2eu'  : [var_d['S2e']]  ,
                         's2q'  : [mean_d['S2q']]  , 's2qu'  : [var_d['S2q']]  ,
                         'Nsipm': [mean_d['Nsipm']], 'Nsipmu': [var_d['Nsipm']],
                         'Xrms' : [mean_d['Xrms']] , 'Xrmsu' : [var_d['Xrms']] ,
                         'Yrms' : [mean_d['Yrms']] , 'Yrmsu' : [var_d['Yrms']]})

    return pars


def kr_time_evolution(ts         : np.array,
                      masks_time : List[np.array],
                      dst        : pd.DataFrame,
                      emaps      : ASectorMap,
                      xr_map     : Tuple[float, float],
                      yr_map     : Tuple[float, float],
                      nx_map     : int,
                      ny_map     : int,
                      zslices_lt : int,
                      zrange_lt  : Tuple[float,float],
                      nbins_dv   : int,
                      zrange_dv  : Tuple[float, float],
                      detector   : str)->pd.DataFrame:
    """
    Computes some average parameters (e0, lt, drift v,
    S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution and for different time slices.
    Returns a DataFrame.

    Parameters
    ----------
    ts: np.array of floats
        Sequence of central times for the different time slices.
    masks_time: list of boolean lists
        Allows dividing the distribution into time slices.
    data: DataFrame
        Kdst distribution to analyze.
    emaps: correction map
        Allows geometrical correction of the energy.
    xr_map, yr_map: length-2 tuple
        Set the X/Y-coordinate range of the correction map.
    nx_map, ny_map: int
        Set the number of X/Y-coordinate bins for the correction map.
    z_slices: int (optional)
        Number of Z-coordinate bins for doing the exponential fit to compute
        the lifetime.
    zrange_lt: length-2 tuple (optional)
        Number of Z-coordinate range for doing the exponential fit to compute
        the lifetime.
    nbins_dv: int (optional)
        Number of bins in Z-coordinate for doing the histogram to compute
        the drift velocity.
    zrange_dv: int (optional)
        Range in Z-coordinate for doing the histogram to compute the drift
        velocity.
    detector: string (optional)
        Used to get the cathode position from DB for the drift velocity
        computation.

    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value for a given parameter.
        Each row corresponds to the parameters for a given time slice.
    """

    frames = []
    for index in range(len(masks_time)):
        sel_dst = dst[masks_time[index]]
        pars    = computing_kr_parameters(sel_dst, ts[index],
                                          emaps,
                                          zslices_lt, zrange_lt,
                                          nbins_dv, zrange_dv,
                                          detector)
        frames.append(pars)

    total_pars = pd.concat(frames, ignore_index=True)

    return total_pars

def cut_time_evolution(masks_time : List[np.array],
                       dst        : pd.DataFrame,
                       masks_cuts : masks_container,
                       pars_table : pd.DataFrame):

    """
    Computes the efficiency evolution in time for a given krypton distribution
    for different time slices.
    Returns the input DataFrame updated with new 3 columns.

    Parameters
    ----------
    masks: list of boolean lists
        Allows dividing the distribution into time slices.
    data: DataFrame
        Kdst distribution to analyze.
    masks_cuts: masks_container
        Container for the S1, S2 and Band cuts masks.
        The masks don't have to be inclusive.
    pars: DataFrame
        Each column corresponds to the average value for a given parameter.
        Each row corresponds to the parameters for a given time slice.

    Returns
    -------
    parspars_table_out: DataFrame
        pars Table imput updated with 3 new columns, one for each cut.
    """

    len_ts = len(masks_time)
    n0     = np.zeros(len_ts)
    nS1    = np.zeros(len_ts)
    nS2    = np.zeros(len_ts)
    nBand  = np.zeros(len_ts)
    for index in range(len_ts):
        t_mask       = masks_time[index]
        n0   [index] = dst[t_mask].event.nunique()
        nS1mask      = t_mask  & masks_cuts.s1
        nS1  [index] = dst[nS1mask].event.nunique()
        nS2mask      = nS1mask & masks_cuts.s2
        nS2  [index] = dst[nS2mask].event.nunique()
        nBandmask    = nS2mask & masks_cuts.band
        nBand[index] = dst[nBandmask].event.nunique()

    pars_table_out = pars_table.assign(S1eff   = nS1   / n0,
                                       S2eff   = nS2   / nS1,
                                       Bandeff = nBand / nS2)
    return pars_table_out
