from typing      import Tuple
from typing      import Callable
from dataclasses import dataclass
from copy        import deepcopy

import pandas as pd
import numpy  as np
import glob
import os

from .. core.kr_types                      import type_of_signal
from .. core.kr_types                      import FitType
from .. core.kr_types                      import masks_container
from .. core.selection_functions           import selection_in_band
from .. core.selection_functions           import select_xy_sectors_df
from .. core.selection_functions           import event_map_df
from .. core.selection_functions           import get_time_series_df
from .. core.fitmap_functions              import fit_map_xy_df
from .. core.map_functions                 import amap_from_tsmap
from .. core.map_functions                 import tsmap_from_fmap
from .. core.map_functions                 import add_mapinfo
from .. core.map_functions                 import amap_replace_nan_by_mean
from .. core.map_functions                 import relative_errors
from .. core.correction_functions          import e0_xy_correction
from .. core.kr_parevol_functions          import kr_time_evolution
from .. core.kr_parevol_functions          import cut_time_evolution
from .. core.kr_parevol_functions          import get_number_of_time_bins
from .. core.io_functions                  import write_complete_maps
from .. core.io_functions                  import compute_and_save_hist_as_pd
from .. core.io_functions                  import compute_and_save_hist2d_as_pd
from .. core.histo_functions               import compute_similar_histo
from .. core.histo_functions               import normalize_histo_and_poisson_error
from .. core.histo_functions               import ref_hist

from . checking_functions                  import check_if_values_in_interval
from . checking_functions                  import check_failed_fits
from . checking_functions                  import get_core


from invisible_cities.core.core_functions  import in_range
from invisible_cities.io  .dst_io          import load_dsts
from invisible_cities.reco.corrections     import ASectorMap
from invisible_cities.reco.corrections     import read_maps
from invisible_cities.types.symbols        import NormStrategy


@dataclass
class ref_hist_container:
    Z_dist_hist : ref_hist

def quality_cut(dst : pd.DataFrame, r_max : float) -> pd.DataFrame:
    """
    Does basic quality cut : R inside the r_max
    Parameters
    ----------
    dst : pd.DataFrame
        Input dst to obtain the mask from.
    r_max: float
        Upper limit for R.
    Returns
    ----------
    mask : pd.DataFrame
        Mask for filtering events not matching the criteria
    """
    mask = in_range(dst.R, 0, r_max)
    return mask

def load_data(input_path         : str ,
              input_dsts         : str ,
              file_bootstrap_map : str ,
              ref_histo_file     : str ,
              key_Z_histo        : str ,
              quality_ranges     : dict ) -> Tuple[pd.DataFrame,
                                                   ASectorMap  ,
                                                   ref_hist_container]:
    """
    Reads kdst files and applies basic R cut. Outputs kdst as pd.DataFrame,
    bootstrap map, and reference histograms

    Parameters
    ----------
    input_path : str
        Path to the input map_folder
    input_dsts : str
        Name criteria for the dst to be read
    file_bootstrap_map : str
        Path to the bootstrap map file
    ref_Z_histo_file : str
        Path to the reference histogram file
    quality_ranges : dict
        Dictionary containing ranges for the quality cuts

    Returns
    ----------
    dst_filtered : pd.DataFrame
        Dst containing all the events once filtered
    bootstrap_map : ASectorMap
        Bootstrap map
    ref_hist_container : ref_hist_container
        To be completed
    """

    input_path         = os.path.expandvars(input_path)
    dst_files          = glob.glob(input_path + input_dsts)
    dst_full           = load_dsts(dst_files, "DST", "Events")
    dst_full           = dst_full.sort_values(by=['time'])
    mask_quality       = quality_cut(dst_full, **quality_ranges)
    dst_filtered       = dst_full[mask_quality]

    file_bootstrap_map = os.path.expandvars(file_bootstrap_map)
    bootstrap_map      = read_maps(file_bootstrap_map)

    ref_histo_file     = os.path.expandvars(ref_histo_file)
    z_pd               = pd.read_hdf(ref_histo_file, key=key_Z_histo)
    z_histo            = ref_hist(bin_centres     = z_pd.bin_centres,
                                  bin_entries     = z_pd.bin_entries,
                                  err_bin_entries = z_pd.err_bin_entries)
    ref_histos         =  ref_hist_container(Z_dist_hist = z_histo)

    return dst_filtered, bootstrap_map, ref_histos

def selection_nS_mask_and_checking(dst        : pd.DataFrame       ,
                                   column     : type_of_signal     ,
                                   interval   : Tuple[float, float],
                                   output_f   : pd.HDFStore        ,
                                   nbins_hist : int                ,
                                   range_hist : Tuple[float, float],
                                   input_mask : np.array = None    ,
                                   norm       : bool     = True    )->np.array:
    """
    Selects nS1(or nS2) == 1 for a given kr dst and
    returns the mask. It also computes selection efficiency,
    checking if the value is within a given interval, and
    saves histogram parameters.
    Parameters
    ----------
    dst: pd.Dataframe
        Krypton dst dataframe.
    column: type_of_signal
        The function can be appplied over nS1 or nS2.
    interval: length-2 tuple
        If the selection efficiency is out of this interval
        (given by the config file) the map production will abort.
    output_f: pd.HDFStore
        File where histogram will be saved.
    input_mask: np.array (Optional)
        Selection mask of the previous cut. If this is the first selection
        cut, input_mask is set to be an all True array.
    nbins_hist: int
        Number of bins to make the histogram.
    range_hist: length-2 tuple (optional)
        Range of the histogram.
    norm: bool
        If True, histogram will be normalized.
    Returns
    ----------
        A mask corresponding to the selected events.
    """
    if input_mask is None:
        input_mask = [True] * len(dst)

    mask             = np.zeros_like(input_mask)
    mask[input_mask] = getattr(dst[input_mask], column.value) == 1
    nevts_after      = dst[mask]      .event.nunique()
    nevts_before     = dst[input_mask].event.nunique()
    eff              = nevts_after / nevts_before

    mod_dst = dst[['event', column.value]].drop_duplicates()
    compute_and_save_hist_as_pd(values     = getattr(mod_dst,
                                                     column.value),
                                out_file   = output_f,
                                hist_name  = column.value,
                                n_bins     = nbins_hist,
                                range_hist = range_hist,
                                norm       = norm)

    message  = "Selection efficiency of "
    message += column.value
    message += "==1 ({0}) out of range ".format(np.round(eff, 3))
    message += "({0} - {1}).".format(interval[0], interval[1])
    check_if_values_in_interval(values          = np.array(eff),
                                low_lim         = interval[0]  ,
                                up_lim          = interval[1]  ,
                                raising_message = message      )
    return mask

def check_Z_dst(Z_vect   : np.array     ,
                ref_hist : pd.DataFrame ,
                n_sigmas : int      = 10)->None:
    """
    From a given Z distribution, this function checks, raising
    an exception, if Z histogram is correct.
    Parameters:
    ----------
    Z_vect : np.array
        Array of Z values for each kr event.
    ref_hist: pd.DataFrame
        Table that contains reference histogram info.
    n_sigmas: int
        Number of sigmas to consider if distributions are similar enough.
    Returns
    -------
        Continue if both Z distributions are compatible
        within a 'n_sigmas' interval.
    """
    N_Z, z_Z   = compute_similar_histo(param     = Z_vect,
                                       reference = ref_hist)
    N_Z, err_N = normalize_histo_and_poisson_error(N = N_Z,
                                                   b = z_Z)

    diff       = N_Z - ref_hist.bin_entries
    diff_sig   = diff / np.sqrt(err_N**2+ref_hist.err_bin_entries**2)

    message    = "Z distribution very different to reference one. "
    message   += "At least 1 point out of {0} sigmas region. ".format(n_sigmas)
    check_if_values_in_interval(values          = diff_sig ,
                                low_lim         = -n_sigmas,
                                up_lim          = n_sigmas ,
                                raising_message = message  )
    return;

def check_rate_and_hist(times      : np.array           ,
                        output_f   : pd.HDFStore        ,
                        name_table : str                ,
                        n_dev      : float       = 5    ,
                        bin_size   : int         = 180  ,
                        normed     : bool        = False)->None:
    """
    Raises exception if evolution of rate vs time is
    not flat. It also computes histogram.
    Parameters
    ----------
    times: np.array
        Time of the events.
    output_f: pd.HDFStore
        File where histogram will be saved.
    name_table: string
        Name for the histogram table inside file.
    n_dev: float
        Relative standard deviation to judge if
        distribution is correct.
    bin_size: int
        Size (in seconds) for histogram bins.
        By default it corresponds to 3 min.
    normed: bool (optional)
        If True, histogram will be normalized.
    Returns
    ----------
        Nothing.

    """
    min_time   = times.min()
    max_time   = times.max()

    ntimebins  = get_number_of_time_bins(bin_size,
                                         min_time,
                                         max_time)

    compute_and_save_hist_as_pd(values    = times      ,
                                out_file  = output_f   ,
                                hist_name = name_table ,
                                n_bins    = ntimebins  ,
                                range_hist= (min_time  ,
                                             max_time) ,
                                norm      = normed     )

    n, _     = np.histogram(times, bins=ntimebins,
                            range = (min_time, max_time))
    mean     = np.mean(n)
    dev      = np.std(n, ddof = 1)
    rel_dev  = dev / mean * 100

    message  = "Relative deviation ({0}) greater ".format(rel_dev)
    message += "than the allowed one ({0}).".format(n_dev)
    check_if_values_in_interval(values          = np.array(rel_dev),
                                low_lim         = 0                ,
                                up_lim          = n_dev            ,
                                raising_message = message          )
    return;

def band_selector_and_check(dst        : pd.DataFrame             ,
                            boot_map   : ASectorMap               ,
                            range_Z    : Tuple[np.array, np.array],
                            range_E    : Tuple[np.array, np.array],
                            nbins_z    : int                      ,
                            nbins_e    : int                      ,
                            nsigma_sel : float                    ,
                            eff_min    : float                    ,
                            eff_max    : float                    ,
                            norm_strat : NormStrategy = NormStrategy.max,
                            input_mask : np.array     = None,
                           )->np.array:
    """
    This function returns a selection of the events that
    are inside the Kr E vz Z band, and checks
    if the selection efficiency is correct.

    Parameters
    ----------
    dst : pd.DataFrame
        Krypton dataframe.
    boot_map: str
        Name of bootstrap map file.
    norm_strt: norm_strategy
        Provides the desired normalization to be used.
    mask_input: np.array
        Mask of the previous selection cut.
    range_Z: Tuple[np.array, np.array]
        Range in Z-axis
    range_E: Tuple[np.array, np.array]
        Range in Energy-axis
    nbins_z: int
        Number of bins in Z-axis
    nbins_e: int
        Number of bins in energy-axis
    nsigma_sel: float
        Number of sigmas to set the band width
    eff_min: float
        Lower limit of the range where selection efficiency
        is considered correct.
    eff_max: float
        Upper limit of the range where selection efficiency
        is considered correct.
    Returns
    ----------
        A  mask corresponding to the selection made.
    """
    if input_mask is None:
        input_mask = [True] * len(dst)
    else: pass;

    emaps = e0_xy_correction(boot_map, norm_strategy = norm_strat)
    E0    = dst[input_mask].S2e.values * emaps(dst[input_mask].X.values,
                                               dst[input_mask].Y.values)

    sel_krband = np.zeros_like(input_mask)
    sel_krband[input_mask], _, _, _, _ = selection_in_band(dst[input_mask].Z,
                                                           E0,
                                                           range_z = range_Z,
                                                           range_e = range_E,
                                                           nbins_z = nbins_z,
                                                           nbins_e = nbins_e,
                                                           nsigma  = nsigma_sel)

    effsel   = dst[sel_krband].event.nunique()/dst[input_mask].event.nunique()
    message  = "Band selection efficiency {0} ".format(np.round(effsel, 3))
    message += "out of range: ({0} - {1}).".format(eff_min, eff_max)
    check_if_values_in_interval(values          = np.array(effsel),
                                low_lim         = eff_min         ,
                                up_lim          = eff_max         ,
                                raising_message = message         )

    return sel_krband

def get_binning_auto(nevt_sel: int,
                     thr_events_for_map_bins: int = 1e6,
                     n_bins:                  int = None)->int:
    """
    Computes the number of X-Y bins to be used in the creation
    of correction map regarding the number of selected events.
    Parameters
    ---------
    nevt_sel: int
        Number of kr events to compute the map.
    thr_events_for_map_bins: int (optional)
        Threshold to use 50x50 or 100x100 maps (standard values).
    n_bins: int (optional)
        The number of events to use can be chosen a priori.
    Returns
    ---------
    n_bins: int
        Number of bins in each direction (X,Y) (square map).
    """
    if n_bins != None: pass;
    elif nevt_sel<thr_events_for_map_bins:
        n_bins = 50;
    else: n_bins = 100
    return n_bins;

def calculate_map(dst     : pd.DataFrame,
                  XYbins  : Tuple[int, int],
                  nbins_z : int,
                  nbins_e : int,
                  z_range : Tuple[float, float],
                  e_range : Tuple[float, float],
                  chi2_range: Tuple[float, float],
                  lt_range: Tuple[float, float],
                  fit_type: FitType,
                  nmin    : int,
                  x_range : Tuple[float, float],
                  y_range : Tuple[float, float],
                  ):
    """
    Calculates and outputs correction map

    Parameters
    ---------
    dst: pd.DataFrame
        Dst where to stract the map from
    XYbins: Tuple[int, int]
        Number of bins for XY map
    nbins_z : int
        Number of bins for z
        The number of events to use can be chosen a priori.
    Returns
    ---------
    n_bins: int
        Number of bins in each direction (X,Y) (square map).
    """
    xbins = np.linspace(*x_range, XYbins[0]+1)
    ybins = np.linspace(*y_range, XYbins[1]+1)
    KXY   = select_xy_sectors_df(dst, xbins, ybins)
    nXY   = event_map_df(KXY)
    fmxy  = fit_map_xy_df(selection_map = KXY,
                          event_map     = nXY,
                          n_time_bins   = 1,
                          time_diffs    = dst.time.values,
                          nbins_z       = nbins_z,
                          nbins_e       = nbins_e,
                          range_z       = z_range,
                          range_e       = e_range,
                          energy        = 'S2e',
                          z             = 'Z',
                          fit           = fit_type,
                          n_min         = nmin)
    tsm   = tsmap_from_fmap(fmxy)
    am    = amap_from_tsmap(tsm,
                            ts         = 0,
                            range_e    = e_range,
                            range_chi2 = chi2_range,
                            range_lt   = lt_range)

    return am

def find_outliers(maps : ASectorMap, x2range : Tuple[float, float] = (0, 2)):
    """
    For a given maps and deserved range, it returns a mask where values are
    within the interval.

    Parameters
    ---------
    maps: ASectorMap
        Map to check the outliers
    x2range : Tuple[float, float]
        Range for chi2

    Returns
    ---------
    mask: pd.DataFrame
        Mask.
    """
    mask = in_range(maps.chi2, *x2range)
    return mask

def regularize_map(maps : ASectorMap, x2range : Tuple[float, float] = (0, 2) ):
    """
    For a given map and deserved range, it replaces where values are
    outside the provided interval by the average.

    Parameters
    ---------
    maps: ASectorMap
        Map to check the outliers
    x2range : Tuple[float, float]
        Range for chi2

    Returns
    ---------
    amap: ASectorMap
        Regularized map
    """
    amap               = deepcopy(maps)
    outliers           = np.logical_not(find_outliers(amap, x2range))

    amap.lt [outliers] = np.nan
    amap.ltu[outliers] = np.nan
    amap.e0 [outliers] = np.nan
    amap.e0u[outliers] = np.nan
    asm                = relative_errors(amap)
    amap               = amap_replace_nan_by_mean(asm)

    return amap

def remove_peripheral(map       : ASectorMap,
                      nbins     : int       ,
                      rmax      : float     ,
                      rfid      : float     ,
                      ) -> ASectorMap:

    new_map     = deepcopy(map)
    mask_core   = get_core(nbins,rmax, rfid)
    new_map.e0  = new_map.e0.where(mask_core)
    new_map.e0u = new_map.e0u.where(mask_core)
    new_map.lt  = new_map.lt.where(mask_core)
    new_map.ltu = new_map.ltu.where(mask_core)

    return new_map

def add_krevol(maps         : ASectorMap,
               dst          : pd.DataFrame,
               masks_cuts   : masks_container,
               r_fid        : float,
               nStimeprofile: int,
               x_range      : Tuple[float, float],
               y_range      : Tuple[float, float],
               XYbins       : Tuple[int, int]    ,
               **kwargs                          ) -> None:
    """
    Adds time evolution dataframe to the map

    Parameters
    ---------
    maps: ASectorMap
        Map to check the outliers
    dst: pd.DataFrame
        Dst where to stract the data from
    masks_cuts: masks_container
        Container for the S1, S2 and Band cuts masks
    r_fid: float
        Maximum radius for fiducial sample
    nStimeprofile: int
        Number of seconds for each time bin
    x_range, y_range: Tuple[float, float]:
        Range for x and y for the map
    XYbins: Tuple[int, int]
        Number of bins for XY map

    Returns
    ---------
    Nothing
    """

    fmask     = (dst.R < r_fid) & masks_cuts.s1 & masks_cuts.s2 & masks_cuts.band
    dstf      = dst[fmask]
    min_time  = dstf.time.min()
    max_time  = dstf.time.max()
    ntimebins = get_number_of_time_bins(nStimeprofile = nStimeprofile,
                                        tstart        = min_time,
                                        tfinal        = max_time)

    ts, masks_time = get_time_series_df(time_bins  = ntimebins,
                                        time_range = (min_time, max_time),
                                        dst        = dst)

    masks_timef    = [mask[fmask] for mask in masks_time]
    pars           = kr_time_evolution(ts         = ts,
                                       masks_time = masks_timef,
                                       dst        = dstf,
                                       emaps      = maps,
                                       xr_map     = x_range,
                                       yr_map     = y_range,
                                       nx_map     = XYbins[0],
                                       ny_map     = XYbins[1])

    pars_ec        = cut_time_evolution(masks_time = masks_time,
                                        dst        = dst,
                                        masks_cuts = masks_cuts,
                                        pars_table = pars)

    e0par       = np.array([pars['e0'].mean(), pars['e0'].var()**0.5])
    ltpar       = np.array([pars['lt'].mean(), pars['lt'].var()**0.5])
    print("    Mean core E0: {0:.1f}+-{1:.1f} pes".format(*e0par))
    print("    Mean core Lt: {0:.1f}+-{1:.1f} mus".format(*ltpar))


    maps.t_evol = pars_ec

    return

def check_drift_v_computation(drift_v      : np.ndarray ,
                              dv_maxFailed : float = 0.5,
                              **kwargs                  ) -> None:
    """
    Checks if drift velocity computation has failed in too many temporal
    bins. If that is the case, map computation will abort.

    Parameters
    ---------
    drift_v: np.ndarray
        Array of drift velocities per time bin
    dv_maxFailed: float
        Threshold to consider too many failing fits.
        (0.5 means that map creation will abort if more than 50% of fits fail)

    Returns
    ---------
    Nothing
    """

    n_failing_dv_fits = sum(np.isnan(drift_v))
    n_temp_bins       = len(drift_v)
    fail_fits_ratio   = n_failing_dv_fits/n_temp_bins
    message           = f'Drift_v fit failed in more than {dv_maxFailed*100}% of time bins.'
    message          += f'({n_failing_dv_fits}/{n_temp_bins})'

    check_if_values_in_interval(values          = np.array(fail_fits_ratio),
                                low_lim         = 0.                       ,
                                up_lim          = dv_maxFailed             ,
                                raising_message = message                  )

def compute_map(dst          : pd.DataFrame,
                run_number   : int,
                XYbins       : Tuple[int, int],
                nbins_z      : int,
                nbins_e      : int,
                z_range      : Tuple[float, float],
                e_range      : Tuple[float, float],
                chi2_range   : Tuple[float, float],
                lt_range     : Tuple[float, float],
                fit_type     : FitType,
                nmin         : int,
                maxFailed    : int,
                r_max        : float,
                x_range      : Tuple[float, float],
                y_range      : Tuple[float, float]) -> ASectorMap:

    maps = calculate_map (dst      = dst,
                          XYbins   = XYbins,
                          nbins_z  = nbins_z,
                          nbins_e  = nbins_e,
                          z_range  = z_range,
                          e_range  = e_range,
                          chi2_range = chi2_range,
                          lt_range = lt_range,
                          fit_type = fit_type,
                          nmin     = nmin,
                          x_range  = x_range,
                          y_range  = y_range)

    check_failed_fits(maps      = maps,
                      maxFailed = maxFailed,
                      nbins     = XYbins[0],
                      rmax      = r_max,
                      rfid      = r_max)
    regularized_maps = regularize_map(maps    = maps,
                                      x2range = chi2_range)

    no_peripheral    = remove_peripheral(regularized_maps,
                                         XYbins[0]       ,
                                         r_max           ,
                                         r_max)

    no_peripheral    = add_mapinfo(asm        = no_peripheral,
                                   xr         = x_range,
                                   yr         = y_range,
                                   nx         = XYbins[0],
                                   ny         = XYbins[1],
                                   run_number = int(run_number))

    return no_peripheral


def select_physical_events(dst              : pd.DataFrame,
                           lower            : Callable,
                           upper            : Callable,
                           eff_interval     : Tuple[float, float],
                           output           : pd.HDFStore,
                           diff_histo_params: dict) -> (pd.DataFrame, np.ndarray):
    n0   = dst.event.nunique()
    mask = in_range(dst.Zrms**2, lower(dst.DT), upper(dst.DT))
    dst  = dst[mask]
    n1   = dst.event.nunique()
    eff  = n1/n0
    compute_and_save_hist2d_as_pd(values    = (dst.DT, dst.Zrms),
                                  out_file  = output,
                                  hist_name = "DTrms2_vs_DT",
                                  **diff_histo_params)

    message  = f"Selection efficiency of diffusion band ({eff}) out of range:"
    message += f"({eff_interval[0]} - {eff_interval[1]})"
    check_if_values_in_interval(np.array(eff), *eff_interval, message)

    return dst[mask], mask


def recompute_npeaks(dst):
    events = dst.groupby("event")
    ns1    = events.s1_peak.nunique().values
    ns2    = events.s2_peak.nunique().values
    n      = events.s1_peak.count  ().values
    dst.loc[:, "nS1"] = np.repeat(ns1, n)
    dst.loc[:, "nS2"] = np.repeat(ns2, n)
    return dst


def apply_cuts(dst              : pd.DataFrame       ,
               S1_signal        : type_of_signal     ,
               nS1_eff_interval : Tuple[float, float],
               store_hist_s1    : pd.HDFStore        ,
               ns1_histo_params : dict               ,
               S2_signal        : type_of_signal     ,
               nS2_eff_interval : Tuple[float, float],
               store_hist_s2    : pd.HDFStore        ,
               ns2_histo_params : dict               ,
               ref_Z_histo      : pd.DataFrame       ,
               nsigmas_Zdst     : float              ,
               bootstrapmap     : ASectorMap         ,
               band_sel_params  : dict               ,
               ) -> (pd.DataFrame, masks_container):
    n0    = dst.event.nunique()
    mask1 = selection_nS_mask_and_checking(dst = dst                  ,
                                           column = S1_signal         ,
                                           interval = nS1_eff_interval,
                                           output_f = store_hist_s1   ,
                                           **ns1_histo_params         )
    nS1   = dst[mask1].event.nunique()
    print("    1 S1 cut efficiency within the expectations ({0:2.2f}%)".format(nS1/n0*100))
    mask2 = selection_nS_mask_and_checking(dst = dst                  ,
                                           column = S2_signal         ,
                                           interval = nS2_eff_interval,
                                           output_f = store_hist_s2   ,
                                           input_mask = mask1         ,
                                           **ns2_histo_params         )
    nS2   = dst[mask2].event.nunique()
    print("    1 S2 cut efficiency within the expectations ({0:2.2f}%)".format(nS2/nS1*100))
    check_Z_dst(Z_vect   = dst[mask2].Z,
                ref_hist = ref_Z_histo ,
                n_sigmas = nsigmas_Zdst)

    mask3 = band_selector_and_check(dst        = dst         ,
                                    boot_map   = bootstrapmap,
                                    input_mask = mask2       ,
                                    **band_sel_params        )
    nZb   = dst[mask3].event.nunique()
    print("    Z band cut efficiency within the expectations ({0:2.2f}%)".format(nZb/nS2*100))

    masks = masks_container(s1   = mask1,
                            s2   = mask2,
                            band = mask3)
    return dst[mask3], masks


def map_builder(config):

    print("Map builder starting...")
    print("Reading input files:")
    print("    Input dst folder   : {}".format(config.folder))
    print("    Input boostrap map : {}".format(config.file_bootstrap_map))
    print("    Input histogram map: {}".format(config.ref_Z_histogram['ref_histo_file']))


    dst, bootstrapmap, ref_histos  = load_data(input_path         = config.folder            ,
                                               input_dsts         = config.file_in           ,
                                               file_bootstrap_map = config.file_bootstrap_map,
                                               quality_ranges     = config.quality_ranges    ,
                                               **config.ref_Z_histogram                      )

    with pd.HDFStore(config.file_out_hists, "w", complib=str("zlib"), complevel=4) as store_hist:
        print("Checking the dst and appling 1S1, 1S2 and z-band selections:")

        nev_before = dst.event.nunique()
        print("    Number of events before any selection: {0}".format(nev_before))
        check_rate_and_hist(times      = dst.time         ,
                            output_f   = store_hist       ,
                            name_table = "rate_before_sel",
                            n_dev      = config.n_dev_rate,
                            **config.rate_histo_params    )

        dst_phys, mask = select_physical_events(dst,
                                                config.diff_band_lower    ,
                                                config.diff_band_upper    ,
                                                (config.diff_band_eff_min,
                                                 config.diff_band_eff_max),
                                                store_hist                ,
                                                config.diff_histo_params  )
        dst_phys = recompute_npeaks(dst_phys)

        nev_phys = dst_phys.event.nunique()
        ratio    = nev_phys / nev_before * 100
        print("    Number of physical events before cuts: {0} ({1:2.2f}%)".format(nev_phys, ratio))

        dst_passed_cut, masks = apply_cuts(dst       = dst_phys               ,
                                    S1_signal        = type_of_signal.nS1     ,
                                    nS1_eff_interval = (config.nS1_eff_min    ,
                                                        config.nS1_eff_max)   ,
                                    store_hist_s1    = store_hist             ,
                                    ns1_histo_params = config.ns1_histo_params,
                                    S2_signal        = type_of_signal.nS2     ,
                                    nS2_eff_interval = (config.nS2_eff_min    ,
                                                        config.nS2_eff_max)   ,
                                    store_hist_s2    = store_hist             ,
                                    ns2_histo_params = config.ns2_histo_params,
                                    nsigmas_Zdst     = config.nsigmas_Zdst    ,
                                    ref_Z_histo      = ref_histos.Z_dist_hist ,
                                    bootstrapmap     = bootstrapmap           ,
                                    band_sel_params  = config.band_sel_params )

        check_rate_and_hist(times      = dst_passed_cut.time,
                            output_f   = store_hist         ,
                            name_table = "rate_after_sel"   ,
                            n_dev      = config.n_dev_rate  ,
                            **config.rate_histo_params      )

        nev_after = dst_passed_cut.event.nunique()
        ratio     = nev_after/nev_phys*100
        print("    Number of events passing the cuts: {0} ({1:2.2f}%)".format(nev_after, ratio))


    print("Map computation:")
    number_of_bins = get_binning_auto(nevt_sel                = dst_passed_cut.event.nunique()  ,
                                      thr_events_for_map_bins = config.thr_evts_for_sel_map_bins,
                                      n_bins                  = config.default_n_bins           )

    print("    Number of bins: {0}x{0}".format(number_of_bins))

    final_map      = compute_map(dst        = dst_passed_cut   ,
                                 run_number = config.run_number,
                                 XYbins     = (number_of_bins  ,
                                               number_of_bins) ,
                                 fit_type   = FitType.unbined  ,
                                 **config.map_params           )

    add_krevol(maps  = final_map,
               dst   = dst_phys,
               masks_cuts = masks,
               XYbins     = (number_of_bins,
                             number_of_bins),
               **config.krevol_params)

    check_drift_v_computation(final_map.t_evol.dv,
                              **config.map_params)

    write_complete_maps(asm      = final_map          ,
                        filename = config.file_out_map)
    print("Map successfully computed and saved in : {0}".format(config.file_out_map))
    print("Control histograms saved in            : {0}".format(config.file_out_hists))
