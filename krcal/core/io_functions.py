import numpy  as np
import pandas as pd
from   invisible_cities.core.core_functions import shift_to_bin_centers
from   typing         import Tuple
from . kr_types       import ASectorMap

def write_complete_maps(asm      : ASectorMap,
                        filename : str       )->None:

    asm.chi2.to_hdf(filename, key='chi2', mode='w')
    asm.e0  .to_hdf(filename, key='e0'  , mode='a')
    asm.e0u .to_hdf(filename, key='e0u' , mode='a')
    asm.lt  .to_hdf(filename, key='lt'  , mode='a')
    asm.ltu .to_hdf(filename, key='ltu' , mode='a')
    if hasattr(asm, 'mapinfo'):
        asm.mapinfo.to_hdf(filename, key='mapinfo'       , mode='a')
    if hasattr(asm, 't_evol'):
        asm.t_evol .to_hdf(filename, key='time_evolution', mode='a')


def compute_and_save_hist_as_pd(values     : np.array           ,
                                out_file   : pd.HDFStore        ,
                                hist_name  : str                ,
                                n_bins     : int                ,
                                range_hist : Tuple[float, float],
                                norm       : bool = False       )->None:
    """
    Computes 1d-histogram and saves it in a file.
    The name of the table inside the file must be provided.
    Parameters
    ----------
    values : np.array
        Array with values to be plotted.
    out_file: pd.HDFStore
        File where histogram will be saved.
    hist_name: string
        Name of the pd.Dataframe to contain the histogram.
    n_bins: int
        Number of bins to make the histogram.
    range_hist: length-2 tuple (optional)
        Range of the histogram.
    norm: bool
        If True, histogram will be normalized.
    """
    n, b = np.histogram(values, bins = n_bins,
                        range = range_hist,
                        density = norm)
    table = pd.DataFrame({'entries': n,
                          'magnitude': shift_to_bin_centers(b)})
    out_file.put(hist_name, table, format='table', data_columns=True)

    return

def compute_and_save_hist2d_as_pd(values     : np.array           ,
                                  out_file   : pd.HDFStore        ,
                                  hist_name  : str                ,
                                  n_bins     : int                ,
                                  range_hist : Tuple[float, float],
                                  norm       : bool = False       )->None:
    """
    Computes 1d-histogram and saves it in a file.
    The name of the table inside the file must be provided.
    Parameters
    ----------
    values : np.array
        Array with values to be plotted.
    out_file: pd.HDFStore
        File where histogram will be saved.
    hist_name: string
        Name of the pd.Dataframe to contain the histogram.
    n_bins: int
        Number of bins to make the histogram.
    range_hist: length-2 tuple (optional)
        Range of the histogram.
    norm: bool
        If True, histogram will be normalized.
    """
    n, bx, by = np.histogram2d(*values, bins = n_bins,
                               range = range_hist,
                               density = norm)
    bx = shift_to_bin_centers(np.unique(bx))
    by = shift_to_bin_centers(np.unique(by))
    bx, by = map(np.ravel, np.meshgrid(bx, by, indexing="ij"))
    table = pd.DataFrame({'entries': n.flatten(),
                          'magnitude_x': bx,
                          'magnitude_y': by,
                          })
    out_file.put(hist_name, table, format='table', data_columns=True)

    return
