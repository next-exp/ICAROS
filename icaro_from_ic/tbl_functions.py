import re
import numpy as np
import tables as tb
import pandas as pd

from argparse import Namespace

from ..evm.ic_cotainers import SensorParams
from ..evm.event_model  import MCInfo
from ..core.exceptions  import NoParticleInfoInFile


def read_FEE_table(fee_t):
    """Read the FEE table and return a PD Series for the simulation
    parameters and a PD series for the values of the capacitors used
    in the simulation.
    """

    fa = fee_t.read()

    F = pd.Series([fa[0][ 0], fa[0][ 1], fa[0][ 2], fa[0][ 3], fa[0][ 4],
                   fa[0][ 5], fa[0][ 6], fa[0][ 7], fa[0][ 8], fa[0][ 9],
                   fa[0][10], fa[0][11], fa[0][12], fa[0][13], fa[0][14],
                   fa[0][15], fa[0][16], fa[0][17]],
                  index=["OFFSET", "CEILING", "PMT_GAIN", "FEE_GAIN", "R1",
                         "C1", "C2", "ZIN", "DAQ_GAIN", "NBITS", "LSB",
                         "NOISE_I", "NOISE_DAQ", "t_sample", "f_sample",
                         "f_mc", "f_LPF1", "f_LPF2"])
    FEE = Namespace()
    FEE.fee_param     = F
    FEE.coeff_c       = np.array(fa[0][18], dtype=np.double)
    FEE.coeff_blr     = np.array(fa[0][19], dtype=np.double)
    FEE.adc_to_pes    = np.array(fa[0][20], dtype=np.double)
    FEE.pmt_noise_rms = np.array(fa[0][21], dtype=np.double)
    return FEE


def table_from_params(table, params):
    row = table.row
    for param in table.colnames:
        row[param] = params[param]
    row.append()
    table.flush()


def table_to_params(table):
    params = {}
    for param in table.colnames:
        params[param] = table[0][param]
    return params


def get_rwf_vectors(h5in):
    """Return the most relevant fields stored in a raw data file.

    Parameters
    ----------
    h5f : tb.File
        (Open) hdf5 file.

    Returns
    -------
    NEVT   : number of events in array
    pmtrwf : tb.EArray
        RWF array for PMTs
    pmtblr : tb.EArray
        BLR array for PMTs
    sipmrwf : tb.EArray
        RWF array for PMTs

    """
    pmtrwf, pmtblr, sipmrwf  = get_vectors(h5in)
    NEVT_pmt , _, _          = pmtrwf .shape
    NEVT_simp, _, _          = sipmrwf.shape

    #assert NEVT_simp == NEVT_pmt
    return max(NEVT_pmt, NEVT_simp), pmtrwf, sipmrwf, pmtblr


def get_rd_vectors(h5in):
    "Return MC RD vectors and sensor data."
    pmtrd            = h5in.root.pmtrd
    sipmrd           = h5in.root.sipmrd
    NEVT_pmt , _, _  = pmtrd .shape
    NEVT_simp, _, _  = sipmrd.shape
    assert NEVT_simp == NEVT_pmt

    return NEVT_pmt, pmtrd, sipmrd


def get_sensor_params_from_vectors(pmtrwf, sipmrwf):
    _, NPMT,   PMTWL   = pmtrwf .shape
    _, NSIPM, SIPMWL   = sipmrwf.shape
    return SensorParams(NPMT, PMTWL, NSIPM, SIPMWL)


def get_sensor_params(filename):
    with tb.open_file(filename, "r") as h5in:
        _, pmtrwf, sipmrwf, _ = get_rwf_vectors(h5in)
        return get_sensor_params_from_vectors(pmtrwf, sipmrwf)


def get_nof_events(table, column_name="evt_number"):
    """Find number of events in table by asking number of different values
    in column.

    Parameters
    ----------
    table : tb.Table
        Table to be read.
    column_name : string
        Name of the column with a unique value for each event.

    Returns
    -------
    nevt : int
        Number of events in table.

    """
    return len(set(table.read(field=column_name)))


def get_event_numbers_and_timestamps_from_file_name(filename):
    with tb.open_file(filename, "r") as h5in:
        return get_event_numbers_and_timestamps_from_file(h5in)

def get_event_numbers_and_timestamps_from_file(file):
    event_numbers = file.root.Run.events.cols.evt_number[:]
    timestamps    = file.root.Run.events.cols.timestamp [:]
    return event_numbers, timestamps


def event_number_from_input_file_name_hash(filename):
    file_base_name = filename.split('/')[-1]
    base_hash = hash(file_base_name)
    # Something, somewhere, is preventing us from using the full
    # 64 bit space, and limiting us to 32 bits. TODO find and
    # eliminate it.
    limited_hash = base_hash % int(1e9)
    return limited_hash
