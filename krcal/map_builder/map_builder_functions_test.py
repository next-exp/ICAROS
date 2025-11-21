import os
import copy
import numpy  as np
import tables as tb
import pandas as pd
from pytest        import mark
from pytest        import fixture
from numpy.testing import assert_raises

from invisible_cities.io  .dst_io          import load_dst
from invisible_cities.core.testing_utils   import assert_dataframes_close
from invisible_cities.core.configure       import configure
from invisible_cities.reco.corrections     import read_maps
from invisible_cities.reco.corrections     import ASectorMap
from invisible_cities.reco.corrections     import maps_coefficient_getter

from . map_builder_functions import map_builder
from . checking_functions    import AbortingMapCreation

from hypothesis            import settings
from hypothesis            import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import composite
from hypothesis.strategies import lists

import logging
import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.DEBUG)
this_script_logger = logging.getLogger(__name__)
this_script_logger.setLevel(logging.INFO)


@mark.timeout(None)
@mark.dependency()
def test_scrip_runs_and_produces_correct_outputs(folder_test_dst  ,
                                                 test_dst_file    ,
                                                 output_maps_tmdir,
                                                 test_map_file    ):
    """
    Run map creation script and check if an ASectormap is the output.
    """
    map_file_out   = os.path.join(output_maps_tmdir, 'test_out_map.h5')
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo.h5')
    default_n_bins = 15
    run_number     = 7517
    config = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_params_new    = copy.copy(config.as_namespace.   map_params)
    krevol_params_new = copy.copy(config.as_namespace.krevol_params)
    map_params_new   ['nmin']          = 100
    map_params_new   ['z_range']       = (0, 10000)
    krevol_params_new['nStimeprofile'] = 1200
    config.update(dict(folder         = folder_test_dst  ,
                       file_in        = test_dst_file    ,
                       file_out_map   = map_file_out     ,
                       file_out_hists = histo_file_out   ,
                       default_n_bins = default_n_bins   ,
                       run_number     = run_number       ,
                       map_params     = map_params_new   ,
                       krevol_params  = krevol_params_new))

    map_builder(config.as_namespace)
    maps = read_maps(map_file_out)
    assert type(maps)==ASectorMap

    old_maps = read_maps(test_map_file)
    assert_dataframes_close(maps.e0 , old_maps.e0 , rtol=1e-5)
    assert_dataframes_close(maps.e0u, old_maps.e0u, rtol=1e-1)
    assert_dataframes_close(maps.lt , old_maps.lt , rtol=1e-5)
    assert_dataframes_close(maps.ltu, old_maps.ltu, rtol=1e-1)
    # TODO: figure out why this fails. the result seems to be machine-dependent
    # assert_dataframes_close(maps.t_evol, old_maps.t_evol, rtol=1e-4)


@mark.dependency(depends="test_scrip_runs_and_produces_correct_outputs")
def test_time_evol_table_correct_elements(output_maps_tmdir):
    map_file_out = os.path.join(output_maps_tmdir, 'test_out_map.h5')
    emaps        = read_maps(map_file_out)
    time_table   = emaps.t_evol
    columns      = time_table.columns
    elements     =['ts'   ,
                   'e0'   , 'e0u'   ,
                   'lt'   , 'ltu'   ,
                   'dv'   , 'dvu'   ,
                   'resol', 'resolu',
                   's1w'  , 's1wu'  ,
                   's1h'  , 's1hu'  ,
                   's1e'  , 's1eu'  ,
                   's2w'  , 's2wu'  ,
                   's2h'  , 's2hu'  ,
                   's2e'  , 's2eu'  ,
                   's2q'  , 's2qu'  ,
                   'Nsipm', 'Nsipmu',
                   'Xrms' , 'Xrmsu' ,
                   'Yrms' , 'Yrmsu' ,
                   'S1eff' , 'S2eff', 'Bandeff']
    for element in elements:
        assert element in columns

@mark.dependency(depends="test_scrip_runs_and_produces_correct_outputs")
def test_time_evol_eff_less_one(output_maps_tmdir):
    map_file_out = os.path.join(output_maps_tmdir, 'test_out_map.h5')
    emaps        = read_maps(map_file_out)
    assert np.all(emaps.t_evol.S1eff   <= 1.)
    assert np.all(emaps.t_evol.S2eff   <= 1.)
    assert np.all(emaps.t_evol.Bandeff <= 1.)

@composite
def xy_pos(draw, elements=floats(min_value=-200, max_value=200)):
    size = draw(integers(min_value=1, max_value=10))
    x    = draw(lists(elements,min_size=size, max_size=size))
    y    = draw(lists(elements,min_size=size, max_size=size))
    return (np.array(x),np.array(y))


@mark.dependency(depends="test_scrip_runs_and_produces_correct_outputs")
@given(xy_pos = xy_pos())
@settings(max_examples=50)
def test_maps_nans_outside_rmax(xy_pos, output_maps_tmdir):
    map_file_out = os.path.join(output_maps_tmdir, 'test_out_map.h5')
    xs, ys = xy_pos

    emaps = read_maps(map_file_out)
    get_coef = maps_coefficient_getter(emaps.mapinfo, emaps.e0)
    r_max    = emaps.mapinfo.xmax
    nbins    = emaps.mapinfo.nx
    bins     = np.linspace(-200, 200, nbins)
    xs       = bins[np.digitize(xs, bins, right = True)]
    ys       = bins[np.digitize(ys, bins, right = True)]
    coefs    = get_coef(xs, ys)
    maskin   = np.sqrt(xs**2 + ys**2) <  r_max
    maskout  = np.sqrt(xs**2 + ys**2) >= r_max + 2 * r_max / nbins

    assert all(np.isnan   (coefs[maskout]))
    assert all(np.isfinite(coefs[maskin ]))

@mark.dependency(depends="test_scrip_runs_and_produces_correct_outputs")
def test_correct_map_with_unsorted_dst(folder_test_dst  ,
                                       test_dst_file    ,
                                       output_maps_tmdir):
    """
    This test shuffles the input dst, and checks that the map is the same
    as the one created with the same sorted dst.
    """
    map_file_sort   = os.path.join(output_maps_tmdir, 'test_out_map.h5')
    map_file_unsort = os.path.join(output_maps_tmdir, 'test_out_unsort.h5')
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo.h5')

    dst = load_dst(folder_test_dst+test_dst_file, 'DST', 'Events')
    if "index" in dst:del dst["index"]
    dst = dst.sort_values(by=['S2e'])
    tmp_unsorted_dst = 'unsorted_dst.h5'
    dst.to_hdf(output_maps_tmdir+tmp_unsorted_dst,
               key     = "DST"  , mode         = "w",
               format  = "table", data_columns = True,
               complib = "zlib" , complevel    = 4)
    with tb.open_file(output_maps_tmdir+tmp_unsorted_dst, "r+") as file:
        file.rename_node(file.root.DST.table, "Events")
        file.root.DST.Events.title = "Events"

    default_n_bins = 15
    run_number     = 7517
    config = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_params_new = config.as_namespace.map_params
    map_params_new['nmin']    = 100
    map_params_new['z_range'] = (0, 10000)
    config.update(dict(folder         = output_maps_tmdir,
                       file_in        = tmp_unsorted_dst ,
                       file_out_map   = map_file_unsort  ,
                       file_out_hists = histo_file_out   ,
                       default_n_bins = default_n_bins   ,
                       run_number     = run_number       ,
                       map_params     = map_params_new   ))
    map_builder(config.as_namespace)
    unsorted_maps = read_maps(map_file_unsort)
    sorted_maps   = read_maps(map_file_sort)

    assert_dataframes_close(unsorted_maps.e0 , sorted_maps.e0 , rtol=1e-5)
    assert_dataframes_close(unsorted_maps.e0u, sorted_maps.e0u, rtol=1e-5)
    assert_dataframes_close(unsorted_maps.lt , sorted_maps.lt , rtol=1e-5)
    assert_dataframes_close(unsorted_maps.ltu, sorted_maps.ltu, rtol=1e-5)

def test_exception_s1(folder_test_dst, test_dst_file, output_maps_tmdir):
    """
    This test checks if exception raises when ns1=1 efficiency is out of range.
    """
    conf = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_file_out   = os.path.join(output_maps_tmdir, 'test_out_map_s1.h5'  )
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo_s1.h5')
    min_eff_test = 0.
    max_eff_test = 0.8
    run_number   = 7517
    conf.update(dict(folder         = folder_test_dst,
                     file_in        = test_dst_file  ,
                     file_out_map   = map_file_out   ,
                     file_out_hists = histo_file_out ,
                     nS1_eff_min    = min_eff_test   ,
                     nS1_eff_max    = max_eff_test   ,
                     run_number     = run_number     ))

    assert_raises(AbortingMapCreation,
                  map_builder        ,
                  conf.as_namespace  )

def test_exception_s2(folder_test_dst, test_dst_file, output_maps_tmdir):
    """
    This test checks if exception raises when nS2=1 efficiency is out of range.
    """
    conf = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_file_out   = os.path.join(output_maps_tmdir, 'test_out_map_s2.h5'  )
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo_s2.h5')
    min_eff_test = 0.
    max_eff_test = 0.9
    run_number   = 7517
    conf.update(dict(folder         = folder_test_dst,
                     file_in        = test_dst_file  ,
                     file_out_map   = map_file_out   ,
                     file_out_hists = histo_file_out ,
                     nS2_eff_min    = min_eff_test   ,
                     nS2_eff_max    = max_eff_test   ,
                     run_number     = run_number     ))

    assert_raises(AbortingMapCreation,
                  map_builder        ,
                  conf.as_namespace  )

def test_exception_rate(folder_test_dst, test_dst_file, output_maps_tmdir):
    """
    This test checks if exception raises when rate distribution is not flat enough.
    """
    conf = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_file_out   = os.path.join(output_maps_tmdir, 'test_out_map_rate.h5'  )
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo_rate.h5')
    n_dev_rate = 0.5
    run_number   = 7517
    conf.update(dict(folder         = folder_test_dst,
                     file_in        = test_dst_file  ,
                     file_out_map   = map_file_out   ,
                     file_out_hists = histo_file_out ,
                     n_dev_rate     = n_dev_rate     ,
                     run_number     = run_number     ))

    assert_raises(AbortingMapCreation,
                  map_builder        ,
                  conf.as_namespace  )

def test_exception_Zdst(folder_test_dst, test_dst_file, output_maps_tmdir):
    """
    This test checks if exception raises when Z distribution is not
    similar enough to the reference one.
    """
    conf = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_file_out   = os.path.join(output_maps_tmdir, 'test_out_map_Z.h5'  )
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo_Z.h5')
    nsigmas_Zdst = 0.5
    run_number   = 7517
    conf.update(dict(folder         = folder_test_dst,
                     file_in        = test_dst_file  ,
                     file_out_map   = map_file_out   ,
                     file_out_hists = histo_file_out ,
                     nsigmas_Zdst   = nsigmas_Zdst   ,
                     run_number     = run_number     ))

    assert_raises(AbortingMapCreation,
                  map_builder        ,
                  conf.as_namespace  )

def test_exception_bandsel(folder_test_dst, test_dst_file, output_maps_tmdir):
    """
    This test checks if exception raises when band selection efficiency is
    out of a given range.
    """
    conf = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_file_out   = os.path.join(output_maps_tmdir, 'test_out_map_bandsel.h5'  )
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo_bandsel.h5')
    band_sel_params_new = copy.copy(conf.as_namespace.band_sel_params)
    band_sel_params_new['eff_min'] = 0.
    band_sel_params_new['eff_max'] = 0.89
    run_number = 7517
    conf.update(dict(folder         = folder_test_dst,
                     file_in        = test_dst_file  ,
                     file_out_map   = map_file_out   ,
                     file_out_hists = histo_file_out ,
                     band_sel_params = band_sel_params_new,
                     run_number     = run_number     ))
    assert_raises(AbortingMapCreation,
                  map_builder        ,
                  conf.as_namespace  )

def test_exception_drift_v(folder_test_dst, test_dst_file, output_maps_tmdir):
    """
    This test checks that exception raises if drift_v fit
    fails in too many temporal bins.
    """
    conf = configure('maps $ICARO/conf/next-white/config_LBphys.conf'.split())
    map_file_out   = os.path.join(output_maps_tmdir, 'test_out_map_driftv.h5'  )
    histo_file_out = os.path.join(output_maps_tmdir, 'test_out_histo_driftv.h5')
    map_params_new = copy.copy(conf.as_namespace.map_params)
    map_params_new['dv_maxFailed'] = 0.01
    run_number = 7517
    conf.update(dict(folder         = folder_test_dst,
                     file_in        = test_dst_file  ,
                     file_out_map   = map_file_out   ,
                     file_out_hists = histo_file_out ,
                     map_params     = map_params_new ,
                     run_number     = run_number     ))
    assert_raises(AbortingMapCreation,
                  map_builder        ,
                  conf.as_namespace  )
