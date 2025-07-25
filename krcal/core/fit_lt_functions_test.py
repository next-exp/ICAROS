"""
Tests for fit_functions
"""

import numpy as np

from pytest                import mark
from pytest                import approx
from pytest                import warns

from invisible_cities.core               import fit_functions as fitf
from invisible_cities.core.testing_utils import float_arrays

from . testing_utils       import energy_lt_experiment
from . testing_utils       import energy_lt_experiments
from . testing_utils       import fit_lifetime_experiments
from . fit_functions       import expo_seed
from . stat_functions      import mean_and_std
from . fit_lt_functions    import fit_lifetime_profile
from . fit_lt_functions    import fit_lifetime_unbined
from . fit_lt_functions    import pars_from_fcs
from . fit_lt_functions    import lt_params_from_fcs
from . kr_types            import FitType
from . kr_types            import FitResult
from . kr_types            import FitCollection


def test_lt_profile_yields_same_result_expo_fit():

    Nevt  = int(1e5)
    e0 = 1e+4 # pes
    std = 0.05 * e0
    lt = 2000 # lifetime in mus
    nbins_z = 12
    range_z = (1, 500)
    z, es = energy_lt_experiment(Nevt, e0, lt, std)

    x, y, yu     = fitf.profileX(z, es, nbins_z, range_z)
    valid_points = ~np.isnan(yu)

    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]
    seed = expo_seed(x, y)
    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)

    par  = np.array(f.values)
    err  = np.array(f.errors)
    e0   = par[0]
    lt   = - par[1]
    e0_u = err[0]
    lt_u = err[1]

    _, _,  fr = fit_lifetime_profile(z, es, nbins_z, range_z)
    assert e0   == approx(fr.par[0],  rel=0.05)
    assert lt   == approx(fr.par[1],  rel=0.05)
    assert e0_u == approx(fr.err[0],  rel=0.05)
    assert lt_u == approx(fr.err[1],  rel=0.05)

# @flaky(max_runs=5, min_passes=2)
# @given(floats(min_value = 0.01,
#               max_value = 0.1),
#        floats(min_value = 100,
#               max_value = 10000))
# @settings(max_examples=10)
# def test_lt_profile_yields_compatible_results_with_unbined_fit():
#     sigma = 0.1
#     lt = 100
#     Nevt  = int(1e4)
#     e0 = 1e+4 # pes
#     std = sigma * e0
#     lt = 2000 # lifetime in mus
#     nbins_z = 12
#     range_z = (1, 500)
#     z, es = energy_lt_experiment(Nevt, e0, lt, std)

#     _, _,  frp = fit_lifetime_profile(z, es, nbins_z, range_z)
#     _, _,  fru = fit_lifetime_unbined(z, es, nbins_z, range_z)

#     assert frp.par[0] == approx(fru.par[0],  rel=0.2)
#     assert frp.par[1] == approx(fru.par[1],  rel=0.2)
#     assert frp.err[0] == approx(fru.err[0],  rel=0.2)
#     assert frp.err[1] == approx(fru.err[1],  rel=0.5)
#     assert frp.chi2   == approx(fru.chi2,    rel=0.5)


@mark.parametrize("length error_type".split(),
                  ((0, "Type"       ),
                   (1, "LinAlgError"),
                   (2, "LinAlgError")))
def test_fit_lifetime_unbined_warns_with_insufficient_data_points(caplog, length, error_type):
    fit_lifetime_unbined(np.zeros(length), np.ones(length), 10, (0, 10))
    assert f"{error_type} error found in fit_lifetime_unbined: not enough events for fit" in caplog.text


def test_fit_lifetime_experiments_yield_good_pars_and_pulls():
    mexperiments = 1e+3
    nsample      = 1e+3
    e0 = 1e+4 # pes
    std = 0.05 * e0
    lt = 2000 # lifetime in mus

    zs, es = energy_lt_experiments(mexperiments, nsample, e0, lt, std)
    fcp = fit_lifetime_experiments(zs, es, nbins_z=12, nbins_e = 50,
                                   range_z = (1, 500), range_e = (7e+3, 11e+3),
                                   fit=FitType.profile)
    e0s, ue0s, lts,ults, chi2p = lt_params_from_fcs(fcp)

    p_e0, p_e0u = mean_and_std(e0s,   range_ =(e0 - 100, e0 + 100))
    p_lt, p_ltu = mean_and_std(lts,   range_ =(lt - 150, lt + 150))
    p_c2, p_c2u = mean_and_std(chi2p, range_ =(0, 2))
    assert p_e0   == approx(e0,  rel=0.01)
    assert p_lt   == approx(lt,  rel=0.01)
    assert p_c2   == approx(1,   rel=0.5)
    #assert p_c2u  == approx(0.2, rel=0.5)

    p_mu, p_std = mean_and_std((e0s-e0) / ue0s, range_ =(-5,5))
    assert p_mu   == approx(0,  abs=0.1)
    assert p_std  == approx(1,  rel=0.1)

    p_mu, p_std = mean_and_std((lts-lt) / ults, range_ =(-5,5))
    assert p_mu   == approx(0,  abs=0.1)
    assert p_std  == approx(1,  rel=0.1)
    p_c2, p_c2u = mean_and_std(chi2p, range_ =(0, 2))
    zs, es = energy_lt_experiments(mexperiments, nsample, e0, lt, std)
    fcp = fit_lifetime_experiments(zs, es, nbins_z=12, nbins_e = 50,
                                   range_z = (1, 500), range_e = (7e+3, 11e+3),
                                   fit=FitType.unbined)
    e0s, ue0s, lts,ults, chi2p = lt_params_from_fcs(fcp)

    p_e0, p_e0u = mean_and_std(e0s, range_ =(e0 - 100, e0 + 100))
    p_lt, p_ltu = mean_and_std(lts, range_ =(lt - 150, lt + 150))


    assert p_e0   == approx(e0,  rel=0.01)
    assert p_lt   == approx(lt,  rel=0.01)
    p_mu, p_std = mean_and_std((e0s-e0) / ue0s, range_ =(-5,5))
    assert p_mu   <= 0  # the pull is biased
    assert p_std  == approx(1,  rel=0.2)

    p_mu, p_std = mean_and_std((lts-lt) / ults, range_ =(-5,5))
    assert p_mu   <= 0  # the pull is biased
    assert p_std  == approx(1,  rel=0.2)
    assert p_c2   == approx(1,   rel=0.5)


def test_pars_from_fcs_warns_for_invalid_fits(caplog):
    fitcol = FitCollection(fp = None,
                           hp = None,
                           fr = FitResult(par = None, err = None, chi2 = None, valid = False))

    with warns(UserWarning, match="fit did not succeed, returning NaN"):
        pars_from_fcs([fitcol])


def test_fit_lt_unbined_uses_correct_zrange():
    """
    This tests checks that fit_lifetime_unbined
    function uses z_range to fit properly.
    """
    Nevt  = 1e5
    e0    = 12800
    lt    = 10000
    std   = 0.05 * e0
    max_z = 500
    z, es = energy_lt_experiment(Nevt, e0, lt, std, zmax=max_z)

    # we add now random values out of the z_range,
    # so they shouldn't affect the fit
    add_z  = np.linspace(550, 1000, 1000)
    add_es = add_z
    z_new  = np.append(z , add_z )
    es_new = np.append(es, add_es)

    z_range = (0, max_z)
    nbins_z = 50

    _, _, pars     , _ = fit_lifetime_unbined(z    , es    , nbins_z, z_range);
    _, _, pars_new , _ = fit_lifetime_unbined(z_new, es_new, nbins_z, z_range);
    lt , e0  = pars.par
    ltu, e0u = pars.err
    lt_new , e0_new  = pars_new.par
    ltu_new, e0u_new = pars_new.err

    assert lt  == lt_new
    assert ltu == ltu_new
    assert e0  == e0_new
    assert e0u == e0u_new
