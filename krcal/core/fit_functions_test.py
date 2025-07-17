"""
Tests for fit_functions
"""

import numpy                                as np

from   numpy.testing                        import assert_allclose

from   pytest                               import approx
from   pytest                               import raises
from   pytest                               import mark

from  flaky                                 import flaky

from   hypothesis                           import given
from   hypothesis.strategies                import floats
from   hypothesis.strategies                import builds

import invisible_cities.database.load_db    as     DB

from invisible_cities.core                  import  fit_functions as fitf

from invisible_cities.core.testing_utils    import float_arrays
from invisible_cities.core.testing_utils    import random_length_float_arrays

from . fit_functions                        import sigmoid
from . fit_functions                        import compute_drift_v
from . fit_functions                        import chi2f
from . fit_functions                        import sigmoid
from . fit_functions                        import compute_drift_v
from . fit_functions                        import relative_errors
from . fit_functions                        import to_relative
from . fit_functions                        import fit_profile_1d_expo
from . fit_functions                        import fit_slices_2d_gauss
from . fit_functions                        import fit_slices_2d_expo

from . kr_types                             import Measurement
sensible_floats          = floats(-1e4, +1e4)
fractions                = floats(   0,    1)
sensible_arrays_variable = random_length_float_arrays(1, 10, min_value=-1e4, max_value=1e4)
sensible_arrays_fixed    =               float_arrays(    5, min_value=-1e4, max_value=1e4)
measurements             = builds(Measurement, sensible_arrays_fixed, sensible_arrays_fixed.map(abs))


def test_get_chi2f_when_data_equals_error_and_fit_equals_zero():
    Nevt  = int(1e6)
    xdata = np.zeros(Nevt) # Dummy value, not needed
    ydata = np.random.uniform(1, 100, Nevt)
    errs  = ydata
    f     = lambda x: np.zeros_like(x)
    chi2 = chi2f(f, 0, xdata, ydata, errs)
    assert chi2   == approx(1  , rel=1e-3)



def test_get_chi2f_when_data_equals_fit():
    Nevt  = int(1e6)
    xdata = np.zeros(Nevt) # Dummy value, not needed
    ydata = np.random.uniform(1, 100, Nevt)
    f     = lambda x: ydata
    errs  = ydata**0.5 # Dummy value, not needed
    chi2  = chi2f(f, 0, ydata, ydata, errs)

    assert chi2   == approx(0., rel=1e-3)


def test_chi2f_str_line():
    def line(x, m, n):
        return m * x + n
    y  = np.array([ 9.108e3, 10.34e3,   1.52387e5,   1.6202e5])
    ey = np.array([ 3.17   , 13.5   ,  70        ,  21       ])
    x  = np.array([29.7    , 33.8   , 481        , 511       ])

    fit = fitf.fit(line, x, y, seed=(1,1), sigma=ey)
    f = lambda x: line(x, *fit.values)
    assert chi2f(f, 2, x, y, ey) == approx(14, rel=1e-02)


@mark.parametrize("percentual", (True, False))
@given(values = sensible_arrays_variable.filter(lambda x: np.all(x != 0)),
       error  = fractions)
def test_relative_errors(percentual, values, error):
    abs_errors = np.abs(values) * error
    rel_errors = relative_errors(values, abs_errors, default=0, percentual=percentual)
    assert_allclose(rel_errors, error * 100 if percentual else error)


@mark.parametrize("percentual", (True, False))
@given(values  = sensible_arrays_variable,
       error   = fractions,
       default = sensible_floats)
def test_relative_errors_default(percentual, values, error, default):
    if not any(values == 0): return

    where_zero = values == 0

    abs_errors = np.abs(values) * error
    rel_errors = relative_errors(values, abs_errors, default=default, percentual=percentual)

    assert_allclose(rel_errors[ where_zero], default)
    assert_allclose(rel_errors[~where_zero], error * 100 if percentual else error)


@given(measurements)
def test_to_relative(measurement):
    relative = to_relative(measurement, default=0)
    zeros    = measurement.value == 0
    assert          measurement.value               == approx(relative.value)
    assert np.all  (relative   .uncertainty[ zeros] == 0)
    assert_allclose(relative   .uncertainty[~zeros] * np.abs(relative.value[~zeros]), measurement.uncertainty[~zeros])


def test_fit_profile_1d_expo_fixed_example():
    xdata = np.linspace(0, 10, 100)
    ydata = np.pi * np.exp(-xdata / 100)
    nbins = 5

    expected_chi2   = 0.019290474747231997
    expected_pvalue = 0.9963609538067274
    expected_coeffs = 3.14274919, -99.50545093
    expected_errors = 0.00380451,   2.07172415

    f = fit_profile_1d_expo(xdata, ydata, nbins)

    assert f.chi2 == approx(expected_chi2)
    assert_allclose(expected_coeffs, f.values)
    assert_allclose(expected_errors, f.errors, rtol=1e-5)



def test_fit_profile_1d_expo_raises_valueerror_when_data_is_bad():
    xdata = np.linspace(0, 10, 10)
    ydata = xdata
    nbins = 10
    # With only one data point per bin it should fail

    with raises(ValueError):
        fit_profile_1d_expo(xdata, ydata, nbins)


def test_fit_slices_2d_gauss_fixed_example():
    xdata = [0.5] * 30 + [1.5] * 30 + [0.3] * 30 + [1.8] * 30
    ydata = [2.5] * 30 + [8.5] * 30 + [6.5] * 30 + [3.5] * 30
    zdata = [#np.random.normal(1.00, 0.27, 30)
             0.83151222, 0.89478203, 0.95122575, 1.19681632, 1.21592953,
             0.59258251, 0.85634256, 1.06269388, 1.49981856, 1.19565073,
             0.83236772, 0.97424502, 0.96496171, 1.25245324, 0.94648805,
             0.98816722, 1.01072126, 1.45467232, 1.24497004, 1.21416081,
             1.25510415, 1.57439475, 1.23291016, 1.3162867 , 0.7462562 ,
             1.14176433, 1.05873601, 0.97729419, 0.85749642, 0.81017485,
             #np.random.normal(0.90, 0.25, 30)
             0.74149347, 0.91145939, 0.78703531, 0.74849776, 0.9140171 ,
             0.99268373, 0.9304295 , 1.11410189, 0.92293438, 0.86072217,
             1.18521016, 1.14887795, 1.27914219, 0.68677073, 1.18970609,
             0.77015265, 0.98338957, 0.58789884, 1.05552157, 1.25197916,
             1.1630278 , 1.55823302, 0.96127485, 1.04464286, 0.05388236,
             0.06424938, 0.79830302, 0.98181607, 1.15296395, 1.18869654,
             #np.random.normal(1.25, 0.29, 30)
             1.64151947, 1.73622749, 1.07960685, 1.14927624, 0.83441397,
             1.50708668, 1.36977956, 1.11066818, 1.5014107 , 0.78468776,
             1.35652132, 1.561734  , 1.18813624, 1.224365  , 1.01280963,
             1.00918479, 1.30971066, 1.87880162, 0.82910072, 1.54668927,
             1.26205772, 1.25692053, 0.81738151, 1.6475947 , 0.87525242,
             1.27261835, 1.1924033 , 1.83866199, 1.57444608, 1.30138609,
             #np.random.normal(1.05, 0.24, 30)
             0.63546908, 0.77065478, 0.90705201, 0.75402492, 0.90956183,
             1.22629662, 0.60231432, 0.82457011, 0.88161914, 1.36571455,
             1.0500734 , 0.96864934, 1.3546291 , 0.80908968, 0.83510295,
             0.96886958, 1.29375071, 1.38113243, 1.23969477, 0.73520622,
             0.68516141, 1.08196105, 1.10595496, 0.71843094, 0.96830186,
             0.86464236, 0.6877866 , 1.02389637, 0.9050221 , 0.43615485]

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    zdata = np.array(zdata)

    xbins = np.linspace(0, 2,  3)
    ybins = np.linspace(0, 9,  3)
    zbins = np.linspace(0, 2, 15)

    (got_mean, got_sigma,
     got_chi2, got_valid) = fit_slices_2d_gauss(xdata, ydata, zdata,
                                                xbins, ybins, zbins,
                                                min_entries=0)
    expected_means   = [[1.02859384, 1.28457599],
                        [0.88808945, 0.98438076]]
    expected_meanus  = [[0.04256058, 0.05218108],
                        [0.02793445, 0.05043672]]
    expected_sigmas  = [[0.24456703, 0.33476073],
                        [0.22153181, 0.22877872]]
    expected_sigmaus = [[0.04256064, 0.05352718],
                        [0.02793447, 0.05043673]]
    expected_chi2s   = [[2.28464226, 1.35686866],
                        [1.21367525, 3.60471558]]

    expected_valid   = [[True] * 2] * 2

    assert_allclose(got_mean.value       , expected_means  , rtol=1e-4)
    assert_allclose(got_mean.uncertainty , expected_meanus , rtol=1e-4)
    assert_allclose(got_sigma.value      , expected_sigmas , rtol=1e-4)
    assert_allclose(got_sigma.uncertainty, expected_sigmaus, rtol=1e-4)
    assert_allclose(got_chi2             , expected_chi2s  , rtol=1e-4)
    assert np.all(got_valid == expected_valid)


@flaky(max_runs=5, min_passes=3)
def test_fit_slices_2d_gauss_statistics():
    nx, ny = 3, 4
    nz     = 10
    ndata  = 1000
    nxy    = nx * ny

    xbins = np.linspace( -10,   10, nx + 1)
    ybins = np.linspace( -50,  100, ny + 1)
    zbins = np.linspace( 800, 1300, nz + 1)

    means  = np.random.uniform(1000, 1100, nxy)
    sigmas = np.random.uniform(  50,   80, nxy)
    xdata  = np.concatenate([np.random.uniform(*xbins[i:i+2], size=ndata) for i in range(nx) for j in range(ny)])
    ydata  = np.concatenate([np.random.uniform(*ybins[j:j+2], size=ndata) for i in range(nx) for j in range(ny)])
    zdata  = np.concatenate([np.random.normal (     m,     s, size=ndata) for m, s in zip(means, sigmas)]       )

    (got_mean, got_sigma,
     got_chi2, got_valid) = fit_slices_2d_gauss(xdata, ydata, zdata,
                                                xbins, ybins, zbins,
                                                min_entries=0)

    assert np.all(np.abs(got_mean .value - means .reshape(nx, ny)) < 10 * got_mean .uncertainty)
    assert np.all(np.abs(got_sigma.value - sigmas.reshape(nx, ny)) < 10 * got_sigma.uncertainty)
    assert np.all(got_valid)


def test_fit_slices_2d_expo_fixed_example():
    xdata = [0.5] * 20 + [1.5] * 20 + [0.3] * 20 + [1.8] * 20
    ydata = [2.5] * 20 + [8.5] * 20 + [6.5] * 20 + [3.5] * 20
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    z1 = np.linspace( 0, 20, 20)
    z2 = np.linspace( 0, 21, 20)
    z3 = np.linspace(10, 18, 20)
    z4 = np.linspace( 5, 15, 20)
    zdata = np.concatenate([z1, z2, z3, z4])
    tdata = np.concatenate([np.exp(-z1/5), np.exp(-z2/8), np.exp(-z3/4), np.exp(-z4/3)])

    xbins  = np.linspace(0, 2, 3)
    ybins  = np.linspace(0, 9, 3)
    nzbins = 10
    zrange = 0, 25

    (got_const, got_slope,
     got_chi2 , got_valid) = fit_slices_2d_expo(xdata, ydata, zdata, tdata,
                                                xbins, ybins, nzbins, zrange,
                                                min_entries=0)

    expected_consts  = [[1.01032857 ,  0.75859013],
                        [1.13879766 ,  1.00254821]]
    expected_constus = [[0.08235294 ,  0.15158681],
                        [0.25808684 ,  0.05342974]]
    expected_slopes  = [[-5.02349732, -4.45998744],
                        [-2.92969552, -7.98687894]]
    expected_slopeus = [[0.17547667 ,  0.24570432],
                        [0.18435443 ,  0.29685494]]
    expected_chi2s   = [[0.12020284 ,  1.13175809],
                        [0.06120783 ,  0.13756717]]
    expected_valid   = [[True] * 2] * 2

    assert_allclose(got_const.value      , expected_consts , rtol=1e-5)
    assert_allclose(got_const.uncertainty, expected_constus, rtol=1e-5)
    assert_allclose(got_slope.value      , expected_slopes , rtol=1e-5)
    assert_allclose(got_slope.uncertainty, expected_slopeus, rtol=1e-5)
    assert_allclose(got_chi2             , expected_chi2s  , rtol=1e-5)
    assert np.all(got_valid == expected_valid)


def test_fit_slices_2d_expo_zrange_None():
    xdata = [0.5] * 20 + [1.5] * 20 + [0.3] * 20 + [1.8] * 20
    ydata = [2.5] * 20 + [8.5] * 20 + [6.5] * 20 + [3.5] * 20
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    zdata = np.concatenate([np.linspace( 0, 25, 20)] * 4)
    tdata = np.exp(-zdata/5.5)

    xbins  = np.linspace(0, 2, 3)
    ybins  = np.linspace(0, 9, 3)
    nzbins = 5

    (got_const, got_slope,
     got_chi2 , got_valid) = fit_slices_2d_expo(xdata, ydata, zdata, tdata,
                                                xbins, ybins, nzbins,
                                                zrange      = None,
                                                min_entries = 0)

    expected_consts  = [[ 1.12937291,  1.12937291],
                        [ 1.12937291,  1.12937291]]
    expected_constus = [[ 0.15748558,  0.15748558],
                        [ 0.15748558,  0.15748558]]
    expected_slopes  = [[-5.34687691, -5.34687691],
                        [-5.34687691, -5.34687691]]
    expected_slopeus = [[ 0.27008785,  0.27008785],
                        [ 0.27008785,  0.27008785]]
    expected_chi2s   = [[ 0.06615768,  0.06615768],
                        [ 0.06615768,  0.06615768]]
    expected_valid   = [[True] * 2] * 2

    assert_allclose(got_const.value      , expected_consts , rtol=1e-5)
    assert_allclose(got_const.uncertainty, expected_constus, rtol=1e-5)
    assert_allclose(got_slope.value      , expected_slopes , rtol=1e-5)
    assert_allclose(got_slope.uncertainty, expected_slopeus, rtol=1e-5)
    assert_allclose(got_chi2             , expected_chi2s  , rtol=1e-5)
    assert np.all(got_valid == expected_valid)


@given(sensible_floats, sensible_floats)
def test_sigmoid_limits(A, D):
    aux_sigmoid = lambda x: sigmoid(x, A, 0, 10, D)
    assert aux_sigmoid(-10) == approx(D  , rel=1e-2)
    assert aux_sigmoid( 10) == approx(A+D, rel=1e-2)


@given(sensible_floats,
       sensible_floats,
       sensible_floats.map(lambda x: x / 100),
       sensible_floats)
def test_sigmoid_values_at_abscissa_axis(A, B, C, D):
    value = sigmoid(0, A, B, C, D)
    test  = A / (1 + np.exp(B*C)) + D
    assert value == test


@flaky(max_runs=10, min_passes=9)
def test_compute_drift_v_when_moving_edge():
    edge    = np.random.uniform(530, 570)
    Nevents = 100 * 1000
    data    = np.random.uniform(450, edge, Nevents)
    data    = np.random.normal(data, 1)
    dv, dvu = compute_drift_v(data, 60, [500,600], "new", [1500, 550,1,0])
    dv_th   = DB.DetectorGeo('new').ZMAX[0]/edge

    assert dv_th == approx(dv, abs=5*dvu)

def test_sigmoid_failing_fit_return_nan():
    dst     = np.random.rand(1000)
    dv_vect = compute_drift_v(dst, nbins=35, zrange=(500, 640), seed=None, detector="new")
    nans    = np.array([np.nan, np.nan])
    assert dv_vect, nans
