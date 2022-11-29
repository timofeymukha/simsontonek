import numpy.polynomial.chebyshev

from simsontonek.converter import (
    Converter,
    glc_points,
    chebyshev_interpolation,
    fft_interpolate,
)
from numpy.testing import assert_array_almost_equal
import numpy as np


def test_glc():
    glc = glc_points(1)
    assert_array_almost_equal(glc, [1, -1])
    glc = glc_points(2)
    assert_array_almost_equal(glc, [1, 0, -1])
    pass


def test_chebyshev_interpolation():
    glc = glc_points(2)
    data = np.sin(glc)
    chebyshev_interpolation(data, glc, 0)

    # vals = [
    #    chebyshev_interpolation(data, glc, data[i])
    #    for i in range(data.size)
    # ]
    #    assert_array_almost_equal(data, vals)
    pass


def test_fft_interpolation():
    x = np.linspace(0, 2 * np.pi, 6)
    data = np.sin(x)
    xn = np.full(1, 0.1)

    fft_interpolate(data, x, xn)
    pass


def test_converter():

    inlet_y = np.linspace(0, 20, 100)
    inlet_z = np.linspace(0, 100, 250)
    simson = (
        "C:\\Users\\tiamm\\Software\\inflow_interpolation\\simson_data.hdf5"
    )
    Converter(
        simson,
        inlet_y,
        inlet_z,
        length_scale=1,
        velocity_scale=1,
        shift_y=0,
        shift_z=0,
    )
