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

    simson = (
        "C:\\Users\\tiamm\\Nek5000\\run\\generate_planes\\wide_data.hdf5"
    )
    inlet_y = np.genfromtxt(
        "C:\\Users\\tiamm\\Nek5000\\run\\tbl\\wmles_tsfp\\simson_n2\\inlet_y.txt",
    )
    inlet_z = np.genfromtxt(
        "C:\\Users\\tiamm\\Nek5000\\run\\tbl\\wmles_tsfp\\simson_n2\\inlet_z.txt",
    )
    Converter(
        simson,
        "C:\\Users\\tiamm\\Nek5000\\run\\tbl\\wmles_tsfp\\test_inflow\\planes",
        inlet_y,
        inlet_z,
        length_scale=14.2172,
        velocity_scale=1,
        shift_y=0,
        shift_z=0,
    )
