import numpy as np
from numpy.fft import fft, fftshift
from numpy.polynomial.chebyshev import Chebyshev


def chebyshev_interpolation(data: np.ndarray, nodes: np.ndarray, xn: float):
    """This does not seem to work very well, and probably barycentric
    interpolation should be used instead.

    Currently not used in the code.
    """
    # polynomial order
    n = data.size - 1

    xs = nodes[0]
    xe = nodes[-1]

    poly = Chebyshev.basis(n).deriv()

    # map to [-1, 1]
    chi = (2 * nodes - (xs + xe)) / (xe - xs)
    xnc = (2 * xn - (xs + xe)) / (xe - xs)

    psi = np.zeros(n + 1)

    dchebyshev = poly(xnc)

    for i in range(n + 1):
        cj = 2 if i in [0, n] else 1
        psi[i] = ((-1) ** (i + 1) * (1 - xnc**2) * dchebyshev) / (
            cj * n**2 * (xnc - chi[i])
        )

    return np.sum(data * psi)


def fft_interpolate(u: np.ndarray, x: np.ndarray, xn: np.ndarray):
    """Interpolates u onto xn using FFT.

    Note: the size of u and x *must* be an even number!

    Currently not used in the code.
    """
    NN = x.size
    N = int(NN - 1)
    n = xn.size
    L = x[-1] - x[0]

    # FFT
    uhat = fft(u[:N])
    # Shift the frequencies so that the 0-freq is in the centre
    uhat = fftshift(uhat)

    # The central coefficient, which corresponds to the mean
    c0 = uhat[int((N - 1) / 2)]
    c_pl = uhat[int(((N - 1) / 2) + 1) :]
    c_mi = uhat[: int((N - 1) / 2)]

    an = c_pl + np.flipud(c_mi)
    bn = 1j * (c_pl - np.flipud(c_mi))
    un = np.zeros(n)

    # sum the sines and cosines on the new grid
    for k in range(xn.size):
        un[k] = c0
        for i in range(an.size):
            un[k] += an[i] * np.cos(2 * np.pi * i * xn[k] / L) + bn[
                i
            ] * np.sin(2 * np.pi * i * xn[k] / L)

    return (1 / N) * un
