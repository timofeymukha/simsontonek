import numpy as np
from numpy.fft import fft, fftshift
from numpy.polynomial.chebyshev import Chebyshev
from scipy.interpolate import BarycentricInterpolator
import h5py
from os.path import join

__all__ = ["Converter", "glc_points"]


def glc_points(n: int) -> np.ndarray:
    return np.cos(np.pi * np.arange(n + 1) / n)


def chebyshev_interpolation(data: np.ndarray, nodes: np.ndarray, xn: float):
    """This does not seem to work very well, and probably barycentric
    interpolation should be used instead.
    """
    # polynomial order
    n = data.size - 1

    xs = nodes[0]
    xe = nodes[-1]

    poly = Chebyshev.basis(n).deriv()

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


class Converter:
    def __init__(
        self,
        simson_data_path: str,
        write_path: str,
        inlet_y: np.ndarray,
        inlet_z: np.ndarray,
        length_scale: float,
        velocity_scale: float,
        shift_y: float,
        shift_z: float,
    ) -> None:

        data = h5py.File(simson_data_path, "r")
        nplanes = int(data.attrs["nplanes"])

        ny = int(data.attrs["ny"])
        nz = int(data.attrs["nz"])
        zl = data.attrs["zl"]
        dstar = data.attrs["dstar"]

        # Free stream values to be added on to of the tbl
        # Note, index 0 here since values in Simson are inverted
        vinf = np.mean(data["v"][0, :, :]) / velocity_scale
        uinf = np.mean(data["u"][0, :, :]) / velocity_scale
        freestream = {"u": uinf, "v": vinf, "w": 0}

        # rescale with dstar
        yl = 2 / dstar
        zl = zl / dstar

        # The simson grid
        y = np.zeros(ny)
        for i in range(ny):
            y[i] = (np.cos(np.pi * i / (ny - 1)) + 1) / 2 * yl

        y = y[::-1]
        z = zl / nz * np.arange(nz)

        ys, zs = np.meshgrid(y, z, indexing="ij")

        # scale the grid the desired length scale
        y = y / length_scale + shift_y
        z = z / length_scale + shift_z

        # generate nek mesh
        yn, zn = np.meshgrid(inlet_y, inlet_z, indexing="ij")

        # find index in the inflow mesh, where the Simson TBL data ends
        y_ind = np.argwhere(yn[:, 1] >= np.max(y))
        y_ind = y_ind[0][0] if y_ind.size > 0 else yn.shape[0]

        interp = BarycentricInterpolator(y)

        nplanes = 2
        for t in range(nplanes):
            for component in ["u", "v", "w"]:

                # Read the velocity values from Simson
                u = np.flipud(data[component][:, :, t] / velocity_scale)

                # Values interpolated in y, for each z value of the
                # Simson grid.
                # Note: we can pass the whole 2d array to the
                # interpolator, just need to tell which axis
                # corresponds to the interpolation direction
                interp.set_yi(u, axis=0)
                utemp = interp(yn[:y_ind, 0])

                # FFT in z
                uhat = np.fft.fft(utemp, axis=1)
                uhat = np.hstack(
                    [uhat, np.conjugate(uhat[:, int(nz / 2)][:, None])]
                )
                freq = np.fft.fftfreq(nz, d=z[1] - z[0])
                freq = np.concatenate([freq, [-freq[int(nz / 2)]]])

                # Interpolate using inverse fft
                z3d = np.expand_dims(inlet_z, axis=(0, 1))
                u_nek = (
                    1
                    / nz
                    * np.sum(
                        uhat[:, :, None]
                        * np.exp(2j * np.pi * freq[None, :, None] * z3d),
                        axis=1,
                    )
                )

                u_nek = np.vstack(
                    (
                        u_nek,
                        np.full(
                            (yn.shape[0] - y_ind, yn.shape[1]),
                            freestream[component],
                        ),
                    ),
                )

                u_nek = np.real(u_nek)

                # Write
                # Transposing because writing is in row-major order
                # and fortran assumes column-major
                u_nek.T.tofile(join(write_path, f"b{component}in_{t}"))

                continue

        pass
