import numpy as np
from numpy.fft import fft, fftshift
from numpy.polynomial.chebyshev import Chebyshev
from scipy.interpolate import BarycentricInterpolator
import h5py
from tqdm import trange
from mpi4py import MPI
from os.path import join
from simsontonek.chunks import chunks_and_offsets

__all__ = ["Converter", "glc_points"]


def glc_points(n: int) -> np.ndarray:
    return np.cos(np.pi * np.arange(n + 1) / n)


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

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

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

        [chunks, offsets] = chunks_and_offsets(nprocs, nplanes)

        if rank == 0:
            loop_range = trange(chunks[rank])
        else:
            loop_range = range(chunks[rank])

        for t in loop_range:
            position = offsets[rank] + t
            for component in ["u", "v", "w"]:

                # Read the velocity values from Simson
                u = np.flipud(data[component][:, :, position] / velocity_scale)

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
                u_nek.T.tofile(join(write_path, f"b{component}in_{position}"))

                continue

        comm.Barrier()
