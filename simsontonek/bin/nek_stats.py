import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
from os.path import join


def main():
    parser = argparse.ArgumentParser(
        description="A utility for inspecting the statisitics of  \
                     Nek5000 inflow planes."
    )

    parser.add_argument(
        "--data",
        type=str,
        help="The folder with the data",
        required=True,
    )

    parser.add_argument(
        "--inlety",
        type=str,
        help="Text file with the values of the y coordinates in "
        "the Nek5000 plane",
        required=True,
    )

    parser.add_argument(
        "--inletz",
        type=str,
        help="Text file with the values of the y coordinates in "
        "the Nek5000 plane",
        required=True,
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Whether to save figures with the mean flow and re-stresses.",
    )
    parser.add_argument("--no-plots", action="store_false", dest="plots")
    parser.set_defaults(plots=True)

    args = parser.parse_args()

    data_path = args.data
    plots = args.plots

    y = np.genfromtxt(args.inlety)
    z = np.genfromtxt(args.inletz)

    ny = y.size
    nz = z.size

    search_string = join(data_path, "b[uvw]in_*")
    datafiles = glob.glob(search_string)

    u = np.stack(
        [
            np.reshape(np.fromfile(i), (ny, nz), "F")
            for i in datafiles
            if "buin" in i
        ],
        axis=2,
    )

    v = np.stack(
        [
            np.reshape(np.fromfile(i), (ny, nz), "F")
            for i in datafiles
            if "bvin" in i
        ],
        axis=2,
    )

    w = np.stack(
        [
            np.reshape(np.fromfile(i), (ny, nz), "F")
            for i in datafiles
            if "bwin" in i
        ],
        axis=2,
    )

    print(u.shape)
    umean = np.mean(u, axis=(1, 2))
    vmean = np.mean(v, axis=(1, 2))
    wmean = np.mean(v, axis=(1, 2))

    uprime = u - umean[:, np.newaxis, np.newaxis]
    vprime = v - vmean[:, np.newaxis, np.newaxis]
    wprime = w - wmean[:, np.newaxis, np.newaxis]

    uu = np.mean(uprime**2, axis=(1, 2))
    vv = np.mean(vprime**2, axis=(1, 2))
    ww = np.mean(wprime**2, axis=(1, 2))
    uv = np.mean(uprime * vprime, axis=(1, 2))

    idx = np.argmin(np.abs(umean - 0.99))
    delta99 = y[idx]
    delta_star = np.trapz((1 - umean / umean[-1]), x=y)
    theta = np.trapz(umean / umean[-1] * (1 - umean / umean[-1]), x=y)

    print(f"delta99 = {delta99}")
    print(f"delta* = {delta_star}")
    print(f"theta = {theta}")

    if plots:
        plt.figure()
        plt.plot(y, umean, label="<u>")
        plt.plot(y, 30 * vmean, label="30<v>")
        plt.xlim(0, 1.5 * delta99)
        plt.xlabel(r"$y$")
        plt.grid()
        plt.legend()
        plt.savefig("nek_u.pdf")

        plt.figure()
        plt.plot(y, uu, label="<u'u'>")
        plt.plot(y, vv, label="<v'v'>")
        plt.plot(y, ww, label="<w'w'>")
        plt.plot(y, uv, label="<u'v'>")
        plt.xlabel(r"$y$")
        plt.xlim(0, 1.5 * delta99)
        plt.grid()
        plt.legend()
        plt.savefig("nek_restresses.pdf")
