import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="A utility for inspecting the statisitics of a  \
                     Simson inflow database stored in hdf5 format. \
                     timeseries module."
    )

    parser.add_argument(
        "--data",
        type=str,
        help="The hdf5 file with the data",
        required=True,
    )

    parser.add_argument(
        "--store",
        action="store_true",
        help="Whether to store the computed data in the hdf5 file.",
    )
    parser.add_argument("--no-store", action="store_false", dest="store")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Whether to save figures with the mean flow and re-stresses.",
    )
    parser.add_argument("--no-plots", action="store_false", dest="plots")
    parser.set_defaults(store=True, plots=True)

    args = parser.parse_args()

    file_path = args.data
    store = args.store
    plots = args.plots

    access = "r+" if store else "r"
    data = h5py.File(file_path, access)

    nplanes = int(data.attrs["nplanes"])
    ny = int(data.attrs["ny"])
    nz = int(data.attrs["nz"])
    zl = data.attrs["zl"]
    dstar = data.attrs["dstar"]

    print(f"The number of planes is {nplanes}")
    print(f"d* =  {dstar}")
    print(f"Ny =  {ny}")
    print(f"Nz =  {nz}")

    # rescale with dstar
    yl = 2 / dstar
    zl = zl / dstar

    # The simson grid
    y = np.zeros(ny)
    for i in range(ny):
        y[i] = (np.cos(np.pi * i / (ny - 1)) + 1) / 2 * yl

    y = y[::-1]
    z = zl / nz * np.arange(nz)

    print(f"y-span : {np.min(y)}, {np.max(y)}")
    print(f"z-span : {np.min(z)}, {np.max(z)}")

    if "theta" in data.attrs.keys():
        print("Statistics are already stored...")
        theta = data.attrs["theta"]
        delta99 = data.attrs["delta99"]
        delta_star = data.attrs["delta_star"]

        u = data["umean"][:]
        v = data["vmean"][:]
        w = data["wmean"][:]
        uu = data["uu"][:]
        vv = data["vv"][:]
        ww = data["ww"][:]
        uv = data["uv"][:]

    else:
        print("Computing statistics...")

        u = np.mean(data["u"], axis=(1, 2))
        v = np.mean(data["v"], axis=(1, 2))
        w = np.mean(data["v"], axis=(1, 2))

        uprime = data["u"] - u[:, np.newaxis, np.newaxis]
        vprime = data["v"] - v[:, np.newaxis, np.newaxis]
        wprime = data["w"] - w[:, np.newaxis, np.newaxis]

        uu = np.mean(uprime**2, axis=(1, 2))
        vv = np.mean(vprime**2, axis=(1, 2))
        ww = np.mean(wprime**2, axis=(1, 2))
        uv = np.mean(uprime * vprime, axis=(1, 2))

        u = u[::-1]
        v = v[::-1]
        w = w[::-1]
        uu = uu[::-1]
        vv = vv[::-1]
        ww = ww[::-1]
        uv = uv[::-1]

        idx = np.argmin(np.abs(u - 0.99))
        delta99 = y[idx]
        delta_star = np.trapz((1 - u / u[-1]), x=y)
        theta = np.trapz(u / u[-1] * (1 - u / u[-1]), x=y)

        if store:
            data.create_dataset("umean", data=u)
            data.create_dataset("vmean", data=v)
            data.create_dataset("wmean", data=w)
            data.create_dataset("uu", data=uu)
            data.create_dataset("vv", data=vv)
            data.create_dataset("ww", data=ww)
            data.create_dataset("uv", data=uv)
            data.attrs.create("theta", theta)
            data.attrs.create("delta99", delta99)
            data.attrs.create("delta_star", delta_star)

    print(f"delta99 = {delta99}")
    print(f"delta* = {delta_star}")
    print(f"theta = {theta}")
    if plots:
        plt.figure()
        plt.plot(y, u, label="<u>")
        plt.plot(y, 30 * v, label="30<v>")
        plt.xlim(0, 1.5 * delta99)
        plt.xlabel(r"$y$")
        plt.grid()
        plt.legend()
        plt.savefig("simson_u.pdf")

        plt.figure()
        plt.plot(y, uu, label="<u'u'>")
        plt.plot(y, vv, label="<v'v'>")
        plt.plot(y, ww, label="<w'w'>")
        plt.plot(y, uv, label="<u'v'>")
        plt.xlabel(r"$y$")
        plt.xlim(0, 1.5 * delta99)
        plt.grid()
        plt.legend()
        plt.savefig("simson_restresses.pdf")

        #    re_delta_star = delta_star * u[-1] / nu;
        #    re_delta99 = delta99 * u0 / nu;

        print(u.shape)

        data.close()


if __name__ == "__main__":
    main()
