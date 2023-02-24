from os.path import join
import argparse
import numpy as np
import h5py
import struct


def read_int(infile, emode, nvar):
    """read integer array"""
    isize = 4
    llist = infile.read(isize * nvar)
    llist = list(struct.unpack(emode + nvar * "i", llist))
    return llist


def read_flt(infile, emode, wdsize, nvar):
    """read real array"""
    if wdsize == 4:
        realtype = "f"
    elif wdsize == wdsize:
        realtype = "d"
    llist = infile.read(wdsize * nvar)
    llist = np.frombuffer(llist, dtype=emode + realtype, count=nvar)
    return llist


def read_component(filename, nplanes, wdsize, endian):
    u_file = open(filename, "rb")

    # Read in some stuff to get to the point with velocity data
    _ = read_int(u_file, endian, 1)
    _ = read_flt(u_file, endian, wdsize, 1)
    _ = read_int(u_file, endian, 1)
    _ = read_flt(u_file, endian, wdsize, 1)
    _ = read_flt(u_file, endian, wdsize, 1)
    _ = read_flt(u_file, endian, wdsize, 1)
    _ = read_flt(u_file, endian, wdsize, 1)
    _ = read_int(u_file, endian, 2)

    _ = read_int(u_file, endian, 1)[0]
    ny = read_int(u_file, endian, 1)[0]
    nz = read_int(u_file, endian, 1)[0]
    _ = read_int(u_file, endian, 1)
    _ = read_int(u_file, endian, 2)

    _ = read_int(u_file, endian, 1)
    _ = read_int(u_file, endian, 1)
    _ = read_flt(u_file, endian, wdsize, 1)
    _ = read_int(u_file, endian, 1)
    _ = read_flt(u_file, endian, wdsize, 1)
    dstar = read_int(u_file, endian, 1)

    t = np.zeros(nplanes)
    u = np.zeros((ny, nz, nplanes))

    for i in range(nplanes):
        _ = read_int(u_file, endian, 1)
        t[i] = read_flt(u_file, endian, wdsize, 1) / dstar
        _ = read_flt(u_file, endian, wdsize, 1)
        _ = read_int(u_file, endian, 2)
        u[:, :, i] = read_flt(u_file, endian, wdsize, ny * nz).reshape(
            ny, nz, order="F"
        )
        _ = read_int(u_file, endian, 1)

    u_file.close()
    return t, u


def main():

    parser = argparse.ArgumentParser(
        description="A utility to convert planes sampled from Simson  \
                     to hdf5 format."
    )

    parser.add_argument(
        "--data",
        type=str,
        help="The path to the folder with the data",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="The name of the output hdf5.",
        required=True,
    )

    parser.add_argument(
        "--nplanes",
        type=int,
        help="The number of planes (i.e. timesteps) in the data",
        required=True,
    )

    parser.add_argument(
        "--endian",
        type=str,
        choices=(">", "<"),
        help="Either '>' o '<'",
        required=True,
    )

    parser.add_argument(
        "--wdsize",
        type=int,
        choices=(4, 8),
        help="Either 4 or 8 for single and double precision.",
        required=True,
    )

    args = parser.parse_args()

    # Set this to the folder containing the plane_ files
    data_path = args.data
    nplanes = args.nplanes
    endian = args.endian
    wdsize = args.wdsize
    output = args.output

    # Read the velocity data and time
    t, u = read_component(
        join(data_path, "plane_u.dat"), nplanes, wdsize, endian
    )
    _, v = read_component(
        join(data_path, "plane_v.dat"), nplanes, wdsize, endian
    )
    _, w = read_component(
        join(data_path, "plane_w.dat"), nplanes, wdsize, endian
    )

    # Read in some attributes
    with open(join(data_path, "plane_u.dat"), "rb") as u_file:

        _ = read_int(u_file, endian, 1)
        re = read_flt(u_file, endian, wdsize, 1)
        _ = read_int(u_file, endian, 1)
        xl = read_flt(u_file, endian, wdsize, 1)
        zl = read_flt(u_file, endian, wdsize, 1)
        _ = read_flt(u_file, endian, wdsize, 1)
        _ = read_flt(u_file, endian, wdsize, 1)
        _ = read_int(u_file, endian, 2)

        nx = read_int(u_file, endian, 1)[0]
        ny = read_int(u_file, endian, 1)[0]
        nz = read_int(u_file, endian, 1)[0]
        _ = read_int(u_file, endian, 1)
        _ = read_int(u_file, endian, 2)

        _ = read_int(u_file, endian, 1)
        _ = read_int(u_file, endian, 1)
        _ = read_flt(u_file, endian, wdsize, 1)
        _ = read_int(u_file, endian, 1)
        dstar = read_flt(u_file, endian, wdsize, 1)

    output_file = h5py.File(output, "w")

    output_file.create_dataset("u", data=u)
    output_file.create_dataset("v", data=v)
    output_file.create_dataset("w", data=w)
    output_file.create_dataset("t", data=t)

    output_file.attrs.create("nplanes", float(u.shape[-1]))
    output_file.attrs.create("nx", float(nx))
    output_file.attrs.create("ny", float(ny))
    output_file.attrs.create("nz", float(nz))
    output_file.attrs.create("dstar", dstar)

    output_file.attrs.create("xl", xl)
    output_file.attrs.create("zl", zl)
    output_file.attrs.create("re", re)

    output_file.close()


if __name__ == "__main__":
    main()
