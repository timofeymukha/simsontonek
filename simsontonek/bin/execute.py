from os.path import join
import argparse
import numpy as np
from simsontonek.converter import Converter
import h5py
import struct


def main():

    parser = argparse.ArgumentParser(
        description="Execute the interpolation to hdf5 format."
    )

    parser.add_argument(
        "--data",
        type=str,
        help="The hdf5 file with the data",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="The path for the generated planes.",
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
        "--lengthscale",
        type=float,
        help="The length scale which will correspond to length 1 "
        "in the Nek5000",
        required=False,
    )

    parser.add_argument(
        "--velocityscale",
        type=float,
        help="The velocity scale which will correspond to velocity 1 "
        "in the Nek5000",
        required=False,
    )

    parser.set_defaults(lengthscale=1.0, velocityscale=1.0)

    args = parser.parse_args()
    file_path = args.data
    output_path = args.output
    length_scale = args.lengthscale
    velocity_scale = args.velocityscale
    inlet_y = np.genfromtxt(args.inlety)
    inlet_z = np.genfromtxt(args.inletz)

    Converter(
        file_path,
        output_path,
        inlet_y,
        inlet_z,
        length_scale=length_scale,
        velocity_scale=velocity_scale,
        shift_y=0,
        shift_z=0,
    )


if __name__ == "__main__":
    main()
