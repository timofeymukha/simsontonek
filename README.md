# simsontonek

The is a Python package to convert TBL data planes sampled from Simson onto a mesh used for a Nek5000 simulation.
Although, in principle, any other structured plane-mesh as well.
The interpolated data is saved in a format that is read by a reader internally at KTH.
So, mostly this is an internal tool for our local needs.
If you are someone external looking at this, chances are this is a waste of your time.

## Installing
You need a Python environment with a working `mpi4py` and preferably also `h5py` built on top of a parallel-enabled HDF5.
Instructions for obtaining this on a supercomputer are given in Getting_h5py.md in this repo.
These particular instructions are for Dardel at PDC in Stockholm, but can probably be easily adapted to other machines.

Once you have that, install with pip
`pip install --editable .`
The `--editable` flag will allow you to tweak the code without having to reinstall.
The flag is thus optional.

## Using
The package provides several scripts that should be in your `PATH` after the installation.
All scripts are given some documentation for the required command line parameters.

* `simsontonek_convert_to_hdf5` will convert the `planes_{uvw}`files from Simson to an HDF5 file.
* `simsontonek_simson_stats` will compute some statistics for the data in the HDF5 file. By default, two pdf file with 1s and 2nd order statistics will be generate as well.
* `simsontonek_nek_stats` is the same as above, but for the generated plane files in Nek5000 format. Useful to check that things have worked as expected.
* `simsontonek_execute` will actually perform the conversion. The script is mpi-enabled, which saves a lot of time!

