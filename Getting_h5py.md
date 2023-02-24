# Instruction for installing h5py and mpi4py on Dardel

Kindly provided by Xin Li at PDC. Hopefully, these are transferrable to other machines with some modifications.

### load modules
````
module load PDC/21.11
module load cpeGNU/21.11
module load cray-python/3.9.4.2
````
### set compiler wrappers
````
export CC=cc
export CXX=CC
export FC=ftn
````
### create virtual environment
````
python3 -m venv my-venv
source my-venv/bin/activate
````
### install mpi4py
````
MPICC=cc python3 -m pip install --no-binary=mpi4py -v mpi4py
````
### Download hdf5 from https://www.hdfgroup.org

Note: hdf5 should be built with the high level library option
(--enable-hl) which is enabled by default.
See https://github.com/h5py/h5py/issues/1640
````
tar xf hdf5-1.12.2.tar.gz

cd hdf5-1.12.2/

./configure -h

./configure --prefix=/path/to/install/hdf5-1.12.2/build --enable-parallel --enable-shared

srun -n 1 make -j 16

make install

export HDF5_DIR=/path/to/install/hdf5-1.12.2/build

export LD_LIBRARY_PATH=/path/to/install/hdf5-1.12.2/build/lib:$LD_LIBRARY_PATH
````
### install h5py
````
HDF5_MPI="ON" CC=cc srun -n 1 python3 -m pip install --no-binary=h5py -v h5py
srun -n 1 python3 -c 'import h5py'
srun -n 1 python3 -c 'import h5py; print(h5py.get_config().mpi)'
````
To check whether h5py support parallel I/O:
`c = h5py.get_config()`
`is_parallel = (hasattr(c, "mpi") and c.mpi)`
