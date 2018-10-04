.. _parallelsection:

Parallel Implementation
===================================

Parallel implementation in parmest is **preliminary**.
If you would like to run parmest in parallel, you need mpi and the mpi4py Python package.
If you do NOT have mpi installed, everything is still supposed to work (you should not get MPI import errors).
Examples are included for Rooney Biegler and Semibatch.

Note that it seems to be easiest to install the mpi4py package with conda. 
The package requires a *compatible* mpi installation.

Verify Installation
----------------------

In a Unix terminal window, use these commands to verify parallel installation::

    cd pyomo/contrib/parmest/examples/semibatch
    time mpiexec -np 2 python sb_drive_parmest.py
    time python sb_drive_parmest.py

The first one should be faster. Both take many minutes.
	
Notes for Mac Users
-------------------

In preliminary testing, this only worked with Python 3.5 using the following for installation::

    conda create -n parmest-parallel python=3.5
    source activate parmest-parallel
    conda install -c conda-forge mpi4py
	
Create a conda environment with Python 3.5 and install the following::

    conda install -c conda-forge mpi4py

This should install libgfortran, mpi, mpi4py, and openmpi.

.. The following is not shown in the UM
   I have NOT had luck with mpich instead of openmpi

To compare parmest with and without parallel computing, we ran the following:

    i.      In parmest/examples/semibatch run “time python sb_drive_parmest_serial.py” – it should generate sb_boot_serial.png
    ii.      In parmest/examples/semibatch run “time mpirun -np 2 python sb_drive_parmest_parallel.py” – it should generate sb_boot_parallel.png
    iii.      The two figures should be the same.
    iv.      The execution time should be faster using mpirun.


