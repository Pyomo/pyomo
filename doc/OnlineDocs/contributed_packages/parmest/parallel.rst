.. _parallelsection:

Parallel Implementation
===================================

Parallel implementation in parmest is **preliminary**.
To run parmest in parallel, you need the mpi4py Python package and a *compatible* MPI installation.
If you do NOT have mpi4py or a MPI installation, parmest still works (you should not get MPI import errors).

For example, the following command can be used to run the semibatch model in parallel::

	mpiexec -n 4 python semibatch_parmest_parallel.py

The file **semibatch_parmest_parallel.py** is shown below. 
Results are saved to file for later analysis.

.. literalinclude:: ../../../../pyomo/contrib/parmest/examples/semibatch/semibatch_parmest_parallel.py
   :language: python
   
Installation
-------------

The mpi4py Python package should be installed using conda. 
The following installation instructions were tested on a Mac with Python 3.5.

Create a conda environment and install mpi4py using the following commands::

    conda create -n parmest-parallel python=3.5
    source activate parmest-parallel
    conda install -c conda-forge mpi4py
	
This should install libgfortran, mpi, mpi4py, and openmpi.

To verify proper installation, create a Python file with the following::

	from mpi4py import MPI
	import time 
	comm = MPI.COMM_WORLD 
	rank = comm.Get_rank() 
	print('Rank = ',rank)
	time.sleep(10) 

Save the file as test_mpi.py and run the following command::

	time mpiexec -n 4 python test_mpi.py 
	time python test_mpi.py 
	
The first one should be faster and should start 4 instances of Python.
