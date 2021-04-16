MPI-Based Block Vectors and Matrices
====================================

PyNumero's MPI-based block vectors and matrices
(:py:class:`~pyomo.contrib.pynumero.sparse.mpi_block_vector.MPIBlockVector`
and
:py:class:`~pyomo.contrib.pynumero.sparse.mpi_block_matrix.MPIBlockMatrix`)
behave very similarly to `BlockVector` and `BlockMatrix`. The primary
difference is in construction. With `MPIBlockVector` and
`MPIBlockMatrix`, each block is owned by either a single process/rank
or all processes/ranks.

Consider the following example (in a file called "parallel_vector_ops.py").

.. code-block:: python

  import numpy as np
  from mpi4py import MPI
  from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  
  owners = [2, 0, 1, -1]
  x = MPIBlockVector(4, rank_owner=owners, mpi_comm=comm)
  x.set_block(owners.index(rank), np.ones(3)*(rank + 1))
  x.set_block(3, np.array([1, 2, 3]))
  
  y = MPIBlockVector(4, rank_owner=owners, mpi_comm=comm)
  y.set_block(owners.index(rank), np.ones(3)*(rank + 1))
  y.set_block(3, np.array([1, 2, 3]))
  
  z1: MPIBlockVector = x + y  # add x and y
  z2 = x.dot(y)  # dot product
  z3 = np.abs(x).max()  # infinity norm
  
  z1_local = z1.make_local_copy()
  if rank == 0:
      print(z1_local.flatten())
      print(z2)
      print(z3)

This example can be run with

.. code-block::

   mpirun -np 3 python -m mpi4py parallel_vector_ops.py

The output is

.. code-block::

  [6. 6. 6. 2. 2. 2. 4. 4. 4. 2. 4. 6.]
  56.0
  3

Note that the `make_local_copy()` method is not efficient and should
only be used for debugging.

The -1 in `owners` means that the block at that index (index 3 in this
example) is owned by all processes. The non-negative integer values
indicate that the block at that index is owned by the process with
rank equal to the value. In this example, rank 0 owns block 1, rank 1
owns block 2, and rank 2 owns block 0. Block 3 is owned by all ranks.
Note that blocks should only be set if the process/rank owns that
block.

The operations performed with `MPIBlockVector` are identical to the
same operations peformed with `BlockVector` (or even NumPy arrays),
except that the operations are now performed in parallel.

`MPIBlockMatrix` construction is very similar. Consider the following
example in a file called "parallel_matvec.py".

.. code-block::

  import numpy as np
  from mpi4py import MPI
  from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
  from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
  from scipy.sparse import random
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  
  owners = [0, 1, 2, -1]
  x = MPIBlockVector(4, rank_owner=owners, mpi_comm=comm)
  
  owners = np.array([[ 0, -1, -1, 0],
                     [-1,  1, -1, 1],
                     [-1, -1,  2, 2]])
  a = MPIBlockMatrix(3, 4, rank_ownership=owners, mpi_comm=comm)
  
  np.random.seed(0)
  x.set_block(3, np.random.uniform(-10, 10, size=10))
  
  np.random.seed(rank)
  x.set_block(rank, np.random.uniform(-10, 10, size=10))
  a.set_block(rank, rank, random(10, 10, density=0.1))
  a.set_block(rank, 3, random(10, 10, density=0.1))
  
  b = a * x  # parallel matrix-vector dot product
  
  # check the answer
  local_x = x.make_local_copy().flatten()
  local_a = a.to_local_array()
  local_b = b.make_local_copy().flatten()
  
  if rank == 0:
      print('error: ', np.abs(local_a.dot(local_x) - local_b).max())

Which can be run with

.. code-block::

   mpirun -np 3 python -m mpi4py parallel_matvec.py

The output is

.. code-block::

   error:  4.440892098500626e-16

The most difficult part of using `MPIBlockVector` and `MPIBlockMatrix`
is determining the best structure and rank ownership to maximize
parallel efficiency.
