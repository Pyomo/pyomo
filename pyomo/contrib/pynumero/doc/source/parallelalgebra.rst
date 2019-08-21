Parallel Block Algebra
======================

Structure in optimization problems can be exploited through parallel
algorithms. PyNumero supports parallel linear algebra by extending the
block algebra classes presented in the previous section. The
:py:class:`BlockVector
<pyomo.contrib.pynumero.sparse.block_vector.BlockVector>` and
:py:class:`BlockMatrix
<pyomo.contrib.pynumero.sparse.block_matrix.BlockMatrix>` classes
described earlier are computationally efficient in that most of their
operations are performed in C/C++. Therefore, these classes can be used
to build efficient serial optimization algorithms without significant
coding effort. However, many operations in these block algebra classes
can be further accelerated using parallelization. Indeed, that's the
main goal of retaining structure in block algebra operations. The idea
is then to allocate blocks to different processors and operate on them
in parallel. A simple example of this would be summing all elements in a
:py:class:`BlockVector
<pyomo.contrib.pynumero.sparse.block_vector.BlockVector>`. If each block
is assigned to a different processor, all processors sum their own block
and then communicate the result to the rest to add them up. A similar
procedure can be done when performing a dot product, a matrix-vector, or
a matrix-matrix product.

We extended the :py:class:`BlockVector
<pyomo.contrib.pynumero.sparse.block_vector.BlockVector>` and
:py:class:`BlockMatrix
<pyomo.contrib.pynumero.sparse.block_matrix.BlockMatrix>` to perform
operations in parallel. To do this we relied on the package `MPI4py
<https://mpi4py.readthedocs.io/en/stable/>`_ that allowed us to
implement :py:class:`MPIBlockVector
<pyomo.contrib.pynumero.sparse.mpi_block_vector.MPIBlockVector>` and
:py:class:`MPIBlockMatrix
<pyomo.contrib.pynumero.sparse.mpi_block_matrix.MPIBlockMatrix>` . These
two classes are very similar to their serial counter parts. They inherit
from ``numpy.ndarray`` and ``scipy.sparse.spmatrix`` and they look
almost as the :py:class:`BlockVector
<pyomo.contrib.pynumero.sparse.block_vector.BlockVector>` and the
:py:class:`BlockMatrix
<pyomo.contrib.pynumero.sparse.block_matrix.BlockMatrix>`. However, when
building an :py:class:`MPIBlockVector
<pyomo.contrib.pynumero.sparse.mpi_block_vector.MPIBlockVector>` or an
:py:class:`MPIBlockMatrix
<pyomo.contrib.pynumero.sparse.mpi_block_matrix.MPIBlockMatrix>` , users
must specify the processors where the blocks live. We demonstrate some
of the features of the :py:class:`MPIBlockVector
<pyomo.contrib.pynumero.sparse.mpi_block_vector.MPIBlockVector>` in the
code below

.. code-block:: python

  # This code is available in pyomo.contrib.pynumero.examples
  # to run the code use the following command
  # mpirun -n 2 python -m mpi4py mpi_block_vector_example.py
  from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
  from mpi4py import MPI
  import numpy as np
  comm = MPI.COMM_WORLD

  # create a parallel block vector
  nblocks = 4
  rank_ownership = [0,1,0,1]
  v = MPIBlockVector(nblocks, rank_ownership, comm)

  # blocks 0 and 2 live in processor 0
  # blocks 1 and 3 live in processor 1
  rank = comm.Get_rank()
  if rank == 0:
      v[0] = np.array([1.,2, 3, 4.])
      v[2] = np.array([4., 5, 6])

  if rank == 1:
      v[1] = np.zeros(2)
      v[3] = np.ones(2)

  # inform other processors of the sizes of the blocks
  # this method has to be called after assignment of blocks is done
  v.broadcast_block_sizes()

  # get number of blocks
  nblocks = v.nblocks

  # get size
  size = v.size

  # printing info of the block vector
  if rank == 0:
      print(v)

  # get more information
  for bid in v.owned_blocks:
      print("block", bid, "equals", v[bid],"and is owned by", rank)

  # some basic operations
  # cumpute the sum in parallel
  res = np.sum(v) # this is the same as v.sum()
  # compute absolute values
  res = np.absolute(v)
  # scale vector in parallel
  res = 3.0 * v
  # compute dot product
  res = v.dot(v)
  # add two vectors
  res = np.add(v, v) # this is the same as v + v

  # make a copy of all blocks in all processors
  v2 = v.make_local_copy()
  v2.pprint()

  # make a copy with different ownership
  v3 = v.make_new_MPIBlockVector([1,0,1,0])
  v3.pprint()

Some basic functionality of the :py:class:`MPIBlockMatrix
<pyomo.contrib.pynumero.sparse.mpi_block_matrix.MPIBlockMatrix>` is
presented below:

.. code-block:: python

  # This code is available in pyomo.contrib.pynumero.examples
  # to run the code use the following command
  # mpirun -n 2 python -m mpi4py mpi_block_matrix_example.py
  from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
  from scipy.sparse import coo_matrix
  from mpi4py import MPI
  import numpy as np

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  # create blocks
  row = np.array([0, 3, 1, 2, 3, 0])
  col = np.array([0, 0, 1, 2, 3, 3])
  data = np.array([2., 1, 3, 4, 5, 1])
  m = coo_matrix((data, (row, col)), shape=(4, 4))
  m2 = m.tocsc()

  # create parallel block matrix
  rank_ownership = [[0, -1], [-1, 1]]
  bm = MPIBlockMatrix(2, 2, rank_ownership, comm)

  # block 0,0 lives in processor 0
  # block 1,1 lives in processor 1
  # blocks 0,1 and 1,0 live in both processors
  if rank == 0:
      bm[0, 0] = m
  if rank == 1:
      bm[1, 1] = m2

  # inform other processors of the sizes of the blocks
  # this method has to be called after assignment of blocks is done
  bm.broadcast_block_sizes()

  # get block shape
  nbrows, nbcols = bm.bshape

  # get shape
  nrows, ncols = bm.shape

  # get number of nonzeros
  nnz = bm.nnz

  # transpose matrix
  bm_t = bm.T

  # scale matrix
  scaled_bm = 2.0 * bm

  # some basic operations
  res1 = bm + bm
  res2 = bm - bm

  # looping using owned_blocks
  for i, j in bm.owned_blocks:
      if bm[i,j] is not None:
          print(rank, i, j, bm[i,j].shape)
      else:
          print(rank, i, j, None)


  # looping using ownership_mask
  rows, columns = np.nonzero(bm.ownership_mask)
  for i, j in zip(rows, columns):
      if bm[i, j] is not None:
          print(rank, i, j, bm[i, j].shape)
      else:
          print(rank, i, j, None)

.. note:: We highlight that the interfaces of the block-algebra classes
   are very similar to the mpi-block-algebra classes. Hence, if
   algorithms are written properly it is possible to invoke the same
   algorithm implementation for the parallel or the serial code.  This
   is important for developing decomposition algorithms where often a
   serial implementation is developed for comparison purposes and
   profiling. With our mpi-block-algebra classes extending the serial
   implementation of optimization algorithms is intended to be
   straightforward.

   Differently than :py:class:`BlockVectors
   <pyomo.contrib.pynumero.sparse.block_vector.BlockVector>` and
   :py:class:`BlockMatrix
   <pyomo.contrib.pynumero.sparse.block_matrix.BlockMatrix>`, the
   mpi-block-classes require the user to specify the
   rank-ownership. That is the rank of the processor where each block
   lives. For blocks that live in all processors the rank-ownership is
   -1.
   
   To keep blocks within different processors, the mpi-block-classes do
   not support **flatten** or **tocoo**, **tocsc**, **tocsr** or
   **toarray** operations.

   The key for good scaling of parallel algorithms is to minimize
   communication between processors. When using the
   block-algebra-classes is very important to know in which processor
   are the blocks allocated. For example, when performing a dot product,
   blocks of both vectors must have the same rank_owenership. That
   means, if block 1 lives in processor 1 in the first vector, block 1
   in the second vector must live in the same processor as well. This
   principle applies for any vector-vector and matrix-vector operations.

   The :py:class:`MPIBlockVector
   <pyomo.contrib.pynumero.sparse.mpi_block_vector.MPIBlockVector>`
   supports numpy ``ufuncs`` but differently than :py:class:`BlockVector
   <pyomo.contrib.pynumero.sparse.block_vector.BlockVector>` it executes
   the operation in parallel using MPI functionality.
   
   
   
