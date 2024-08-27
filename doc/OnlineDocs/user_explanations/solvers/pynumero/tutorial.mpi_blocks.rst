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

.. literalinclude:: /../../pyomo/contrib/pynumero/examples/parallel_vector_ops.py

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
same operations performed with `BlockVector` (or even NumPy arrays),
except that the operations are now performed in parallel.

`MPIBlockMatrix` construction is very similar. Consider the following
example in a file called "parallel_matvec.py".

.. literalinclude:: /../../pyomo/contrib/pynumero/examples/parallel_matvec.py

Which can be run with

.. code-block::

   mpirun -np 3 python -m mpi4py parallel_matvec.py

The output is

.. code-block::

   error:  4.440892098500626e-16

The most difficult part of using `MPIBlockVector` and `MPIBlockMatrix`
is determining the best structure and rank ownership to maximize
parallel efficiency.
