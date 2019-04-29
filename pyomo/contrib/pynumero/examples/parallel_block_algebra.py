#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from scipy.sparse import coo_matrix
from mpi4py import MPI
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    raise RuntimeError("These example needs mpi4py")

if comm.Get_size() < 2:
    msg = "These example needs at least 2 processors." + \
        "Call the example as mpirun -n 2 python -m mpi4py parallel_block_algebra.py"
    raise RuntimeError(msg)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


row = np.array([0, 3, 1, 2, 3, 0])
col = np.array([0, 0, 1, 2, 3, 3])
data = np.array([2., 1, 3, 4, 5, 1])
m = coo_matrix((data, (row, col)), shape=(4, 4))

# create parallel matrix
rank_ownership = [[0, -1], [-1, 1]]
bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
if rank == 0:
    bm[0, 0] = m
if rank == 1:
    bm[1, 1] = m
bm[0, 1] = m

# communicate block sizes information (need to be done once)
bm.broadcast_block_sizes()

matrix_shape = bm.shape
matrix_blocks = bm.block_shapes()
bm.pprint()

if rank == 0:
    print("Shape:",matrix_shape)
    print("Block sizes", matrix_blocks)

# looping through the elements of a parallel matrix
for i in range(bm.bshape[0]):
    for j in range(bm.bshape[1]):
        owner = bm.rank_ownership[i, j]
        if owner == rank:
            if bm[i, j] is not None:
                print("bm[{},{}] = \n{}".format(i, j, bm[i, j].toarray()))
            else:
                print("bm[{},{}] = None".format(i, j))
        if owner == -1:
            if rank == 0:
                if bm[i, j] is not None:
                    print("bm[{},{}] = \n{}".format(i, j, bm[i, j].toarray()))
                else:
                    print("bm[{},{}] = None".format(i, j))

# alternative one can loop using the ownership attribute
# rows, columns = np.nonzero(bm.ownership_mask)
# for i, j in zip(rows, columns):
#     if bm[i, j] is not None:
#         print("bm[{},{}] = \n{}".format(i, j, bm[i, j].toarray()))
#     else:
#         print("bm[{},{}] = None".format(i, j))


# add parallel matrix
res = bm + bm

# substract a block matrix
serial_bm = BlockMatrix(2, 2)
serial_bm[0, 0] = m
serial_bm[1, 1] = m

res = bm - serial_bm

# getting the transpose
bm_transposed = bm.T

# matrix vector multiply in parallel
bv = MPIBlockVector(2, [0, 1], comm)
if rank == 0:
    bv[0] = np.arange(4, dtype=np.float64)
if rank == 1:
    bv[1] = np.arange(4, dtype=np.float64) + 4

res = bm.dot(bv)
res.pprint()

# looping through the elements of a parallel vector
for i in range(res.nblocks):
    owner = res.rank_ownership[i]
    if owner == rank:
        print("res[{}] = {}".format(i, res[i]))
    if owner == -1:
        if rank == 0:
            print("res[{}] = {}".format(i, res[i]))

# alternative one can loop using the ownership attribute
# owned_blocks = np.nonzero(res.ownership_mask)[0]
# for i in owned_blocks:
#     print("res[{}] = {}".format(i, res[i]))

# dot product
res = res.dot(res)
if rank == 0:
    print("The dot product is:", res)
