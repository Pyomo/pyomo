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
