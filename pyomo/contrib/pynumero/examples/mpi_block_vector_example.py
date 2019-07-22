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
