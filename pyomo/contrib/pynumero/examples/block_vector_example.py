from pyomo.contrib.pynumero.sparse import BlockVector
import numpy as np

# create some arrays
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# create one block vector with assignment of each block
bv = BlockVector(2)
bv[0] = a
bv[1] = -b

# get number of blocks
nblocks = bv.nblocks

# get overall size of the vector
size = bv.size

# Get unstructured vector
flat_bv = bv.flatten()

# create one block vector directly by passing a list of vectors
bv2 = BlockVector([a, b])

# compute absolute value
res1 = np.absolute(bv)

# compute reciprocal
res2 = np.reciprocal(bv)

# add two block vectors
res3 = np.add(bv,bv2)
res4 = np.add(bv, c)
res5 = bv + c # this is the same to res4 and to c + bv

# building block vectors from arrays or from other BlockVectors
bv.copyfrom(c)
bv.copyfrom(bv2)
new_bv = bv.copy_structure()
bv2.copyto(new_bv)

# creating copies of block vectors
new_bv = bv.copy()
