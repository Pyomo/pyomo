from pyomo.contrib.pynumero.sparse import (BlockMatrix,
                                           BlockVector)
from scipy.sparse import coo_matrix
import numpy as np


# create one block
row = np.array([0, 1, 2, 3])
col = np.array([0, 1, 0, 1])
data = np.array([1., 1., 1., 1.])
m1 = coo_matrix((data, (row, col)), shape=(4, 2))

# create another block
row = np.array([0, 3, 1, 2, 3, 0])
col = np.array([0, 0, 1, 2, 3, 3])
data = np.array([2., 1, 3, 4, 5, 1])
m2 = coo_matrix((data, (row, col)), shape=(4, 4))

# create block matrix
bm = BlockMatrix(2,2)
bm[0, 0] = m1
bm[1, 1] = m2

# get block shape of matrix
nbrows, nbcols = bm.bshape

# get number of nonzero values in matrix
nnz = bm.nnz

# get shape of matrix
nrows, ncols = bm.shape

# create scipy matrices from BlockMatrix
coo_mat = bm.tocoo()
csc_mat = bm.tocsc()
csr_mat = bm.tocsr()

# create numpy array from BlockMatrix
dense_mat = bm.toarray()

# transpose matrix
bm_t = bm.T # or bm.transpose()

# add and substract matrices
res1_bm = bm + bm
res2_bm = bm + coo_mat
res3_bm = csr_mat - bm

# matrix matrix product
rr = bm * bm_t

# get absolute value of a matrix
abs_bm = abs(bm)

# scale a matrix
scaled_bm = 2.0 * bm

# building BlockMatrices from spmatrices or other BlockMatrices
copy_bm = bm.copy()
new_bm = bm.copy_structure()
new_bm.copyfrom(csr_mat)
new_bm.copyfrom(scaled_bm)
