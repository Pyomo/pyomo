#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from scipy.sparse.sputils import isscalarlike
from scipy.sparse import tril, triu, isspmatrix

import numpy as np


def is_symmetric_dense(mat):

    flag = False
    if isinstance(mat, np.ndarray):
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
            if np.allclose(mat, mat.T, atol=1e-6):
                flag = True
    elif isscalarlike(mat):
        flag = True
    else:
        raise RuntimeError("Format not recognized {}".format(type(mat)))
    return flag


def is_symmetric_sparse(mat):
    from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
    # Note: this check is expensive
    flag = False
    if isinstance(mat, np.ndarray):
        flag = is_symmetric_dense(mat)
    elif isscalarlike(mat):
        flag = True
    elif isspmatrix(mat) or isinstance(mat, BlockMatrix):
        if mat.shape[0] != mat.shape[1]:
            flag = False
        else:
            if isinstance(mat, BlockMatrix):
                mat = mat.tocoo()
            # get upper and lower triangular
            l = tril(mat)
            u = triu(mat)
            diff = l - u.transpose()
            z = np.zeros(diff.nnz)
            flag = np.allclose(diff.data, z, atol=1e-6)
    else:
        raise RuntimeError("Format not recognized {}".format(type(mat)))
    return flag


