#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.sparse.base import SparseBase
from pyomo.contrib.pynumero.sparse.extract import triu, tril
from scipy.sparse.sputils import isscalarlike

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


def _is_symmetric_numerically(mat):

    # Note: this check is expensive
    flag = False
    if isinstance(mat, np.ndarray):
        flag = is_symmetric_dense(mat)
    elif isscalarlike(mat):
        flag = True
    elif isinstance(mat, SparseBase):
        if mat.is_symmetric:
            flag = mat.is_symmetric
        else:
            if mat.shape[0] != mat.shape[1]:
                flag = False
            else:
                # get upper and lower triangular
                from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
                if isinstance(mat, BlockMatrix):
                    mat = mat.tofullmatrix()
                l = tril(mat)
                u = triu(mat)
                diff = l - u.transpose()
                z = np.zeros(diff.nnz)
                flag = np.allclose(diff.data, z, atol=1e-6)
    else:
        if mat.shape[0] != mat.shape[1]:
            flag = False
        else:
            # get upper and lower triangular
            l = tril(mat)
            u = triu(mat)
            diff = l - u.transpose()
            z = np.zeros(diff.nnz)
            flag = np.allclose(diff.data, z, atol=1e-6)
    return flag


def _convert_matrix_to_symmetric(mat, check_symmetry=True):

    if not isinstance(mat, SparseBase):
        raise RuntimeError("Operation only supported for pynumero matrices")

    if mat.is_symmetric:
        return mat

    if check_symmetry and not _is_symmetric_numerically(mat):
        err_msg = "Cannot convert matrix because it has no symmetry"
        raise RuntimeError(err_msg)

    from pyomo.contrib.pynumero.sparse import (COOSymMatrix,
                                               CSCMatrix,
                                               CSCSymMatrix,
                                               CSRMatrix,
                                               CSRSymMatrix)

    l = tril(mat)
    if isinstance(mat, CSCMatrix):
        return CSCSymMatrix(l)
    elif isinstance(mat, CSRMatrix):
        return CSRSymMatrix(l)
    else:
        return COOSymMatrix(l)
