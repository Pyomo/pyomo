from pyomo.contrib.pynumero.sparse.base import SparseBase
from pyomo.contrib.pynumero.sparse.extract import triu, tril
from scipy.sparse.sputils import isscalarlike

import numpy as np


"""
def read_matrix(filename, matrix_format='coo', verbose=False, offset=1):
    from pyomo.contrib.pynumero.sparse import (COOMatrix,
                                               COOSymMatrix,
                                               CSCMatrix,
                                               CSCSymMatrix,
                                               CSRMatrix,
                                               CSRSymMatrix)


    with open(filename, 'r') as f:
        header = f.readline()
        dimensions = f.readline().split()
        m = int(dimensions[0])
        n = int(dimensions[1])
        nnz = int(dimensions[2])

        if verbose:
            print(header)
            print('n_rows: {}'.format(m))
            print('n_cols: {}'.format(n))
            print('n_nz: {}'.format(nnz))

        irows = np.zeros(nnz)
        jcols = np.zeros(nnz)
        data = np.zeros(nnz)
        counter = 0
        for line in f:
            if '%' not in line:
                entries = line.split()
                if len(entries) != 3:
                    raise RuntimeError('Wrong format')
                irows[counter] = int(entries[0]) - offset
                jcols[counter] = int(entries[1]) - offset
                data[counter] = float(entries[2])
                counter += 1

        assert counter == nnz, 'wrong number of entries'

        if 'symmetric' in header:
            if matrix_format == 'coo':
                return COOSymMatrix((data, (irows, jcols)), shape=(m, n))
            elif matrix_format == 'csc':
                return CSCSymMatrix((data, (irows, jcols)), shape=(m, n))
            elif matrix_format == 'csr':
                return CSRSymMatrix((data, (irows, jcols)), shape=(m, n))
            else:
                raise RuntimeError('Matrix format not supported')
        else:
            if matrix_format == 'coo':
                return COOMatrix((data, (irows, jcols)), shape=(m, n))
            elif matrix_format == 'csc':
                return CSCMatrix((data, (irows, jcols)), shape=(m, n))
            elif matrix_format == 'csr':
                return CSRMatrix((data, (irows, jcols)), shape=(m, n))
            else:
                raise RuntimeError('Matrix format not supported')


def write_matrix(filename, matrix, offset=1):

    with open(filename, 'w') as f:
        m = matrix.tocoo()
        name = m.name
        if name is None:
            name = 'unamed_matrix'
        f.write('Matrix {}\n'.format(name))
        f.write('{} {} {}\n'.format(m.shape[0], m.shape[1], m.nnz))
        for i in range(m.nnz):
            f.write('{} {} {}\n'.format(m.row[i] + offset, m.col[i] + offset, m.data[i]))
"""

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
