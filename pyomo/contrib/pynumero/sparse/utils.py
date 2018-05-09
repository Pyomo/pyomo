from pyutilib.misc.timing import tic, toc
from pynumero.sparse import (COOMatrix,
                             COOSymMatrix,
                             CSCMatrix,
                             CSCSymMatrix,
                             CSRMatrix,
                             CSRSymMatrix)

import numpy as np



def read_matrix(filename, matrix_format='coo', verbose=False, offset=1):

    with open(filename, 'r') as f:
        header = f.readline()
        dimensions = f.readline().split()
        m = int(dimensions[0])
        n = int(dimensions[1])
        nnz = int(dimensions[2])

        if verbose:
            print(header)
            print("n_rows: {}".format(m))
            print("n_cols: {}".format(n))
            print("n_nz: {}".format(nnz))

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
            name = "unamed_matrix"
        f.write("Matrix {}\n".format(name))
        f.write("{} {} {}\n".format(m.shape[0], m.shape[1], m.nnz))
        for i in range(m.nnz):
            f.write("{} {} {}\n".format(m.row[i] + offset, m.col[i] + offset, m.data[i]))
