#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from scipy.sparse import isspmatrix_coo, coo_matrix
import numpy as np

try:
    import mumps
except ImportError as e:
    raise ImportError('Error importing mumps. Install pymumps '
                      'conda install -c conda-forge pymumps')

from pyomo.contrib.pynumero.sparse.utils import is_symmetric_sparse
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector


class MumpsCentralizedAssembledLinearSolver(object):
    """
    A thin wrapper around pymumps which uses the centralized assembled matrix format. 
    In other words ICNTL(5) = 0 and ICNTL(18) = 0.

    Solve matrix * x = rhs for x.

    See the Mumps documentation for descriptions of the parameters. The section numbers 
    listed below refer to the Mumps documentation for version 5.2.1.

    Parameters
    ----------
    sym: int, optional
        See section 5.2.1 of the Mumps documentation
    par: int, optional
        See section 5.1.3
    comm: mpi4py comm, optional
        See section 5.1.3
    cntl_options: dict, optional
        See section 6.2
    icntl_options: dict, optional
        See section 6.1
    """
    def __init__(self, sym=0, par=1, comm=None, cntl_options=None, icntl_options=None):
        self._nnz = None
        self._dim = None
        self.mumps = mumps.DMumpsContext(sym=sym, par=par, comm=comm)
        self.mumps.set_silent()
        if cntl_options is None:
            cntl_options = dict()
        if icntl_options is None:
            icntl_options = dict()
        for k, v in cntl_options.items():
            self.set_cntl(k, v)
        for k, v in icntl_options.items():
            self.set_icntl(k, v)
        
    def do_symbolic_factorization(self, matrix):
        """
        Perform Mumps analysis. 

        Parameters
        ----------
        matrix: scipy.sparse.spmatrix or pyomo.contrib.pynumero.sparse.BlockMatrix
            This matrix must have the same nonzero structure as the matrix passed into
            do_numeric_factorization. The matrix will be converted to coo format if it 
            is not already in coo format. If sym is 1 or 2, the matrix must be lower 
            or upper triangular.
        """
        if type(matrix) == np.ndarray:
            matrix = coo_matrix(matrix)
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('matrix is not square')
        self._dim = nrows
        self._nnz = matrix.nnz
        self.mumps.set_shape(nrows)
        self.mumps.set_centralized_assembled_rows_cols(matrix.row + 1, matrix.col + 1)
        self.mumps.run(job=1)

    def do_numeric_factorization(self, matrix):
        """
        Perform Mumps factorization. Note that do_symbolic_factorization should be called 
        before do_numeric_factorization. 

        Parameters
        ----------
        matrix: scipy.sparse.spmatrix or pyomo.contrib.pynumero.sparse.BlockMatrix
            This matrix must have the same nonzero structure as the matrix passed into
            do_symbolic_factorization. The matrix will be converted to coo format if it 
            is not already in coo format. If sym is 1 or 2, the matrix must be lower 
            or upper triangular.
        """
        if self._nnz is None:
            raise RuntimeError('Call do_symbolic_factorization first.')
        if type(matrix) == np.ndarray:
            matrix = coo_matrix(matrix)
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('matrix is not square')
        if self._dim != nrows:
            raise ValueError('The shape of the matrix changed between symbolic and numeric factorization')
        if self._nnz != matrix.nnz:
            raise ValueError('The number of nonzeros changed between symbolic and numeric factorization')
        self.mumps.set_centralized_assembled_values(matrix.data)
        self.mumps.run(job=2)

    def do_back_solve(self, rhs):
        """
        Perform back solve with Mumps. Note that both do_symbolic_factorization and 
        do_numeric_factorization should be called before do_back_solve. 

        Parameters
        ----------
        rhs: numpy.ndarray or pyomo.contrib.pynumero.sparse.BlockVector
            The right hand side in matrix * x = rhs.
        
        Returns
        -------
        result: numpy.ndarray or pyomo.contrib.pynumero.sparse.BlockVector
            The x in matrix * x = rhs. If rhs is a BlockVector, then, result 
            will be a BlockVector with the same block structure as rhs.
        """
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
            result = _rhs
        else:
            result = rhs.copy()

        self.mumps.set_rhs(result)
        self.mumps.run(job=3)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result
        
        return result

    def __del__(self):
        self.mumps.destroy()

    def set_icntl(self, key, value):
        self.mumps.set_icntl(key, value)

    def set_cntl(self, key, value):
        self.mumps.id.cntl[key-1] = value

    def solve(self, matrix, rhs):
        self.do_symbolic_factorization(matrix)
        self.do_numeric_factorization(matrix)
        return self.do_back_solve(rhs)

    def get_info(self, key):
        return self.mumps.id.info[key-1]

    def get_infog(self, key):
        return self.mumps.id.infog[key-1]

    def get_rinfo(self, key):
        return self.mumps.id.rinfo[key-1]

    def get_rinfog(self, key):
        return self.mumps.id.rinfog[key-1]
