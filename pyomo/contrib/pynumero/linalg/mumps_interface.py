#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from scipy.sparse import isspmatrix_coo, coo_matrix, tril, spmatrix
import numpy as np
from .base import DirectLinearSolverInterface, LinearSolverResults, LinearSolverStatus
from typing import Union, Tuple, Optional

from pyomo.common.dependencies import attempt_import

mumps, mumps_available = attempt_import(
    'mumps',
    error_message="Error importing mumps. PyNumero's "
    "mumps_interface requires pymumps; install it with, e.g., "
    "'conda install -c conda-forge pymumps'",
)

from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix


class MumpsCentralizedAssembledLinearSolver(DirectLinearSolverInterface):
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
        self._mumps = None
        self._mumps = mumps.DMumpsContext(sym=sym, par=par, comm=comm)
        self._mumps.set_silent()
        self._icntl_options = dict()
        self._cntl_options = dict()
        self._sym = sym

        if cntl_options is None:
            cntl_options = dict()
        if icntl_options is None:
            icntl_options = dict()
        for k, v in cntl_options.items():
            self.set_cntl(k, v)
        for k, v in icntl_options.items():
            self.set_icntl(k, v)
        self._prev_allocation = None

    def _init(self):
        """
        The purpose of this method is to address issue #12 from pymumps
        """
        self._mumps.run(job=-1)
        self._mumps.set_silent()
        for k, v in self._cntl_options.items():
            self.set_cntl(k, v)
        for k, v in self._icntl_options.items():
            self.set_icntl(k, v)

    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        """
        Perform Mumps analysis.

        Parameters
        ----------
        matrix: scipy.sparse.spmatrix or pyomo.contrib.pynumero.sparse.BlockMatrix
            This matrix must have the same nonzero structure as the matrix passed into
            do_numeric_factorization. The matrix will be converted to coo format if it
            is not already in coo format. If sym is 1 or 2, the matrix will be converted
            to lower triangular.
        """
        self._init()
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        if self._sym in {1, 2}:
            matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('matrix is not square')
        self._dim = nrows
        self._nnz = matrix.nnz
        try:
            self._mumps.set_shape(nrows)
            self._mumps.set_centralized_assembled_rows_cols(
                matrix.row + 1, matrix.col + 1
            )
            self._mumps.run(job=1)
            self._prev_allocation = max(self.get_infog(16), self.get_icntl(23))
            # INFOG(16) is the Mumps estimate for memory usage; ICNTL(23)
            # is the override used in increase_memory_allocation. Both are
            # already rounded to MB, so neither should every be negative.
        except RuntimeError as err:
            if raise_on_error:
                raise err

        stat = self.get_infog(1)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        elif stat in {-6, -10}:
            res.status = LinearSolverStatus.singular
        elif stat < 0:
            res.status = LinearSolverStatus.error
        else:
            res.status = LinearSolverStatus.warning
        return res

    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        """
        Perform Mumps factorization. Note that do_symbolic_factorization should be called
        before do_numeric_factorization.

        Parameters
        ----------
        matrix: scipy.sparse.spmatrix or pyomo.contrib.pynumero.sparse.BlockMatrix
            This matrix must have the same nonzero structure as the matrix passed into
            do_symbolic_factorization. The matrix will be converted to coo format if it
            is not already in coo format. If sym is 1 or 2, the matrix will be converted
            to lower triangular.
        """
        if self._nnz is None:
            raise RuntimeError('Call do_symbolic_factorization first.')
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        if self._sym in {1, 2}:
            matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('matrix is not square')
        if self._dim != nrows:
            raise ValueError(
                'The shape of the matrix changed between symbolic and numeric factorization'
            )
        if self._nnz != matrix.nnz:
            raise ValueError(
                'The number of nonzeros changed between symbolic and numeric factorization'
            )
        try:
            self._mumps.set_centralized_assembled_values(matrix.data)
            self._mumps.run(job=2)
        except RuntimeError as err:
            if raise_on_error:
                raise err

        stat = self.get_infog(1)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        elif stat in {-6, -10}:
            res.status = LinearSolverStatus.singular
        elif stat in {-8, -9}:
            res.status = LinearSolverStatus.not_enough_memory
        elif stat < 0:
            res.status = LinearSolverStatus.error
        else:
            res.status = LinearSolverStatus.warning
        return res

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
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

        self._mumps.set_rhs(result)
        self._mumps.run(job=3)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result, LinearSolverResults(LinearSolverStatus.successful)

    def increase_memory_allocation(self, factor):
        # info(16) is rounded to the nearest MB, so it could be zero
        if self._prev_allocation == 0:
            new_allocation = 1
        else:
            new_allocation = int(factor * self._prev_allocation)
        # Here I set the memory allocation directly instead of increasing
        # the "percent-increase-from-predicted" parameter ICNTL(14)
        self.set_icntl(23, new_allocation)
        self._prev_allocation = new_allocation
        return new_allocation

    def __del__(self):
        if getattr(self, '_mumps', None) is not None:
            self._mumps.destroy()

    def set_icntl(self, key, value):
        self._icntl_options[key] = value
        self._mumps.set_icntl(key, value)

    def set_cntl(self, key, value):
        self._cntl_options[key] = value
        self._mumps.id.cntl[key - 1] = value

    def get_icntl(self, key):
        return self._mumps.id.icntl[key - 1]

    def get_cntl(self, key):
        return self._mumps.id.cntl[key - 1]

    def get_info(self, key):
        return self._mumps.id.info[key - 1]

    def get_infog(self, key):
        return self._mumps.id.infog[key - 1]

    def get_rinfo(self, key):
        return self._mumps.id.rinfo[key - 1]

    def get_rinfog(self, key):
        return self._mumps.id.rinfog[key - 1]
