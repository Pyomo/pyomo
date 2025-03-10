#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from .base_linear_solver_interface import IPLinearSolverInterface
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus, LinearSolverResults
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from scipy.sparse import isspmatrix_coo, tril, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from typing import Union


class InteriorPointMA27Interface(MA27, IPLinearSolverInterface):
    @classmethod
    def getLoggerName(cls):
        return 'ma27'

    def __init__(
        self, cntl_options=None, icntl_options=None, iw_factor=1.2, a_factor=2
    ):
        super(InteriorPointMA27Interface, self).__init__(
            cntl_options=cntl_options,
            icntl_options=icntl_options,
            iw_factor=iw_factor,
            a_factor=a_factor,
        )
        self._num_status = None

    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        self._num_status = None
        return super(InteriorPointMA27Interface, self).do_symbolic_factorization(
            matrix=matrix, raise_on_error=raise_on_error
        )

    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        res = super(InteriorPointMA27Interface, self).do_numeric_factorization(
            matrix=matrix, raise_on_error=raise_on_error
        )
        self._num_status = res.status
        return res

    def get_inertia(self):
        if self._num_status is None:
            raise RuntimeError(
                'Must call do_numeric_factorization before inertia can be computed'
            )
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError(
                'Can only compute inertia if the numeric factorization was successful.'
            )
        num_negative_eigenvalues = self.get_info(15)
        num_positive_eigenvalues = self._dim - num_negative_eigenvalues
        return (num_positive_eigenvalues, num_negative_eigenvalues, 0)
