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
from pyomo.contrib.pynumero.linalg.base import LinearSolverResults
from scipy.linalg import eigvals
from scipy.sparse import spmatrix
from pyomo.contrib.pynumero.sparse import BlockMatrix
import logging
import numpy as np
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU
from typing import Union


class ScipyInterface(ScipyLU, IPLinearSolverInterface):
    @classmethod
    def getLoggerName(cls):
        return 'ScipyLU'

    def __init__(self, compute_inertia=False):
        super(ScipyInterface, self).__init__()
        self._inertia = None
        self.compute_inertia = compute_inertia

        self.logger = logging.getLogger('scipy')
        self.logger.propagate = False

    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        res = super(ScipyInterface, self).do_numeric_factorization(
            matrix=matrix, raise_on_error=raise_on_error
        )

        if self.compute_inertia:
            eig = eigvals(matrix.toarray())
            pos_eig = np.count_nonzero((eig > 0))
            neg_eigh = np.count_nonzero((eig < 0))
            zero_eig = np.count_nonzero(eig == 0)
            self._inertia = (pos_eig, neg_eigh, zero_eig)

        return res

    def get_inertia(self):
        if self._inertia is None:
            raise RuntimeError(
                'The inertia was not computed during do_numeric_factorization. Set compute_inertia to True.'
            )
        return self._inertia
