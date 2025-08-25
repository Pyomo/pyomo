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

from collections.abc import Mapping, Sequence
from typing import Optional

from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.contrib.solver.common.util import NoDualsError, NoSolutionError
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData

from .base import SolverBase


class SolutionLoader(SolutionLoaderBase):
    def __init__(self, solver: SolverBase):
        super().__init__()
        self._solver = solver

    def get_number_of_solutions(self) -> int:
        return self._solver.get_num_solutions()

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if vars_to_load is None:
            vars_to_load = self._solver.get_vars()

        x = self._solver.get_primals(vars_to_load)
        if x is None:
            return NoSolutionError()
        return ComponentMap([(var, x[i]) for i, var in enumerate(vars_to_load)])

    # TODO: remove this when the solution loader is fixed.
    def get_primals(self, vars_to_load=None):
        return self.get_vars(vars_to_load)

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]:
        if cons_to_load is None:
            cons_to_load = self._solver.get_cons()

        y = self._solver.get_duals(cons_to_load)
        if y is None:
            return NoDualsError()
        return ComponentMap([(con, y[i]) for i, con in enumerate(cons_to_load)])
