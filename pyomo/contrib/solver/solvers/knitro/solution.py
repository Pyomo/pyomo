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
from typing import Optional, Protocol

from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData


class SolutionProvider(Protocol):
    def get_num_solutions(self) -> int: ...
    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]: ...
    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]: ...


class SolutionLoader(SolutionLoaderBase):
    def __init__(self, provider: SolutionProvider):
        super().__init__()
        self._provider = provider

    def get_number_of_solutions(self) -> int:
        return self._provider.get_num_solutions()

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        return self._provider.get_primals(vars_to_load)

    # TODO: remove this when the solution loader is fixed.
    def get_primals(self, vars_to_load=None):
        return self.get_vars(vars_to_load)

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]:
        return self._provider.get_duals(cons_to_load)
