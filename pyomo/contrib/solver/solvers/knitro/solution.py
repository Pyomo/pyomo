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
from pyomo.contrib.solver.solvers.knitro.typing import ItemType, ValueType
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData


class SolutionProvider(Protocol):

    def get_num_solutions(self) -> int: ...
    def get_values(
        self,
        item_type: type[ItemType],
        value_type: ValueType,
        items: Optional[Sequence[ItemType]] = None,
        *,
        exists: bool,
        solution_id: Optional[int] = None,
    ) -> Mapping[ItemType, float]: ...


class SolutionLoader(SolutionLoaderBase):
    _provider: SolutionProvider
    has_primals: bool
    has_reduced_costs: bool
    has_duals: bool

    def __init__(
        self,
        provider: SolutionProvider,
        *,
        has_primals: bool = True,
        has_reduced_costs: bool = True,
        has_duals: bool = True,
    ):
        super().__init__()
        self._provider = provider
        self.has_primals = has_primals
        self.has_reduced_costs = has_reduced_costs
        self.has_duals = has_duals

    def get_number_of_solutions(self) -> int:
        return self._provider.get_num_solutions()

    # TODO: remove this when the solution loader is fixed.
    def get_primals(self, vars_to_load=None):
        return self.get_vars(vars_to_load)

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        return self._provider.get_values(
            VarData,
            ValueType.PRIMAL,
            vars_to_load,
            exists=self.has_primals,
            solution_id=solution_id,
        )

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=None
    ) -> Mapping[VarData, float]:
        return self._provider.get_values(
            VarData,
            ValueType.DUAL,
            vars_to_load,
            exists=self.has_reduced_costs,
            solution_id=solution_id,
        )

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None, solution_id=None
    ) -> Mapping[ConstraintData, float]:
        return self._provider.get_values(
            ConstraintData,
            ValueType.DUAL,
            cons_to_load,
            exists=self.has_duals,
            solution_id=solution_id,
        )
