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
from typing import Optional, Protocol, Type, Union

from pyomo.common.errors import PyomoException
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.contrib.solver.common.util import (
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData


class SolutionProvider(Protocol):

    def get_error_type(
        self, load_type: Type[Union[VarData, ConstraintData]], *, is_dual: bool
    ) -> Type[PyomoException]:
        if load_type is VarData and not is_dual:
            return NoSolutionError
        elif load_type is VarData and is_dual:
            return NoReducedCostsError
        elif load_type is ConstraintData and is_dual:
            return NoDualsError

    def get_num_solutions(self) -> int: ...
    def get_values(
        self,
        load_type: Type[Union[VarData, ConstraintData]],
        to_load: Optional[Union[Sequence[VarData], Sequence[ConstraintData]]],
        *,
        is_dual: bool,
    ) -> Mapping[Union[VarData, ConstraintData], float]: ...


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

    def get_values(
        self,
        load_type: Type[Union[VarData, ConstraintData]],
        to_load: Optional[Union[Sequence[VarData], Sequence[ConstraintData]]],
        *,
        is_dual: bool,
        check: bool,
    ):
        if not check:
            error_type = self._provider.get_error_type(load_type, is_dual=is_dual)
            raise error_type()

        return self._provider.get_values(load_type, to_load, is_dual=is_dual)

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        return self.get_values(VarData, vars_to_load, is_dual=False, check=self.has_primals)

    # TODO: remove this when the solution loader is fixed.
    def get_primals(self, vars_to_load=None):
        return self.get_vars(vars_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        return self.get_values(VarData, vars_to_load, is_dual=True, check=self.has_reduced_costs)

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]:
        return self.get_values(ConstraintData, cons_to_load, is_dual=True, check=self.has_duals)
