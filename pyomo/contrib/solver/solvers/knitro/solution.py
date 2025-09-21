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

from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Protocol, Type, TypeVar, Union

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
    def get_num_solutions(self) -> int: ...
    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]: ...
    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]: ...
    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]: ...


T = TypeVar("T", bound=Union[VarData, ConstraintData])


def get_values(
    to_load: Optional[Sequence[T]],
    loader: Callable[[Optional[Sequence[T]]], Mapping[T, float]],
    is_success: bool,
    error_type: Type[PyomoException],
) -> Mapping[T, float]:
    if not is_success:
        raise error_type()
    return loader(to_load)


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

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        return get_values(
            vars_to_load, self._provider.get_primals, self.has_primals, NoSolutionError
        )

    # TODO: remove this when the solution loader is fixed.
    def get_primals(self, vars_to_load=None):
        return self.get_vars(vars_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        return get_values(
            vars_to_load,
            self._provider.get_reduced_costs,
            self.has_reduced_costs,
            NoReducedCostsError,
        )

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]:
        return get_values(
            cons_to_load, self._provider.get_duals, self.has_duals, NoDualsError
        )
