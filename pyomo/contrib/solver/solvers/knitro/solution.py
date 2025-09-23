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
from typing import Optional, Protocol, Union, overload

from pyomo.common.errors import PyomoException
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.contrib.solver.common.util import (
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData

from .typing import UnreachableError, ValueType


class SolutionProvider(Protocol):
    @staticmethod
    def get_error_type(item_type: type, value_type: ValueType) -> type[PyomoException]:
        if item_type is VarData and value_type == ValueType.PRIMAL:
            return NoSolutionError
        elif item_type is VarData and value_type == ValueType.DUAL:
            return NoReducedCostsError
        elif item_type is ConstraintData and value_type == ValueType.DUAL:
            return NoDualsError
        raise UnreachableError()

    def get_num_solutions(self) -> int: ...
    @overload
    def get_values(
        self,
        item_type: type[VarData],
        value_type: ValueType,
        items: Optional[Sequence[VarData]] = None,
    ) -> Mapping[VarData, float]: ...
    @overload
    def get_values(
        self,
        item_type: type[ConstraintData],
        value_type: ValueType,
        items: Optional[Sequence[ConstraintData]] = None,
    ) -> Mapping[ConstraintData, float]: ...


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
        item_type: type,
        value_type: ValueType,
        items: Optional[Union[Sequence[VarData], Sequence[ConstraintData]]] = None,
        *,
        exists: bool,
    ):
        if not exists:
            error_type = SolutionProvider.get_error_type(item_type, value_type)
            raise error_type()
        return self._provider.get_values(item_type, value_type, items)

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        return self.get_values(
            VarData, ValueType.PRIMAL, vars_to_load, exists=self.has_primals
        )

    # TODO: remove this when the solution loader is fixed.
    def get_primals(self, vars_to_load=None):
        return self.get_vars(vars_to_load)

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        return self.get_values(
            VarData, ValueType.DUAL, vars_to_load, exists=self.has_reduced_costs
        )

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Mapping[ConstraintData, float]:
        return self.get_values(
            ConstraintData, ValueType.DUAL, cons_to_load, exists=self.has_duals
        )
