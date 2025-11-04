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

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import ApplicationError, DeveloperError, PyomoException
from pyomo.common.numeric_types import value
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoOptimalSolutionError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.contrib.solver.solvers.knitro.config import KnitroConfig
from pyomo.contrib.solver.solvers.knitro.engine import Engine
from pyomo.contrib.solver.solvers.knitro.package import PackageChecker
from pyomo.contrib.solver.solvers.knitro.solution import (
    SolutionLoader,
    SolutionProvider,
)
from pyomo.contrib.solver.solvers.knitro.typing import ItemData, ItemType, ValueType
from pyomo.contrib.solver.solvers.knitro.utils import KnitroModelData
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager


class KnitroSolverBase(SolutionProvider, PackageChecker, SolverBase):
    CONFIG = KnitroConfig()
    config: KnitroConfig

    _engine: Engine
    _model_data: KnitroModelData
    _stream: StringIO
    _saved_var_values: dict[int, Optional[float]]

    def __init__(self, **kwds) -> None:
        PackageChecker.__init__(self)
        SolverBase.__init__(self, **kwds)
        self._engine = Engine()
        self._model_data = KnitroModelData()
        self._stream = StringIO()
        self._saved_var_values = {}

    def solve(self, model: BlockData, **kwds) -> Results:
        tick = datetime.now(timezone.utc)
        self._check_available()

        config = self._build_config(**kwds)
        timer = config.timer or HierarchicalTimer()

        StaleFlagManager.mark_all_as_stale()

        self._presolve(model, config, timer)
        self._validate_problem()

        self._stream = StringIO()
        if config.restore_variable_values_after_solve:
            self._save_var_values()

        with capture_output(TeeStream(self._stream, *config.tee), capture_fd=True):
            self._solve(config, timer)

        if config.restore_variable_values_after_solve:
            self._restore_var_values()

        results = self._postsolve(config, timer)

        tock = datetime.now(timezone.utc)

        results.timing_info.start_timestamp = tick
        results.timing_info.wall_time = (tock - tick).total_seconds()
        return results

    def _build_config(self, **kwds) -> KnitroConfig:
        return self.config(value=kwds, preserve_implicit=True)  # type: ignore

    def _validate_problem(self) -> None:
        if len(self._model_data.objs) > 1:
            msg = f"{self.name} does not support multiple objectives."
            raise IncompatibleModelError(msg)

    def _check_available(self) -> None:
        avail = self.available()
        if not avail:
            msg = f"Solver {self.name} is not available: {avail}."
            raise ApplicationError(msg)

    def _save_var_values(self) -> None:
        self._saved_var_values.clear()
        for var in self._get_vars():
            self._saved_var_values[id(var)] = value(var.value)

    def _restore_var_values(self) -> None:
        StaleFlagManager.mark_all_as_stale(delayed=True)
        for var in self._get_vars():
            var.set_value(self._saved_var_values[id(var)])
        StaleFlagManager.mark_all_as_stale()

    @abstractmethod
    def _presolve(
        self, model: BlockData, config: KnitroConfig, timer: HierarchicalTimer
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _solve(self, config: KnitroConfig, timer: HierarchicalTimer) -> None:
        raise NotImplementedError

    def _postsolve(self, config: KnitroConfig, timer: HierarchicalTimer) -> Results:
        status = self._engine.get_status()
        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_log = self._stream.getvalue()
        results.solver_config = config
        results.solution_status = self._get_solution_status(status)
        results.termination_condition = self._get_termination_condition(status)
        results.incumbent_objective = self._engine.get_obj_value()
        results.extra_info.iteration_count = self._engine.get_num_iters()
        results.timing_info.solve_time = self._engine.get_solve_time()
        results.timing_info.timer = timer

        if (
            config.raise_exception_on_nonoptimal_result
            and results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
        ):
            raise NoOptimalSolutionError()

        results.solution_loader = SolutionLoader(
            self,
            has_primals=results.solution_status
            not in {SolutionStatus.infeasible, SolutionStatus.noSolution},
            has_reduced_costs=results.solution_status == SolutionStatus.optimal,
            has_duals=results.solution_status
            not in {SolutionStatus.infeasible, SolutionStatus.noSolution},
        )
        if config.load_solutions:
            timer.start("load_solutions")
            results.solution_loader.load_vars()
            timer.stop("load_solutions")

        return results

    def get_values(
        self,
        item_type: type[ItemType],
        value_type: ValueType,
        items: Optional[Sequence[ItemType]] = None,
        *,
        exists: bool,
        solution_id: Optional[int] = None,
    ) -> Mapping[ItemType, float]:
        error_type = self._get_error_type(item_type, value_type)
        if not exists:
            raise error_type()
        # KNITRO only supports a single solution
        assert solution_id is None
        if items is None:
            items = self._get_items(item_type)
        x = self._engine.get_values(item_type, value_type, items)
        if x is None:
            raise error_type()
        sign = value_type.sign
        return ComponentMap([(k, sign * xk) for k, xk in zip(items, x)])

    def get_num_solutions(self) -> int:
        return self._engine.get_num_solutions()

    def _get_vars(self) -> list[VarData]:
        return self._model_data.variables

    def _get_items(self, item_type: type[ItemType]) -> Sequence[ItemType]:
        maps = {
            VarData: self._model_data.variables,
            ConstraintData: self._model_data.cons,
        }
        return maps[item_type]

    @staticmethod
    def _get_solution_status(status: int) -> SolutionStatus:
        """
        Map KNITRO status codes to Pyomo SolutionStatus values.

        See https://www.artelys.com/app/docs/knitro/3_referenceManual/returnCodes.html
        """
        if status in {0, -100}:
            return SolutionStatus.optimal
        elif -101 >= status >= -199 or -400 >= status >= -409:
            return SolutionStatus.feasible
        elif status in {-200, -204, -205, -206}:
            return SolutionStatus.infeasible
        else:
            return SolutionStatus.noSolution

    @staticmethod
    def _get_termination_condition(status: int) -> TerminationCondition:
        """
        Map KNITRO status codes to Pyomo TerminationCondition values.

        See https://www.artelys.com/app/docs/knitro/3_referenceManual/returnCodes.html
        """
        if status in {0, -100}:
            return TerminationCondition.convergenceCriteriaSatisfied
        elif status == -202:
            return TerminationCondition.locallyInfeasible
        elif status in {-200, -204, -205}:
            return TerminationCondition.provenInfeasible
        elif status in {-300, -301}:
            return TerminationCondition.infeasibleOrUnbounded
        elif status in {-400, -410}:
            return TerminationCondition.iterationLimit
        elif status in {-401, -411}:
            return TerminationCondition.maxTimeLimit
        elif status == -500:
            return TerminationCondition.interrupted
        elif -500 > status >= -599:
            return TerminationCondition.error
        else:
            return TerminationCondition.unknown

    @staticmethod
    def _get_error_type(
        item_type: type[ItemData], value_type: ValueType
    ) -> type[PyomoException]:
        if item_type is VarData and value_type == ValueType.PRIMAL:
            return NoSolutionError
        elif item_type is VarData and value_type == ValueType.DUAL:
            return NoReducedCostsError
        elif item_type is ConstraintData and value_type == ValueType.DUAL:
            return NoDualsError
        raise DeveloperError(
            f"Unsupported KNITRO item type {item_type} and value type {value_type}."
        )
