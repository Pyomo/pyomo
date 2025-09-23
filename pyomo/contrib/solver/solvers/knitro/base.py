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
from collections.abc import Sequence
from datetime import datetime, timezone
from io import StringIO
from typing import Optional, Union

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import ApplicationError
from pyomo.common.numeric_types import value
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common import base
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoOptimalSolutionError,
)
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager

from .api import knitro
from .config import Config
from .engine import Engine
from .package import PackageChecker
from .solution import SolutionLoader, SolutionProvider
from .typing import UnreachableError, ValueType
from .utils import Problem


class SolverBase(SolutionProvider, PackageChecker, base.SolverBase):
    CONFIG = Config()
    config: Config

    _engine: Engine
    _problem: Problem
    _stream: StringIO
    _saved_var_values: dict[int, Optional[float]]

    def __init__(self, **kwds) -> None:
        PackageChecker.__init__(self)
        base.SolverBase.__init__(self, **kwds)
        self._engine = Engine()
        self._problem = Problem()
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

        with capture_output(TeeStream(self._stream, *config.tee), capture_fd=False):
            self._solve(config, timer)

        if config.restore_variable_values_after_solve:
            self._restore_var_values()

        results = self._postsolve(config, timer)

        tock = datetime.now(timezone.utc)

        results.timing_info.start_timestamp = tick
        results.timing_info.wall_time = (tock - tick).total_seconds()
        return results

    def _build_config(self, **kwds) -> Config:
        return self.config(value=kwds, preserve_implicit=True)

    def _validate_problem(self) -> None:
        if len(self._problem.objs) > 1:
            msg = f"{self.name} does not support multiple objectives."
            raise IncompatibleModelError(msg)

    def _check_available(self) -> None:
        avail = self.available()
        if not avail:
            msg = f"Solver {self.name} is not available: {avail}."
            raise ApplicationError(msg)

    def _save_var_values(self) -> None:
        self._saved_var_values.clear()
        for var in self.get_vars():
            self._saved_var_values[id(var)] = value(var.value)

    def _restore_var_values(self) -> None:
        StaleFlagManager.mark_all_as_stale(delayed=True)
        for var in self.get_vars():
            var.set_value(self._saved_var_values[id(var)])
        StaleFlagManager.mark_all_as_stale()

    @abstractmethod
    def _presolve(
        self, model: BlockData, config: Config, timer: HierarchicalTimer
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _solve(self, config: Config, timer: HierarchicalTimer) -> None:
        raise NotImplementedError

    def _postsolve(self, config: Config, timer: HierarchicalTimer) -> Results:
        status = self._engine.get_status()
        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_log = self._stream.getvalue()
        results.solver_config = config
        results.solution_status = self._get_solution_status(status)
        results.termination_condition = self._get_termination_condition(status)
        results.incumbent_objective = self._engine.get_obj_value()
        results.iteration_count = self._engine.get_num_iters()
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

    def get_vars(self) -> list[VarData]:
        return self._problem.variables

    def get_items(self, item_type: type):
        if item_type is VarData:
            return self._problem.variables
        elif item_type is ConstraintData:
            return self._problem.cons
        raise UnreachableError()

    def get_values(
        self,
        item_type: type,
        value_type: ValueType,
        items: Optional[Union[Sequence[VarData], Sequence[ConstraintData]]] = None,
    ):
        if items is None:
            items = self.get_items(item_type)
        x = self._engine.get_values(item_type, value_type, items)
        if x is None:
            error_type = SolutionProvider.get_error_type(item_type, value_type)
            raise error_type()
        sign = value_type.sign
        return ComponentMap([(k, sign * xk) for k, xk in zip(items, x)])

    def get_num_solutions(self) -> int:
        return self._engine.get_num_solutions()

    @staticmethod
    def _get_solution_status(status: int) -> SolutionStatus:
        if (
            status == knitro.KN_RC_OPTIMAL
            or status == knitro.KN_RC_OPTIMAL_OR_SATISFACTORY
            or status == knitro.KN_RC_NEAR_OPT
        ):
            return SolutionStatus.optimal
        elif status == knitro.KN_RC_FEAS_NO_IMPROVE:
            return SolutionStatus.feasible
        elif (
            status == knitro.KN_RC_INFEASIBLE
            or status == knitro.KN_RC_INFEAS_CON_BOUNDS
            or status == knitro.KN_RC_INFEAS_VAR_BOUNDS
            or status == knitro.KN_RC_INFEAS_NO_IMPROVE
        ):
            return SolutionStatus.infeasible
        else:
            return SolutionStatus.noSolution

    @staticmethod
    def _get_termination_condition(status: int) -> TerminationCondition:
        if (
            status == knitro.KN_RC_OPTIMAL
            or status == knitro.KN_RC_OPTIMAL_OR_SATISFACTORY
            or status == knitro.KN_RC_NEAR_OPT
        ):
            return TerminationCondition.convergenceCriteriaSatisfied
        elif status == knitro.KN_RC_INFEAS_NO_IMPROVE:
            return TerminationCondition.locallyInfeasible
        elif (
            status == knitro.KN_RC_INFEASIBLE
            or status == knitro.KN_RC_INFEAS_CON_BOUNDS
            or status == knitro.KN_RC_INFEAS_VAR_BOUNDS
        ):
            return TerminationCondition.provenInfeasible
        elif (
            status == knitro.KN_RC_UNBOUNDED_OR_INFEAS
            or status == knitro.KN_RC_UNBOUNDED
        ):
            return TerminationCondition.infeasibleOrUnbounded
        elif (
            status == knitro.KN_RC_ITER_LIMIT_FEAS
            or status == knitro.KN_RC_ITER_LIMIT_INFEAS
        ):
            return TerminationCondition.iterationLimit
        elif (
            status == knitro.KN_RC_TIME_LIMIT_FEAS
            or status == knitro.KN_RC_TIME_LIMIT_INFEAS
        ):
            return TerminationCondition.maxTimeLimit
        elif status == knitro.KN_RC_USER_TERMINATION:
            return TerminationCondition.interrupted
        else:
            return TerminationCondition.unknown
