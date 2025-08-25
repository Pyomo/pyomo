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
from collections.abc import Iterable
from datetime import datetime, timezone
from io import StringIO

from pyomo.common.errors import ApplicationError
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common import base
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.util import IncompatibleModelError
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager

from .api import knitro
from .config import Config
from .engine import Engine
from .package import PackageChecker
from .utils import Problem


class SolverBase(PackageChecker, base.SolverBase):
    CONFIG = Config()
    config: Config

    _engine: Engine
    _problem: Problem
    _stream: StringIO

    def __init__(self, **kwds):
        PackageChecker.__init__(self)
        base.SolverBase.__init__(self, **kwds)
        self._engine = Engine()
        self._problem = Problem()
        self._stream = StringIO()

    def solve(self, model: BlockData, **kwds) -> Results:
        tick = datetime.now(timezone.utc)
        self._check_available()

        config = self._build_config(**kwds)
        timer = config.timer or HierarchicalTimer()

        StaleFlagManager.mark_all_as_stale()

        self._presolve(model, config, timer)
        self._validate_problem()

        self._stream = StringIO()
        with capture_output(TeeStream(self._stream, *config.tee), capture_fd=False):
            self._solve(config, timer)

        results = self._postsolve(config, timer)

        tock = datetime.now(timezone.utc)

        results.timing_info.start_timestamp = tick
        results.timing_info.wall_time = (tock - tick).total_seconds()
        return results

    def _build_config(self, **kwds) -> Config:
        return self.config(value=kwds, preserve_implicit=True)

    def _validate_problem(self):
        if len(self._problem.objs) > 1:
            msg = f"{self.name} does not support multiple objectives."
            raise IncompatibleModelError(msg)

    def _check_available(self):
        avail = self.available()
        if not avail:
            msg = f"Solver {self.name} is not available: {avail}."
            raise ApplicationError(msg)

    @abstractmethod
    def _presolve(self, model: BlockData, config: Config, timer: HierarchicalTimer):
        raise NotImplementedError

    @abstractmethod
    def _solve(self, config: Config, timer: HierarchicalTimer) -> int:
        raise NotImplementedError

    def _postsolve(self, config: Config, timer: HierarchicalTimer) -> Results:
        status = self._engine.get_status()
        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_log = self._stream.getvalue()
        results.solver_config = config
        results.solution_status = self.get_solution_status(status)
        results.termination_condition = self.get_termination_condition(status)
        results.incumbent_objective = self._engine.get_obj_value()
        results.iteration_count = self._engine.get_num_iters()
        results.timing_info.solve_time = self._engine.get_solve_time()
        results.timing_info.timer = timer
        return results

    def get_vars(self):
        return self._problem.variables

    def get_objs(self):
        return self._problem.objs

    def get_cons(self):
        return self._problem.cons

    def get_primals(self, vars_to_load: Iterable[VarData]):
        return self._engine.get_primals(vars_to_load)

    def get_duals(self, cons_to_load: Iterable[ConstraintData]):
        return self._engine.get_duals(cons_to_load)

    def get_num_solutions(self):
        return self._engine.get_num_solutions()

    @staticmethod
    def get_solution_status(status: int) -> SolutionStatus:
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
    def get_termination_condition(status: int) -> TerminationCondition:
        if (
            status == knitro.KN_RC_OPTIMAL
            or status == knitro.KN_RC_OPTIMAL_OR_SATISFACTORY
            or status == knitro.KN_RC_NEAR_OPT
        ):
            return TerminationCondition.convergenceCriteriaSatisfied
        elif status == knitro.KN_RC_INFEAS_NO_IMPROVE:
            return TerminationCondition.locallyInfeasible
        elif status == knitro.KN_RC_INFEASIBLE:
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
