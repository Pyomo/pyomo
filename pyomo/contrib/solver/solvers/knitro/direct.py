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


import io
from collections.abc import Mapping, Sequence
from typing import Optional

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import ApplicationError
from pyomo.common.flags import NOTSET
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.base import Availability, SolverBase
from pyomo.contrib.solver.common.results import Results, TerminationCondition
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoOptimalSolutionError,
    NoSolutionError,
)
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager

from .api import knitro
from .mixin import License, SolverMixin
from .config import KnitroConfig
from .engine import Engine
from .utils import ProblemData


class _SolutionLoader(SolutionLoaderBase):
    def __init__(self, engine: Engine, problem: ProblemData):
        super().__init__()
        self._engine = engine
        self._problem = problem

    def get_number_of_solutions(self) -> int:
        return self._engine.get_num_solutions()

    def get_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if vars_to_load is None:
            vars_to_load = self._problem.variables

        x = self._engine.get_primals(vars_to_load)
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
            cons_to_load = self._problem.cons

        y = self._engine.get_duals(cons_to_load)
        if y is None:
            return NoDualsError()
        return ComponentMap([(con, y[i]) for i, con in enumerate(cons_to_load)])


class Solver(SolverMixin, SolverBase):
    NAME = "KNITRO"
    CONFIG = KnitroConfig()
    config: KnitroConfig

    def __init__(self, **kwds):
        SolverMixin.__init__(self)
        SolverBase.__init__(self, **kwds)
        self._engine = Engine()

    def _build_config(self, **kwds) -> KnitroConfig:
        return self.config(value=kwds, preserve_implicit=True)

    def _validate_problem(self, problem: ProblemData):
        if len(problem.objs) > 1:
            raise IncompatibleModelError(
                f"{self.NAME} does not support multiple objectives."
            )

    def solve(self, model: BlockData, **kwds) -> Results:
        config = self._build_config(**kwds)
        timer = config.timer or HierarchicalTimer()

        avail = self.available()
        if not avail:
            raise ApplicationError(f"Solver {self.NAME} is not available: {avail}.")

        StaleFlagManager.mark_all_as_stale()
        timer.start("build_problem")
        problem = ProblemData(model)
        timer.stop("build_problem")

        self._validate_problem(problem)

        stream = io.StringIO()
        ostreams = [stream] + config.tee
        with capture_output(TeeStream(*ostreams), capture_fd=False):
            self._engine.renew()

            timer.start("add_vars")
            self._engine.add_vars(problem.variables)
            timer.stop("add_vars")

            timer.start("add_cons")
            self._engine.add_cons(problem.cons)
            timer.stop("add_cons")

            if problem.objs:
                timer.start("set_objective")
                self._engine.set_obj(problem.objs[0])
                timer.stop("set_objective")

            self._engine.set_outlev()
            if config.threads is not None:
                self._engine.set_num_threads(config.threads)
            if config.time_limit is not None:
                self._engine.set_time_limit(config.time_limit)

            timer.start("load_options")
            self._engine.set_options(**config.solver_options)
            timer.stop("load_options")

            timer.start("solve")
            status = self._engine.solve()
            timer.stop("solve")

        results = Results()
        results.solver_config = config
        results.solver_name = self.NAME
        results.solver_version = self.version()
        results.solver_log = stream.getvalue()
        results.iteration_count = self._engine.get_num_iters()
        results.solution_status = self._engine.get_solution_status(status)
        results.termination_condition = self._engine.get_termination_condition(status)
        results.solution_loader = _SolutionLoader(self._engine, problem)
        if (
            config.raise_exception_on_nonoptimal_result
            and results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
        ):
            raise NoOptimalSolutionError()
        results.timing_info.solve_time = self._engine.get_solve_time()
        results.incumbent_objective = self._engine.get_obj_value()
        if config.load_solutions:
            timer.start("load_solutions")
            results.solution_loader.load_vars()
            timer.stop("load_solutions")

        return results
