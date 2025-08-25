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


import abc
from collections.abc import Iterable
from io import StringIO

from pyomo.common.errors import ApplicationError
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common import base
from pyomo.contrib.solver.common.results import Results
from pyomo.contrib.solver.common.util import IncompatibleModelError
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.staleflag import StaleFlagManager

from .config import Config
from .engine import Engine
from .utils import Problem


class SolverBase(base.SolverBase):
    CONFIG = Config()
    config: Config

    _engine: Engine
    _problem: Problem

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._engine = Engine()
        self._problem = Problem()

    def solve(self, model: BlockData, **kwds) -> Results:
        self._check_available()

        config = self._build_config(**kwds)
        timer = config.timer or HierarchicalTimer()

        StaleFlagManager.mark_all_as_stale()
        self._presolve(model, config, timer)
        self._validate_problem()

        stream = StringIO()
        ostreams = [stream] + config.tee
        with capture_output(TeeStream(*ostreams), capture_fd=False):
            status = self._solve(config, timer)

        return self._postsolve(stream, config, timer, status)

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

    @abc.abstractmethod
    def _presolve(self, model: BlockData, config: Config, timer: HierarchicalTimer):
        raise NotImplementedError

    @abc.abstractmethod
    def _solve(self, config: Config, timer: HierarchicalTimer) -> int:
        raise NotImplementedError

    def _postsolve(
        self, stream: StringIO, config: Config, timer: HierarchicalTimer, status: int
    ) -> Results:
        results = Results()
        results.solver_name = self.name
        results.solver_version = self.version()
        results.solver_log = stream.getvalue()
        results.solver_config = config
        results.solution_status = self._engine.get_solution_status(status)
        results.termination_condition = self._engine.get_termination_condition(status)
        results.incumbent_objective = self._engine.get_obj_value()
        results.iteration_count = self._engine.get_num_iters()
        results.timing_info.timer = timer
        results.timing_info.solve_time = self._engine.get_solve_time()
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
