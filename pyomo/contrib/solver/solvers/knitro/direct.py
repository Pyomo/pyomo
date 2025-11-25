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


from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.solvers.knitro.base import KnitroSolverBase
from pyomo.contrib.solver.solvers.knitro.config import KnitroConfig
from pyomo.core.base.block import BlockData


class KnitroDirectSolver(KnitroSolverBase):
    def _presolve(
        self, model: BlockData, config: KnitroConfig, timer: HierarchicalTimer
    ) -> None:
        timer.start("build_problem")
        self._model_data.set_block(model)
        timer.stop("build_problem")

    def _solve(self, config: KnitroConfig, timer: HierarchicalTimer) -> None:
        self._engine.renew()

        timer.start("add_vars")
        self._engine.add_vars(self._model_data.variables)
        timer.stop("add_vars")

        timer.start("add_cons")
        self._engine.add_cons(self._model_data.cons)
        timer.stop("add_cons")

        if self._model_data.objs:
            timer.start("set_objective")
            self._engine.set_obj(self._model_data.objs[0])
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
        self._engine.solve()
        timer.stop("solve")
