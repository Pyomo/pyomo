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
from pyomo.contrib.solver.common.base import PersistentSolverBase
from pyomo.contrib.solver.solvers.knitro.base import KnitroSolverBase
from pyomo.contrib.solver.solvers.knitro.config import KnitroConfig
from pyomo.contrib.solver.solvers.knitro.utils import KnitroModelData
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.var import VarData


class KnitroPersistentSolver(KnitroSolverBase, PersistentSolverBase):

    _model: BlockData | None
    _staged_model_data: KnitroModelData

    def __init__(self, **kwds) -> None:
        PersistentSolverBase.__init__(self, **kwds)
        KnitroSolverBase.__init__(self, **kwds)
        self._model = None
        self._staged_model_data = KnitroModelData()

    def _presolve(
        self, model: BlockData, config: KnitroConfig, timer: HierarchicalTimer
    ) -> None:
        if self._model is not model:
            self.set_instance(model)
            self._staged_model_data.clear()

        if self._staged_model_data:
            self._update()

    def _solve(self, config: KnitroConfig, timer: HierarchicalTimer) -> None:
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

    def set_instance(self, model: BlockData):
        if self._model is model:
            return
        self._model = model
        self._model_data.set_block(model)
        self._engine.renew()
        self._engine.add_vars(self._model_data.variables)
        self._engine.add_cons(self._model_data.cons)
        if self._model_data.objs:
            self._engine.set_obj(self._model_data.objs[0])

    def add_block(self, block: BlockData):
        self._staged_model_data.add_block(block, clear_objs=True)

    def add_variables(self, variables: list[VarData]):
        self._staged_model_data.add_vars(variables)

    def add_constraints(self, cons: list[ConstraintData]):
        self._staged_model_data.add_cons(cons, existing_vars=self._model_data.variables)

    def set_objective(self, obj: ObjectiveData):
        self._staged_model_data.objs.clear()
        self._staged_model_data.objs.append(obj)

    def _update(self):
        self._model_data.add_vars(self._staged_model_data.variables)
        self._model_data.add_cons(self._staged_model_data.cons)
        
        self._engine.add_vars(self._staged_model_data.variables)
        self._engine.add_cons(self._staged_model_data.cons)

        if self._staged_model_data.objs:
            self._model_data.objs.clear()
            self._model_data.objs.extend(self._staged_model_data.objs)
            self._engine.set_obj(self._model_data.objs[0])
        
        self._staged_model_data.clear()

    def remove_variables(self, variables: list[VarData]) -> None:
        raise NotImplementedError(
            "KnitroPersistentSolver does not support removing variables."
        )

    def remove_constraints(self, cons: list[ConstraintData]) -> None:
        raise NotImplementedError(
            "KnitroPersistentSolver does not support removing constraints."
        )

    def update_variables(self, variables: list[VarData]) -> None:
        raise NotImplementedError(
            "KnitroPersistentSolver does not support updating variables."
        )

    def update_parameters(self) -> None:
        raise NotImplementedError(
            "KnitroPersistentSolver does not support updating parameters."
        )

    def add_parameters(self, params: list[ParamData]) -> None:
        raise NotImplementedError(
            "KnitroPersistentSolver does not support adding parameters."
        )

    def remove_parameters(self, params: list[ParamData]) -> None:
        raise NotImplementedError(
            "KnitroPersistentSolver does not support removing parameters."
        )

    def remove_block(self, block: BlockData) -> None:
        raise NotImplementedError(
            "KnitroPersistentSolver does not support removing blocks."
        )
