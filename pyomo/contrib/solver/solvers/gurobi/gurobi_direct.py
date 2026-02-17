# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import operator

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

from pyomo.contrib.solver.common.util import (
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    IncompatibleModelError,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase
from .gurobi_direct_base import (
    GurobiDirectBase,
    gurobipy,
    GurobiDirectSolutionLoaderBase,
)
import logging

logger = logging.getLogger(__name__)


class GurobiDirectSolutionLoader(GurobiDirectSolutionLoaderBase):
    def __init__(self, solver_model, pyomo_vars, gurobi_vars, con_map) -> None:
        super().__init__(solver_model)
        self._pyomo_vars = pyomo_vars
        self._gurobi_vars = gurobi_vars
        self._con_map = con_map

    def _get_var_lists(self):
        return self._pyomo_vars, self._gurobi_vars

    def _get_var_map(self):
        return ComponentMap(zip(*self._get_var_lists()))

    def _get_con_map(self):
        return self._con_map

    def __del__(self):
        super().__del__()
        if python_is_shutting_down():
            return
        # Free the associated model
        if self._solver_model is not None:
            self._var_map = None
            self._con_map = None
            # explicitly release the model
            self._solver_model.dispose()
            self._solver_model = None


class GurobiDirect(GurobiDirectBase):
    _minimum_version = (9, 0, 0)

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._gurobi_vars = None
        self._pyomo_vars = None

    def _pyomo_gurobi_var_iter(self):
        return zip(self._pyomo_vars, self._gurobi_vars)

    def _create_solver_model(self, pyomo_model, config):
        timer = config.timer

        timer.start('compile_model')
        repn = LinearStandardFormCompiler().write(
            pyomo_model, mixed_form=True, set_sense=None
        )
        timer.stop('compile_model')

        if len(repn.objectives) > 1:
            raise IncompatibleModelError(
                f"The {self.__class__.__name__} solver only supports models "
                f"with zero or one objectives (received {len(repn.objectives)})."
            )

        timer.start('prepare_matrices')
        inf = float('inf')
        ninf = -inf
        bounds = list(map(operator.attrgetter('bounds'), repn.columns))
        lb = [ninf if _b is None else _b for _b in map(operator.itemgetter(0), bounds)]
        ub = [inf if _b is None else _b for _b in map(operator.itemgetter(1), bounds)]
        CON = gurobipy.GRB.CONTINUOUS
        BIN = gurobipy.GRB.BINARY
        INT = gurobipy.GRB.INTEGER
        vtype = [
            (
                CON
                if v.is_continuous()
                else BIN if v.is_binary() else INT if v.is_integer() else '?'
            )
            for v in repn.columns
        ]
        sense_type = list('=<>')  # Note: ordering matches 0, 1, -1
        sense = [sense_type[r[1]] for r in repn.rows]
        timer.stop('prepare_matrices')

        gurobi_model = gurobipy.Model(env=self.env())

        timer.start('transfer_model')
        x = gurobi_model.addMVar(
            len(repn.columns),
            lb=lb,
            ub=ub,
            obj=repn.c.todense()[0] if repn.c.shape[0] else 0,
            vtype=vtype,
        )
        A = gurobi_model.addMConstr(repn.A, x, sense, repn.rhs)
        if repn.c.shape[0]:
            gurobi_model.setAttr('ObjCon', repn.c_offset[0])
            gurobi_model.setAttr('ModelSense', int(repn.objectives[0].sense))
        # Note: calling gurobi_model.update() here is not
        # necessary (it will happen as part of optimize()):
        # gurobi_model.update()
        timer.stop('transfer_model')

        self._pyomo_vars = repn.columns
        timer.start('tolist')
        self._gurobi_vars = x.tolist()
        timer.stop('tolist')

        timer.start('create maps')
        timer.start('con map')
        con_map = {}
        for row, gc in zip(repn.rows, A.tolist()):
            pc = row.constraint
            if pc in con_map:
                # range constraint
                con_map[pc] = (con_map[pc], gc)
            else:
                con_map[pc] = gc
        timer.stop('con map')
        timer.stop('create maps')
        solution_loader = GurobiDirectSolutionLoader(
            solver_model=gurobi_model,
            pyomo_vars=self._pyomo_vars,
            gurobi_vars=self._gurobi_vars,
            con_map=con_map,
        )
        has_obj = len(repn.objectives) > 0

        return gurobi_model, solution_loader, has_obj
