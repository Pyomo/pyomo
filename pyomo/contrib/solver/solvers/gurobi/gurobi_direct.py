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

import operator
from typing import List, Any

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
from pyomo.contrib.solver.common.solution_loader import (
    SolutionLoaderBase,
    load_import_suffixes,
)
from .gurobi_direct_base import GurobiDirectBase, gurobipy


class GurobiDirectSolutionLoader(SolutionLoaderBase):
    def __init__(self, grb_model, grb_cons, grb_vars, pyo_cons, pyo_vars, pyomo_model):
        self._grb_model = grb_model
        self._grb_cons = grb_cons
        self._grb_vars = grb_vars
        self._pyo_cons = pyo_cons
        self._pyo_vars = pyo_vars
        self._pyomo_model = pyomo_model
        GurobiDirectBase._register_env_client()

    def __del__(self):
        if python_is_shutting_down():
            return
        # Free the associated model
        if self._grb_model is not None:
            self._grb_cons = None
            self._grb_vars = None
            self._pyo_cons = None
            self._pyo_vars = None
            self._pyomo_model = None
            # explicitly release the model
            self._grb_model.dispose()
            self._grb_model = None
        # Release the gurobi license if this is the last reference to
        # the environment (either through a results object or solver
        # interface)
        GurobiDirectBase._release_env_client()

    def get_number_of_solutions(self) -> int:
        if self._grb_model.SolCount == 0:
            return 0
        return 1

    def get_solution_ids(self) -> List[Any]:
        return [0]

    def load_vars(self, vars_to_load=None, solution_id=None):
        assert solution_id == None
        if self._grb_model.SolCount == 0:
            raise NoSolutionError()

        iterator = zip(self._pyo_vars, self._grb_vars.x.tolist())
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_val: var_val[0] in vars_to_load, iterator)
        for p_var, g_var in iterator:
            p_var.set_value(g_var, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_vars(self, vars_to_load=None, solution_id=None):
        assert solution_id == None
        if self._grb_model.SolCount == 0:
            raise NoSolutionError()

        iterator = zip(self._pyo_vars, self._grb_vars.x.tolist())
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_val: var_val[0] in vars_to_load, iterator)
        return ComponentMap(iterator)

    def get_duals(self, cons_to_load=None, solution_id=None):
        assert solution_id == None
        if self._grb_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoDualsError()

        def dedup(_iter):
            last = None
            for con_info_dual in _iter:
                if not con_info_dual[1] and con_info_dual[0][0] is last:
                    continue
                last = con_info_dual[0][0]
                yield con_info_dual

        iterator = dedup(zip(self._pyo_cons, self._grb_cons.getAttr('Pi').tolist()))
        if cons_to_load:
            cons_to_load = set(cons_to_load)
            iterator = filter(
                lambda con_info_dual: con_info_dual[0][0] in cons_to_load, iterator
            )
        return {con_info[0]: dual for con_info, dual in iterator}

    def get_reduced_costs(self, vars_to_load=None, solution_id=None):
        assert solution_id == None
        if self._grb_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoReducedCostsError()

        iterator = zip(self._pyo_vars, self._grb_vars.getAttr('Rc').tolist())
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_rc: var_rc[0] in vars_to_load, iterator)
        return ComponentMap(iterator)

    def load_import_suffixes(self, solution_id=None):
        load_import_suffixes(
            pyomo_model=self._pyomo_model, solution_loader=self, solution_id=solution_id
        )


class GurobiDirect(GurobiDirectBase):
    _minimum_version = (9, 0, 0)

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._gurobi_vars = None
        self._pyomo_vars = None

    def _pyomo_gurobi_var_iter(self):
        return zip(self._pyomo_vars, self._gurobi_vars.tolist())

    def _create_solver_model(self, pyomo_model):
        timer = self.config.timer

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
        self._gurobi_vars = x

        solution_loader = GurobiDirectSolutionLoader(
            gurobi_model, A, x, repn.rows, repn.columns, pyomo_model
        )
        has_obj = len(repn.objectives) > 0

        return gurobi_model, solution_loader, has_obj
