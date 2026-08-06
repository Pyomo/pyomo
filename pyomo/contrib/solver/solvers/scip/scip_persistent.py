# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from __future__ import annotations

from typing import List, Mapping, Optional

from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer

from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.base.param import ParamData
from pyomo.core.base.sos import SOSConstraintData
from pyomo.core.base.var import VarData

from pyomo.contrib.observer.model_observer import (
    Observer,
    ModelChangeDetector,
    AutoUpdateConfig,
    Reason,
)
from pyomo.contrib.solver.common.base import PersistentSolverBase
from pyomo.contrib.solver.common.results import Results

import pyomo.contrib.solver.solvers.scip.base as scip_base
import pyomo.contrib.solver.solvers.scip.scip_direct as scip_direct


class ScipPersistentSolutionLoader(scip_base.ScipSolutionLoader):
    def __init__(self, solver_model, var_map, con_map, pyomo_model, opt) -> None:
        super().__init__(solver_model, var_map, con_map, pyomo_model, opt)
        self._valid = True

    def invalidate(self):
        self._valid = False

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def load_vars(self, vars_to_load: List[VarData] | None = None) -> None:
        self._assert_solution_still_valid()
        return super().load_vars(vars_to_load)

    def get_vars(
        self, vars_to_load: List[VarData] | None = None
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return super().get_vars(vars_to_load)

    def get_number_of_solutions(self) -> int:
        self._assert_solution_still_valid()
        return super().get_number_of_solutions()

    def get_solution_ids(self) -> List:
        self._assert_solution_still_valid()
        return super().get_solution_ids()

    def load_import_suffixes(self):
        self._assert_solution_still_valid()
        super().load_import_suffixes()

    def _set_solution_id(self, solution_id: int) -> int:
        self._assert_solution_still_valid()
        return super()._set_solution_id(solution_id)


class ScipPersistentConfig(scip_base.ScipConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        scip_base.ScipConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.auto_updates: bool = self.declare('auto_updates', AutoUpdateConfig())


class ScipPersistent(scip_direct.ScipDirect, PersistentSolverBase, Observer):
    _minimum_version = (5, 5, 0)  # this is probably conservative
    CONFIG = ScipPersistentConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._pyomo_model = None
        self._change_detector = None
        self._last_results_object: Optional[Results] = None
        self._needs_reopt = False
        self._range_constraints = set()

    def _clear(self):
        super()._clear()
        self._pyomo_model = None
        self._change_detector = None
        self._needs_reopt = False
        self._range_constraints = set()

    def _check_reopt(self):
        if self._needs_reopt:
            # self._solver_model.freeReoptSolve()  # when is it safe to use this one???
            self._solver_model.freeTransform()
            self._needs_reopt = False

    def _create_solver_model(self, pyomo_model, config):
        if pyomo_model is self._pyomo_model:
            self.update(**config)
        else:
            self.set_instance(pyomo_model, **config)

        solution_loader = ScipPersistentSolutionLoader(
            solver_model=self._solver_model,
            var_map=self._pyomo_var_to_solver_var_map,
            con_map=self._pyomo_con_to_solver_con_map,
            pyomo_model=pyomo_model,
            opt=self,
        )

        has_obj = self._objective is not None
        return self._solver_model, solution_loader, has_obj

    def solve(self, model, **kwds) -> Results:
        res = super().solve(model, **kwds)
        self._last_results_object = res
        self._needs_reopt = True
        return res

    def update(self, **kwds):
        config = self.config(value=kwds, preserve_implicit=True)
        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        if self._pyomo_model is None:
            raise RuntimeError('must call set_instance or solve before update')
        timer.start('update')
        self._change_detector.update(timer=timer, **config.auto_updates)
        timer.stop('update')

    def set_instance(self, pyomo_model, **kwds):
        config = self.config(value=kwds, preserve_implicit=True)
        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        self._clear()
        self._pyomo_model = pyomo_model
        self._solver_model = scip_base.scip.Model()
        timer.start('set_instance')
        self._change_detector = ModelChangeDetector(
            model=self._pyomo_model, observers=[self], **config.auto_updates
        )
        timer.stop('set_instance')

    def _invalidate_last_results(self):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()

    def _update_variables(self, variables: Mapping[VarData, Reason]):
        new_vars = []
        old_vars = []
        mod_vars = []
        for v, reason in variables.items():
            if reason & Reason.added:
                new_vars.append(v)
            elif reason & Reason.removed:
                old_vars.append(v)
            else:
                mod_vars.append(v)

        if new_vars:
            self._add_variables(new_vars)
        if old_vars:
            self._remove_variables(old_vars)
        if mod_vars:
            self._update_vars_for_real(mod_vars)

    def _update_parameters(self, params: Mapping[ParamData, Reason]):
        new_params = []
        old_params = []
        mod_params = []
        for p, reason in params.items():
            if reason & Reason.added:
                new_params.append(p)
            elif reason & Reason.removed:
                old_params.append(p)
            else:
                mod_params.append(p)

        if new_params:
            self._add_parameters(new_params)
        if old_params:
            self._remove_parameters(old_params)
        if mod_params:
            self._update_params_for_real(mod_params)

    def _update_constraints(self, cons: Mapping[ConstraintData, Reason]):
        new_cons = []
        old_cons = []
        for c, reason in cons.items():
            if reason & Reason.added:
                new_cons.append(c)
            elif reason & Reason.removed:
                old_cons.append(c)
            elif reason & Reason.expr:
                old_cons.append(c)
                new_cons.append(c)

        if old_cons:
            self._remove_constraints(old_cons)
        if new_cons:
            self._add_constraints(new_cons)

    def _update_sos_constraints(self, cons: Mapping[SOSConstraintData, Reason]):
        new_cons = []
        old_cons = []
        for c, reason in cons.items():
            if reason & Reason.added:
                new_cons.append(c)
            elif reason & Reason.removed:
                old_cons.append(c)
            elif reason & Reason.sos_items:
                old_cons.append(c)
                new_cons.append(c)

        if old_cons:
            self._remove_sos_constraints(old_cons)
        if new_cons:
            self._add_sos_constraints(new_cons)

    def _update_objectives(self, objs: Mapping[ObjectiveData, Reason]):
        new_objs = []
        old_objs = []
        for obj, reason in objs.items():
            if reason & Reason.added:
                new_objs.append(obj)
            elif reason & Reason.removed:
                old_objs.append(obj)
            elif reason & (Reason.expr | Reason.sense):
                old_objs.append(obj)
                new_objs.append(obj)

        if old_objs:
            self._remove_objectives(old_objs)
        if new_objs:
            self._add_objectives(new_objs)

    def _add_variables(self, variables: List[VarData]):
        self._check_reopt()
        self._invalidate_last_results()
        for v in variables:
            self._add_var(v)

    def _add_parameters(self, params: List[ParamData]):
        self._check_reopt()
        self._invalidate_last_results()
        for p in params:
            self._add_param(p)

    def _add_constraints(self, cons: List[ConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        for con in cons:
            if type(con.expr) is scip_base.RangedExpression:
                self._range_constraints.add(con)
        super()._add_constraints(cons)

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        return super()._add_sos_constraints(cons)

    def _add_objectives(self, objs: List[ObjectiveData]):
        self._check_reopt()
        if len(objs) > 1:
            raise NotImplementedError(
                'the persistent interface to scip currently '
                f'only supports single-objective problems; got {len(objs)}: '
                f'{[str(i) for i in objs]}'
            )

        if len(objs) == 0:
            return

        obj = objs[0]

        if self._objective is not None:
            raise NotImplementedError(
                'the persistent interface to scip currently '
                'only supports single-objective problems; tried to add '
                f'an objective ({str(obj)}), but there is already an '
                f'active objective ({str(self._objective)})'
            )

        self._invalidate_last_results()
        self._set_objective(obj)

    def _remove_objectives(self, objs: List[ObjectiveData]):
        self._check_reopt()
        for obj in objs:
            if obj is not self._objective:
                raise RuntimeError(
                    'tried to remove an objective that has not been added: '
                    f'{str(obj)}'
                )
            else:
                self._invalidate_last_results()
                self._set_objective(None)

    def _remove_constraints(self, cons: List[ConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        for con in cons:
            scip_con = self._pyomo_con_to_solver_con_map.pop(con)
            self._solver_model.delCons(scip_con)
            self._range_constraints.discard(con)

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        self._check_reopt()
        self._invalidate_last_results()
        for con in cons:
            scip_con = self._pyomo_con_to_solver_con_map.pop(con)
            self._solver_model.delCons(scip_con)

    def _remove_variables(self, variables: List[VarData]):
        self._check_reopt()
        self._invalidate_last_results()
        for v in variables:
            scip_var = self._pyomo_var_to_solver_var_map.pop(v)
            self._solver_model.delVar(scip_var)

    def _remove_parameters(self, params: List[ParamData]):
        self._check_reopt()
        self._invalidate_last_results()
        for p in params:
            scip_var = self._pyomo_param_to_solver_param_map.pop(p)
            self._solver_model.delVar(scip_var)

    def _update_vars_for_real(self, variables: List[VarData]):
        self._check_reopt()
        self._invalidate_last_results()
        for v in variables:
            scip_var = self._pyomo_var_to_solver_var_map[v]
            vtype = self._scip_vtype_from_var(v)
            lb, ub = self._scip_lb_ub_from_var(v)
            self._solver_model.chgVarLb(scip_var, lb)
            self._solver_model.chgVarUb(scip_var, ub)
            self._solver_model.chgVarType(scip_var, vtype)

    def _update_params_for_real(self, params: List[ParamData]):
        self._check_reopt()
        self._invalidate_last_results()
        for p in params:
            scip_var = self._pyomo_param_to_solver_param_map[p]
            lb = ub = p.value
            self._solver_model.chgVarLb(scip_var, lb)
            self._solver_model.chgVarUb(scip_var, ub)
            impacted_vars = self._change_detector.get_variables_impacted_by_param(p)
            if impacted_vars:
                impacted_vars_mapping = ComponentMap(
                    (v, Reason.bounds) for v in impacted_vars
                )
                self._update_variables(impacted_vars_mapping)
            impacted_cons = self._change_detector.get_constraints_impacted_by_param(p)
            for con in impacted_cons:
                if con in self._range_constraints:
                    self._remove_constraints([con])
                    self._add_constraints([con])

    def add_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.add_constraints(cons)

    def add_sos_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.add_sos_constraints(cons)

    def set_objective(self, obj: ObjectiveData):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.add_objectives([obj])

    def remove_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.remove_constraints(cons)

    def remove_sos_constraints(self, cons):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.remove_sos_constraints(cons)

    def update_variables(self, variables):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.update_variables(variables)

    def update_parameters(self, params):
        if self._change_detector is None:
            raise RuntimeError('call set_instance first')
        self._change_detector.update_parameters(params)
