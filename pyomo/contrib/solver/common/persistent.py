#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  __________________________________________________________________________

import abc
import datetime
from typing import List

from pyomo.core.base.constraint import ConstraintData, Constraint
from pyomo.core.base.sos import SOSConstraintData, SOSConstraint
from pyomo.core.base.var import VarData
from pyomo.core.base.param import ParamData, Param
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.results import Results
from pyomo.contrib.solver.common.util import collect_vars_and_named_exprs, get_objective


class PersistentSolverUtils(abc.ABC):
    def __init__(self, treat_fixed_vars_as_params=True):
        """
        Parameters
        ----------
        treat_fixed_vars_as_params: bool
            This is an advanced option that should only be used in special circumstances.
            With the default setting of True, fixed variables will be treated like parameters.
            This means that z == x*y will be linear if x or y is fixed and the constraint
            can be written to an LP file. If the value of the fixed variable gets changed, we have
            to completely reprocess all constraints using that variable. If
            treat_fixed_vars_as_params is False, then constraints will be processed as if fixed
            variables are not fixed, and the solver will be told the variable is fixed. This means
            z == x*y could not be written to an LP file even if x and/or y is fixed. However,
            updating the values of fixed variables is much faster this way.
        """
        self._model = None
        self._active_constraints = {}  # maps constraint to (lower, body, upper)
        self._vars = {}  # maps var id to (var, lb, ub, fixed, domain, value)
        self._params = {}  # maps param id to param
        self._objective = None
        self._objective_expr = None
        self._objective_sense = None
        self._named_expressions = (
            {}
        )  # maps constraint to list of tuples (named_expr, named_expr.expr)
        self._external_functions = ComponentMap()
        self._obj_named_expressions = []
        self._referenced_variables = (
            {}
        )  # var_id: [dict[constraints, None], dict[sos constraints, None], None or objective]
        self._vars_referenced_by_con = {}
        self._vars_referenced_by_obj = []
        self._expr_types = None
        self._treat_fixed_vars_as_params = treat_fixed_vars_as_params
        self._active_config = self.config

    def set_instance(self, model):
        saved_config = self.config
        saved_active_config = self._active_config
        self.__init__()
        self.config = saved_config
        self._active_config = saved_active_config
        self._model = model
        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    @abc.abstractmethod
    def _add_variables(self, variables: List[VarData]):
        pass

    def add_variables(self, variables: List[VarData]):
        for v in variables:
            if id(v) in self._referenced_variables:
                raise ValueError(f'Variable {v.name} has already been added')
            self._referenced_variables[id(v)] = [{}, {}, None]
            self._vars[id(v)] = (
                v,
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
        self._add_variables(variables)

    @abc.abstractmethod
    def _add_parameters(self, params: List[ParamData]):
        pass

    def add_parameters(self, params: List[ParamData]):
        for p in params:
            self._params[id(p)] = p
        self._add_parameters(params)

    @abc.abstractmethod
    def _add_constraints(self, cons: List[ConstraintData]):
        pass

    def _check_for_new_vars(self, variables: List[VarData]):
        new_vars = {}
        for v in variables:
            v_id = id(v)
            if v_id not in self._referenced_variables:
                new_vars[v_id] = v
        self.add_variables(list(new_vars.values()))

    def _check_to_remove_vars(self, variables: List[VarData]):
        vars_to_remove = {}
        for v in variables:
            v_id = id(v)
            ref_cons, ref_sos, ref_obj = self._referenced_variables[v_id]
            if len(ref_cons) == 0 and len(ref_sos) == 0 and ref_obj is None:
                vars_to_remove[v_id] = v
        self.remove_variables(list(vars_to_remove.values()))

    def add_constraints(self, cons: List[ConstraintData]):
        all_fixed_vars = {}
        for con in cons:
            if con in self._named_expressions:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._active_constraints[con] = con.expr
            tmp = collect_vars_and_named_exprs(con.expr)
            named_exprs, variables, fixed_vars, external_functions = tmp
            self._check_for_new_vars(variables)
            self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[con] = external_functions
            self._vars_referenced_by_con[con] = variables
            for v in variables:
                self._referenced_variables[id(v)][0][con] = None
            if not self._treat_fixed_vars_as_params:
                for v in fixed_vars:
                    v.unfix()
                    all_fixed_vars[id(v)] = v
        self._add_constraints(cons)
        for v in all_fixed_vars.values():
            v.fix()

    @abc.abstractmethod
    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        pass

    def add_sos_constraints(self, cons: List[SOSConstraintData]):
        for con in cons:
            if con in self._vars_referenced_by_con:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._active_constraints[con] = tuple()
            variables = con.get_variables()
            self._check_for_new_vars(variables)
            self._named_expressions[con] = []
            self._vars_referenced_by_con[con] = variables
            for v in variables:
                self._referenced_variables[id(v)][1][con] = None
        self._add_sos_constraints(cons)

    @abc.abstractmethod
    def _set_objective(self, obj: ObjectiveData):
        pass

    def set_objective(self, obj: ObjectiveData):
        if self._objective is not None:
            for v in self._vars_referenced_by_obj:
                self._referenced_variables[id(v)][2] = None
            self._check_to_remove_vars(self._vars_referenced_by_obj)
            self._external_functions.pop(self._objective, None)
        if obj is not None:
            self._objective = obj
            self._objective_expr = obj.expr
            self._objective_sense = obj.sense
            tmp = collect_vars_and_named_exprs(obj.expr)
            named_exprs, variables, fixed_vars, external_functions = tmp
            self._check_for_new_vars(variables)
            self._obj_named_expressions = [(i, i.expr) for i in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[obj] = external_functions
            self._vars_referenced_by_obj = variables
            for v in variables:
                self._referenced_variables[id(v)][2] = obj
            if not self._treat_fixed_vars_as_params:
                for v in fixed_vars:
                    v.unfix()
            self._set_objective(obj)
            for v in fixed_vars:
                v.fix()
        else:
            self._vars_referenced_by_obj = []
            self._objective = None
            self._objective_expr = None
            self._objective_sense = None
            self._obj_named_expressions = []
            self._set_objective(obj)

    def add_block(self, block):
        param_dict = {}
        for p in block.component_objects(Param, descend_into=True):
            if p.mutable:
                for _p in p.values():
                    param_dict[id(_p)] = _p
        self.add_parameters(list(param_dict.values()))
        self.add_constraints(
            list(
                block.component_data_objects(Constraint, descend_into=True, active=True)
            )
        )
        self.add_sos_constraints(
            list(
                block.component_data_objects(
                    SOSConstraint, descend_into=True, active=True
                )
            )
        )
        obj = get_objective(block)
        if obj is not None:
            self.set_objective(obj)

    @abc.abstractmethod
    def _remove_constraints(self, cons: List[ConstraintData]):
        pass

    def remove_constraints(self, cons: List[ConstraintData]):
        self._remove_constraints(cons)
        for con in cons:
            if con not in self._named_expressions:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][0].pop(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            self._external_functions.pop(con, None)
            del self._vars_referenced_by_con[con]

    @abc.abstractmethod
    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        pass

    def remove_sos_constraints(self, cons: List[SOSConstraintData]):
        self._remove_sos_constraints(cons)
        for con in cons:
            if con not in self._vars_referenced_by_con:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][1].pop(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            del self._vars_referenced_by_con[con]

    @abc.abstractmethod
    def _remove_variables(self, variables: List[VarData]):
        pass

    def remove_variables(self, variables: List[VarData]):
        self._remove_variables(variables)
        for v in variables:
            v_id = id(v)
            if v_id not in self._referenced_variables:
                raise ValueError(
                    f'Cannot remove variable {v.name} - it has not been added'
                )
            cons_using, sos_using, obj_using = self._referenced_variables[v_id]
            if cons_using or sos_using or (obj_using is not None):
                raise ValueError(
                    f'Cannot remove variable {v.name} - it is still being used by constraints or the objective'
                )
            del self._referenced_variables[v_id]
            del self._vars[v_id]

    @abc.abstractmethod
    def _remove_parameters(self, params: List[ParamData]):
        pass

    def remove_parameters(self, params: List[ParamData]):
        self._remove_parameters(params)
        for p in params:
            del self._params[id(p)]

    def remove_block(self, block):
        self.remove_constraints(
            list(
                block.component_data_objects(
                    ctype=Constraint, descend_into=True, active=True
                )
            )
        )
        self.remove_sos_constraints(
            list(
                block.component_data_objects(
                    ctype=SOSConstraint, descend_into=True, active=True
                )
            )
        )
        self.remove_parameters(
            list(
                dict(
                    (id(p), p)
                    for p in block.component_data_objects(
                        ctype=Param, descend_into=True
                    )
                ).values()
            )
        )

    @abc.abstractmethod
    def _update_variables(self, variables: List[VarData]):
        pass

    def update_variables(self, variables: List[VarData]):
        for v in variables:
            self._vars[id(v)] = (
                v,
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
        self._update_variables(variables)

    @abc.abstractmethod
    def update_parameters(self):
        pass

    def update(self, timer: HierarchicalTimer = None):
        if timer is None:
            timer = HierarchicalTimer()
        config = self._active_config.auto_updates
        new_vars = []
        old_vars = []
        new_params = []
        old_params = []
        new_cons = []
        old_cons = []
        old_sos = []
        new_sos = []
        current_cons_dict = {}
        current_sos_dict = {}
        timer.start('vars')
        if config.update_vars:
            start_vars = {v_id: v_tuple[0] for v_id, v_tuple in self._vars.items()}
        timer.stop('vars')
        timer.start('params')
        if config.check_for_new_or_removed_params:
            current_params_dict = {}
            for p in self._model.component_objects(Param, descend_into=True):
                if p.mutable:
                    for _p in p.values():
                        current_params_dict[id(_p)] = _p
            for p_id, p in current_params_dict.items():
                if p_id not in self._params:
                    new_params.append(p)
            for p_id, p in self._params.items():
                if p_id not in current_params_dict:
                    old_params.append(p)
        timer.stop('params')
        timer.start('cons')
        if config.check_for_new_or_removed_constraints or config.update_constraints:
            current_cons_dict = {
                c: None
                for c in self._model.component_data_objects(
                    Constraint, descend_into=True, active=True
                )
            }
            current_sos_dict = {
                c: None
                for c in self._model.component_data_objects(
                    SOSConstraint, descend_into=True, active=True
                )
            }
            for c in current_cons_dict.keys():
                if c not in self._vars_referenced_by_con:
                    new_cons.append(c)
            for c in current_sos_dict.keys():
                if c not in self._vars_referenced_by_con:
                    new_sos.append(c)
            for c in self._vars_referenced_by_con:
                if c not in current_cons_dict and c not in current_sos_dict:
                    if (c.ctype is Constraint) or (
                        c.ctype is None and isinstance(c, ConstraintData)
                    ):
                        old_cons.append(c)
                    else:
                        assert (c.ctype is SOSConstraint) or (
                            c.ctype is None and isinstance(c, SOSConstraintData)
                        )
                        old_sos.append(c)
        self.remove_constraints(old_cons)
        self.remove_sos_constraints(old_sos)
        timer.stop('cons')
        timer.start('params')
        self.remove_parameters(old_params)

        # sticking this between removal and addition
        # is important so that we don't do unnecessary work
        if config.update_parameters:
            self.update_parameters()

        self.add_parameters(new_params)
        timer.stop('params')
        timer.start('vars')
        self.add_variables(new_vars)
        timer.stop('vars')
        timer.start('cons')
        self.add_constraints(new_cons)
        self.add_sos_constraints(new_sos)
        new_cons_set = set(new_cons)
        new_sos_set = set(new_sos)
        cons_to_remove_and_add = {}
        need_to_set_objective = False
        if config.update_constraints:
            for c in current_cons_dict.keys():
                if c not in new_cons_set and c.expr is not self._active_constraints[c]:
                    cons_to_remove_and_add[c] = None
            sos_to_update = []
            for c in current_sos_dict.keys():
                if c not in new_sos_set:
                    sos_to_update.append(c)
            self.remove_sos_constraints(sos_to_update)
            self.add_sos_constraints(sos_to_update)
        timer.stop('cons')
        timer.start('vars')
        if config.update_vars:
            end_vars = {v_id: v_tuple[0] for v_id, v_tuple in self._vars.items()}
            vars_to_check = [v for v_id, v in end_vars.items() if v_id in start_vars]
        if config.update_vars:
            vars_to_update = []
            for v in vars_to_check:
                _v, lb, ub, fixed, domain_interval, value = self._vars[id(v)]
                if (fixed != v.fixed) or (fixed and (value != v.value)):
                    vars_to_update.append(v)
                    if self._treat_fixed_vars_as_params:
                        for c in self._referenced_variables[id(v)][0]:
                            cons_to_remove_and_add[c] = None
                        if self._referenced_variables[id(v)][2] is not None:
                            need_to_set_objective = True
                elif lb is not v._lb:
                    vars_to_update.append(v)
                elif ub is not v._ub:
                    vars_to_update.append(v)
                elif domain_interval != v.domain.get_interval():
                    vars_to_update.append(v)
            self.update_variables(vars_to_update)
        timer.stop('vars')
        timer.start('cons')
        cons_to_remove_and_add = list(cons_to_remove_and_add.keys())
        self.remove_constraints(cons_to_remove_and_add)
        self.add_constraints(cons_to_remove_and_add)
        timer.stop('cons')
        timer.start('named expressions')
        if config.update_named_expressions:
            cons_to_update = []
            for c, expr_list in self._named_expressions.items():
                if c in new_cons_set:
                    continue
                for named_expr, old_expr in expr_list:
                    if named_expr.expr is not old_expr:
                        cons_to_update.append(c)
                        break
            self.remove_constraints(cons_to_update)
            self.add_constraints(cons_to_update)
            for named_expr, old_expr in self._obj_named_expressions:
                if named_expr.expr is not old_expr:
                    need_to_set_objective = True
                    break
        timer.stop('named expressions')
        timer.start('objective')
        if self._active_config.auto_updates.check_for_new_objective:
            pyomo_obj = get_objective(self._model)
            if pyomo_obj is not self._objective:
                need_to_set_objective = True
        else:
            pyomo_obj = self._objective
        if self._active_config.auto_updates.update_objective:
            if pyomo_obj is not None and pyomo_obj.expr is not self._objective_expr:
                need_to_set_objective = True
            elif pyomo_obj is not None and pyomo_obj.sense is not self._objective_sense:
                # we can definitely do something faster here than resetting the whole objective
                need_to_set_objective = True
        if need_to_set_objective:
            self.set_objective(pyomo_obj)
        timer.stop('objective')

        # this has to be done after the objective and constraints in case the
        # old objective/constraints use old variables
        timer.start('vars')
        self.remove_variables(old_vars)
        timer.stop('vars')


class PersistentSolverMixin:
    """
    The `solve` method in Gurobi and Highs is exactly the same, so this Mixin
    minimizes the duplicate code
    """

    def solve(self, model, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        self._active_config = config = self.config(value=kwds, preserve_implicit=True)
        StaleFlagManager.mark_all_as_stale()

        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        if model is not self._model:
            timer.start('set_instance')
            self.set_instance(model)
            timer.stop('set_instance')
        else:
            timer.start('update')
            self.update(timer=timer)
            timer.stop('update')

        res = self._solve()
        self._last_results_object = res

        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = (end_timestamp - start_timestamp).total_seconds()
        res.timing_info.timer = timer
        self._active_config = self.config

        return res
