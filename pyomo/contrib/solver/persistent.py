#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  __________________________________________________________________________

import abc
from typing import List

from pyomo.core.base.constraint import _GeneralConstraintData, Constraint
from pyomo.core.base.sos import _SOSConstraintData, SOSConstraint
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData, Param
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.contrib.solver.util import collect_vars_and_named_exprs, get_objective


class PersistentSolverUtils(abc.ABC):
    def __init__(self):
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

    def set_instance(self, model):
        saved_config = self.config
        self.__init__()
        self.config = saved_config
        self._model = model
        self.add_block(model)
        if self._objective is None:
            self.set_objective(None)

    @abc.abstractmethod
    def _add_variables(self, variables: List[_GeneralVarData]):
        pass

    def add_variables(self, variables: List[_GeneralVarData]):
        for v in variables:
            if id(v) in self._referenced_variables:
                raise ValueError(
                    'variable {name} has already been added'.format(name=v.name)
                )
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
    def _add_parameters(self, params: List[_ParamData]):
        pass

    def add_parameters(self, params: List[_ParamData]):
        for p in params:
            self._params[id(p)] = p
        self._add_parameters(params)

    @abc.abstractmethod
    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        pass

    def _check_for_new_vars(self, variables: List[_GeneralVarData]):
        new_vars = {}
        for v in variables:
            v_id = id(v)
            if v_id not in self._referenced_variables:
                new_vars[v_id] = v
        self.add_variables(list(new_vars.values()))

    def _check_to_remove_vars(self, variables: List[_GeneralVarData]):
        vars_to_remove = {}
        for v in variables:
            v_id = id(v)
            ref_cons, ref_sos, ref_obj = self._referenced_variables[v_id]
            if len(ref_cons) == 0 and len(ref_sos) == 0 and ref_obj is None:
                vars_to_remove[v_id] = v
        self.remove_variables(list(vars_to_remove.values()))

    def add_constraints(self, cons: List[_GeneralConstraintData]):
        all_fixed_vars = {}
        for con in cons:
            if con in self._named_expressions:
                raise ValueError(
                    'constraint {name} has already been added'.format(name=con.name)
                )
            self._active_constraints[con] = (con.lower, con.body, con.upper)
            tmp = collect_vars_and_named_exprs(con.body)
            named_exprs, variables, fixed_vars, external_functions = tmp
            self._check_for_new_vars(variables)
            self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[con] = external_functions
            self._vars_referenced_by_con[con] = variables
            for v in variables:
                self._referenced_variables[id(v)][0][con] = None
            if not self.config.auto_updates.treat_fixed_vars_as_params:
                for v in fixed_vars:
                    v.unfix()
                    all_fixed_vars[id(v)] = v
        self._add_constraints(cons)
        for v in all_fixed_vars.values():
            v.fix()

    @abc.abstractmethod
    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        pass

    def add_sos_constraints(self, cons: List[_SOSConstraintData]):
        for con in cons:
            if con in self._vars_referenced_by_con:
                raise ValueError(
                    'constraint {name} has already been added'.format(name=con.name)
                )
            self._active_constraints[con] = tuple()
            variables = con.get_variables()
            self._check_for_new_vars(variables)
            self._named_expressions[con] = []
            self._vars_referenced_by_con[con] = variables
            for v in variables:
                self._referenced_variables[id(v)][1][con] = None
        self._add_sos_constraints(cons)

    @abc.abstractmethod
    def _set_objective(self, obj: _GeneralObjectiveData):
        pass

    def set_objective(self, obj: _GeneralObjectiveData):
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
            if not self.config.auto_updates.treat_fixed_vars_as_params:
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
    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        pass

    def remove_constraints(self, cons: List[_GeneralConstraintData]):
        self._remove_constraints(cons)
        for con in cons:
            if con not in self._named_expressions:
                raise ValueError(
                    'cannot remove constraint {name} - it was not added'.format(
                        name=con.name
                    )
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][0].pop(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            self._external_functions.pop(con, None)
            del self._vars_referenced_by_con[con]

    @abc.abstractmethod
    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        pass

    def remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        self._remove_sos_constraints(cons)
        for con in cons:
            if con not in self._vars_referenced_by_con:
                raise ValueError(
                    'cannot remove constraint {name} - it was not added'.format(
                        name=con.name
                    )
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][1].pop(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            del self._vars_referenced_by_con[con]

    @abc.abstractmethod
    def _remove_variables(self, variables: List[_GeneralVarData]):
        pass

    def remove_variables(self, variables: List[_GeneralVarData]):
        self._remove_variables(variables)
        for v in variables:
            v_id = id(v)
            if v_id not in self._referenced_variables:
                raise ValueError(
                    'cannot remove variable {name} - it has not been added'.format(
                        name=v.name
                    )
                )
            cons_using, sos_using, obj_using = self._referenced_variables[v_id]
            if cons_using or sos_using or (obj_using is not None):
                raise ValueError(
                    'cannot remove variable {name} - it is still being used by constraints or the objective'.format(
                        name=v.name
                    )
                )
            del self._referenced_variables[v_id]
            del self._vars[v_id]

    @abc.abstractmethod
    def _remove_parameters(self, params: List[_ParamData]):
        pass

    def remove_parameters(self, params: List[_ParamData]):
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
    def _update_variables(self, variables: List[_GeneralVarData]):
        pass

    def update_variables(self, variables: List[_GeneralVarData]):
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
        config = self.config.auto_updates
        new_vars = []
        old_vars = []
        new_params = []
        old_params = []
        new_cons = []
        old_cons = []
        old_sos = []
        new_sos = []
        current_vars_dict = {}
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
            for c in self._vars_referenced_by_con.keys():
                if c not in current_cons_dict and c not in current_sos_dict:
                    if (c.ctype is Constraint) or (
                        c.ctype is None and isinstance(c, _GeneralConstraintData)
                    ):
                        old_cons.append(c)
                    else:
                        assert (c.ctype is SOSConstraint) or (
                            c.ctype is None and isinstance(c, _SOSConstraintData)
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
        new_vars_set = set(id(v) for v in new_vars)
        cons_to_remove_and_add = {}
        need_to_set_objective = False
        if config.update_constraints:
            cons_to_update = []
            sos_to_update = []
            for c in current_cons_dict.keys():
                if c not in new_cons_set:
                    cons_to_update.append(c)
            for c in current_sos_dict.keys():
                if c not in new_sos_set:
                    sos_to_update.append(c)
            for c in cons_to_update:
                lower, body, upper = self._active_constraints[c]
                new_lower, new_body, new_upper = c.lower, c.body, c.upper
                if new_body is not body:
                    cons_to_remove_and_add[c] = None
                    continue
                if new_lower is not lower:
                    if (
                        type(new_lower) is NumericConstant
                        and type(lower) is NumericConstant
                        and new_lower.value == lower.value
                    ):
                        pass
                    else:
                        cons_to_remove_and_add[c] = None
                        continue
                if new_upper is not upper:
                    if (
                        type(new_upper) is NumericConstant
                        and type(upper) is NumericConstant
                        and new_upper.value == upper.value
                    ):
                        pass
                    else:
                        cons_to_remove_and_add[c] = None
                        continue
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
                    if self.config.auto_updates.treat_fixed_vars_as_params:
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
        if self.config.auto_updates.check_for_new_objective:
            pyomo_obj = get_objective(self._model)
            if pyomo_obj is not self._objective:
                need_to_set_objective = True
        else:
            pyomo_obj = self._objective
        if self.config.auto_updates.update_objective:
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
