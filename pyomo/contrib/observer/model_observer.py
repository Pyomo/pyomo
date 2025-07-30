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
from typing import List, Sequence, Optional

from pyomo.common.config import ConfigDict, ConfigValue
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
from pyomo.common.numeric_types import native_numeric_types


"""
The ModelChangeDetector is meant to be used to automatically identify changes
in a Pyomo model or block. Here is a list of changes that will be detected. 
Note that inactive components (e.g., constraints) are treated as "removed".
  - new constraints that have been added to the model
  - constraints that have been removed from the model
  - new variables that have been detected in new or modified constraints/objectives
  - old variables that are no longer used in any constraints/objectives
  - new parameters that have been detected in new or modified constraints/objectives
  - old parameters that are no longer used in any constraints/objectives
  - new objectives that have been added to the model
  - objectives that have been removed from the model
  - modified constraint expressions (relies on expressions being immutable)
  - modified objective expressions (relies on expressions being immutable)
  - modified objective sense
  - changes to variable bounds, domains, and "fixed" flags
  - changes to named expressions (relies on expressions being immutable)
  - changes to parameter values and fixed variable values
"""


class AutoUpdateConfig(ConfigDict):
    """
    Control which parts of the model are automatically checked and/or updated upon re-solve
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        if doc is None:
            doc = 'Configuration options to detect changes in model between solves'
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.check_for_new_or_removed_constraints: bool = self.declare(
            'check_for_new_or_removed_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old constraints will not be automatically detected on subsequent
                solves. Use False only when manually updating the solver with opt.add_constraints()
                and opt.remove_constraints() or when you are certain constraints are not being
                added to/removed from the model.""",
            ),
        )
        self.check_for_new_objective: bool = self.declare(
            'check_for_new_objective',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old objectives will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.set_objective() or 
                when you are certain objectives are not being added to / removed from the model.""",
            ),
        )
        self.update_constraints: bool = self.declare(
            'update_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to existing constraints will not be automatically detected on 
                subsequent solves. This includes changes to the lower, body, and upper attributes of 
                constraints. Use False only when manually updating the solver with 
                opt.remove_constraints() and opt.add_constraints() or when you are certain constraints 
                are not being modified.""",
            ),
        )
        self.update_vars: bool = self.declare(
            'update_vars',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to existing variables will not be automatically detected on 
                subsequent solves. This includes changes to the lb, ub, domain, and fixed 
                attributes of variables. Use False only when manually updating the observer with 
                opt.update_variables() or when you are certain variables are not being modified.
                Note that changes to values of fixed variables is handled by 
                update_parameters_and_fixed_vars.""",
            ),
        )
        self.update_parameters_and_fixed_vars: bool = self.declare(
            'update_parameters',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to parameter values and fixed variable values will 
                not be automatically detected on subsequent solves. Use False only 
                when manually updating the observer with 
                opt.update_parameters_and_fixed_variables() or when you are certain 
                parameters are not being modified.""",
            ),
        )
        self.update_named_expressions: bool = self.declare(
            'update_named_expressions',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to Expressions will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver with 
                opt.remove_constraints() and opt.add_constraints() or when you are certain 
                Expressions are not being modified.""",
            ),
        )
        self.update_objective: bool = self.declare(
            'update_objective',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to objectives will not be automatically detected on 
                subsequent solves. This includes the expr and sense attributes of objectives. Use 
                False only when manually updating the solver with opt.set_objective() or when you are 
                certain objectives are not being modified.""",
            ),
        )


class Observer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def add_variables(self, variables: List[VarData]):
        pass

    @abc.abstractmethod
    def add_parameters(self, params: List[ParamData]):
        pass

    @abc.abstractmethod
    def add_constraints(self, cons: List[ConstraintData]):
        pass

    @abc.abstractmethod
    def add_sos_constraints(self, cons: List[SOSConstraintData]):
        pass

    @abc.abstractmethod
    def set_objective(self, obj: ObjectiveData):
        pass

    @abc.abstractmethod
    def remove_constraints(self, cons: List[ConstraintData]):
        pass

    @abc.abstractmethod
    def remove_sos_constraints(self, cons: List[SOSConstraintData]):
        pass

    @abc.abstractmethod
    def remove_variables(self, variables: List[VarData]):
        pass

    @abc.abstractmethod
    def remove_parameters(self, params: List[ParamData]):
        pass

    @abc.abstractmethod
    def update_variables(self, variables: List[VarData]):
        pass

    @abc.abstractmethod
    def update_parameters_and_fixed_variables(
        self,
        params: List[ParamData],
        variables: List[VarData],
    ):
        pass


class ModelChangeDetector:
    def __init__(
        self, observers: Sequence[Observer], 
        treat_fixed_vars_as_params=True, 
        **kwds,
    ):
        """
        Parameters
        ----------
        observers: Sequence[Observer]
            The objects to notify when changes are made to the model
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
        self._observers: List[Observer] = list(observers)
        self._model = None
        self._active_constraints = {}  # maps constraint to expression
        self._active_sos = {}
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
        self._referenced_params = (
            {}
        ) # param_id: [dict[constraints, None], dict[sos constraints, None], None or objective]
        self._vars_referenced_by_con = {}
        self._vars_referenced_by_obj = []
        self._params_referenced_by_con = {}
        self._params_referenced_by_obj = []
        self._expr_types = None
        self._treat_fixed_vars_as_params = treat_fixed_vars_as_params
        self.config: AutoUpdateConfig = AutoUpdateConfig()(value=kwds, preserve_implicit=True)

    def set_instance(self, model):
        saved_config = self.config
        self.__init__(observers=self._observers, treat_fixed_vars_as_params=self._treat_fixed_vars_as_params)
        self.config = saved_config
        self._model = model
        self._add_block(model)

    def _add_variables(self, variables: List[VarData]):
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
        for obs in self._observers:
            obs.add_variables(variables)

    def _add_parameters(self, params: List[ParamData]):
        for p in params:
            pid = id(p)
            if pid in self._referenced_params:
                raise ValueError(f'Parameter {p.name} has already been added')
            self._referenced_params[pid] = [{}, {}, None]
            self._params[id(p)] = (p, p.value)
        for obs in self._observers:
            obs.add_parameters(params)

    def _check_for_new_vars(self, variables: List[VarData]):
        new_vars = {}
        for v in variables:
            v_id = id(v)
            if v_id not in self._referenced_variables:
                new_vars[v_id] = v
        self._add_variables(list(new_vars.values()))

    def _check_to_remove_vars(self, variables: List[VarData]):
        vars_to_remove = {}
        for v in variables:
            v_id = id(v)
            ref_cons, ref_sos, ref_obj = self._referenced_variables[v_id]
            if len(ref_cons) == 0 and len(ref_sos) == 0 and ref_obj is None:
                vars_to_remove[v_id] = v
        self._remove_variables(list(vars_to_remove.values()))

    def _check_for_new_params(self, params: List[ParamData]):
        new_params = {}
        for p in params:
            pid = id(p)
            if pid not in self._referenced_params:
                new_params[pid] = p
        self._add_parameters(list(new_params.values()))

    def _check_to_remove_params(self, params: List[ParamData]):
        params_to_remove = {}
        for p in params:
            p_id = id(p)
            ref_cons, ref_sos, ref_obj = self._referenced_params[p_id]
            if len(ref_cons) == 0 and len(ref_sos) == 0 and ref_obj is None:
                params_to_remove[p_id] = p
        self._remove_parameters(list(params_to_remove.values()))

    def _add_constraints(self, cons: List[ConstraintData]):
        all_fixed_vars = {}
        for con in cons:
            if con in self._active_constraints:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._active_constraints[con] = con.expr
            tmp = collect_vars_and_named_exprs(con.expr)
            named_exprs, variables, fixed_vars, parameters, external_functions = tmp
            self._check_for_new_vars(variables)
            self._check_for_new_params(parameters)
            self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[con] = external_functions
            self._vars_referenced_by_con[con] = variables
            self._params_referenced_by_con[con] = parameters
            for v in variables:
                self._referenced_variables[id(v)][0][con] = None
            for p in parameters:
                self._referenced_params[id(p)][0][con] = None
            if not self._treat_fixed_vars_as_params:
                for v in fixed_vars:
                    v.unfix()
                    all_fixed_vars[id(v)] = v
        for obs in self._observers:
            obs.add_constraints(cons)
        for v in all_fixed_vars.values():
            v.fix()

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        all_fixed_vars = {}
        for con in cons:
            if con in self._vars_referenced_by_con:
                raise ValueError(f'Constraint {con.name} has already been added')
            sos_items = list(con.get_items())
            self._active_sos[con] = ([i[0] for i in sos_items], [i[1] for i in sos_items])
            variables = []
            params = []
            for v, p in sos_items:
                variables.append(v)
                if type(p) in native_numeric_types:
                    continue
                if p.is_parameter_type():
                    params.append(p)
            self._check_for_new_vars(variables)
            self._check_for_new_params(params)
            self._named_expressions[con] = []
            self._vars_referenced_by_con[con] = variables
            self._params_referenced_by_con[con] = params
            for v in variables:
                self._referenced_variables[id(v)][1][con] = None
            for p in params:
                self._referenced_params[id(p)][1][con] = None
            if not self._treat_fixed_vars_as_params:
                for v in variables:
                    if v.is_fixed():
                        v.unfix()
                        all_fixed_vars[id(v)] = v
        for obs in self._observers:
            obs.add_sos_constraints(cons)
        for v in all_fixed_vars.values():
            v.fix()

    def _set_objective(self, obj: ObjectiveData):
        if self._objective is not None:
            for v in self._vars_referenced_by_obj:
                self._referenced_variables[id(v)][2] = None
            self._check_to_remove_vars(self._vars_referenced_by_obj)
            self._check_to_remove_params(self._params_referenced_by_obj)
            self._external_functions.pop(self._objective, None)
        if obj is not None:
            self._objective = obj
            self._objective_expr = obj.expr
            self._objective_sense = obj.sense
            tmp = collect_vars_and_named_exprs(obj.expr)
            named_exprs, variables, fixed_vars, parameters, external_functions = tmp
            self._check_for_new_vars(variables)
            self._check_for_new_params(parameters)
            self._obj_named_expressions = [(i, i.expr) for i in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[obj] = external_functions
            self._vars_referenced_by_obj = variables
            self._params_referenced_by_obj = parameters
            for v in variables:
                self._referenced_variables[id(v)][2] = obj
            for p in parameters:
                self._referenced_params[id(p)][2] = obj
            if not self._treat_fixed_vars_as_params:
                for v in fixed_vars:
                    v.unfix()
            for obs in self._observers:
                obs.set_objective(obj)
            for v in fixed_vars:
                v.fix()
        else:
            self._vars_referenced_by_obj = []
            self._params_referenced_by_obj = []
            self._objective = None
            self._objective_expr = None
            self._objective_sense = None
            self._obj_named_expressions = []
            for obs in self._observers:
                obs.set_objective(obj)

    def _add_block(self, block):
        self._add_constraints(
            list(
                block.component_data_objects(Constraint, descend_into=True, active=True)
            )
        )
        self._add_sos_constraints(
            list(
                block.component_data_objects(
                    SOSConstraint, descend_into=True, active=True
                )
            )
        )
        obj = get_objective(block)
        if obj is not None:
            self._set_objective(obj)

    def _remove_constraints(self, cons: List[ConstraintData]):
        for obs in self._observers:
            obs.remove_constraints(cons)
        for con in cons:
            if con not in self._named_expressions:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][0].pop(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[id(p)][0].pop(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            self._check_to_remove_params(self._params_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            self._external_functions.pop(con, None)
            del self._vars_referenced_by_con[con]
            del self._params_referenced_by_con[con]

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        for obs in self._observers:
            obs.remove_sos_constraints(cons)
        for con in cons:
            if con not in self._vars_referenced_by_con:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][1].pop(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[id(p)][1].pop(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            self._check_to_remove_params(self._params_referenced_by_con[con])
            del self._active_sos[con]
            del self._named_expressions[con]
            del self._vars_referenced_by_con[con]
            del self._params_referenced_by_con[con]

    def _remove_variables(self, variables: List[VarData]):
        for obs in self._observers:
            obs.remove_variables(variables)
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

    def _remove_parameters(self, params: List[ParamData]):
        for obs in self._observers:
            obs.remove_parameters(params)
        for p in params:
            p_id = id(p)
            if p_id not in self._referenced_params:
                raise ValueError(
                    f'Cannot remove parameter {p.name} - it has not been added'
                )
            cons_using, sos_using, obj_using = self._referenced_params[p_id]
            if cons_using or sos_using or (obj_using is not None):
                raise ValueError(
                    f'Cannot remove parameter {p.name} - it is still being used by constraints or the objective'
                )
            del self._referenced_params[p_id]
            del self._params[p_id]

    def _update_variables(self, variables: List[VarData]):
        for v in variables:
            self._vars[id(v)] = (
                v,
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
        for obs in self._observers:
            obs.update_variables(variables)

    def _update_parameters_and_fixed_variables(self, params, variables):
        for p in params:
            self._params[id(p)] = (p, p.value)
        for v in variables:
            self._vars[id(v)][5] = v.value
        for obs in self._observers:
            obs.update_parameters_and_fixed_variables(params, variables)

    def _check_for_new_or_removed_sos(self):
        new_sos = []
        old_sos = []
        current_sos_dict = {
            c: None
            for c in self._model.component_data_objects(
                SOSConstraint, descend_into=True, active=True
            )
        }
        for c in current_sos_dict.keys():
            if c not in self._active_sos:
                new_sos.append(c)
        for c in self._active_sos:
            if c not in current_sos_dict:
                old_sos.append(c)
        return new_sos, old_sos

    def _check_for_new_or_removed_constraints(self):
        new_cons = []
        old_cons = []
        current_cons_dict = {
            c: None
            for c in self._model.component_data_objects(
                Constraint, descend_into=True, active=True
            )
        }
        for c in current_cons_dict.keys():
            if c not in self._active_constraints:
                new_cons.append(c)
        for c in self._active_constraints:
            if c not in current_cons_dict:
                old_cons.append(c)
        return new_cons, old_cons

    def _check_for_modified_sos(self):
        sos_to_update = []
        for c, (old_vlist, old_plist) in self._active_sos.items():
            sos_items = list(c.get_items())
            new_vlist = [i[0] for i in sos_items]
            new_plist = [i[1] for i in sos_items]
            if len(old_vlist) != len(new_vlist):
                sos_to_update.append(c)
            elif len(old_plist) != len(new_plist):
                sos_to_update.append(c)
            else:
                needs_update = False
                for v1, v2 in zip(old_vlist, new_vlist):
                    if v1 is not v2:
                        needs_update = True
                        break
                for p1, p2 in zip(old_plist, new_plist):
                    if p1 is not p2:
                        needs_update = True
                        if needs_update:
                            break
                if needs_update:
                    sos_to_update.append(c)
        return sos_to_update

    def _check_for_modified_constraints(self):
        cons_to_update = []
        for c, expr in self._active_constraints.items():
            if c.expr is not expr:
                cons_to_update.append(c)
        return cons_to_update

    def _check_for_var_changes(self):
        vars_to_update = []
        cons_to_update = {}
        update_obj = False
        for vid, (v, _lb, _ub, _fixed, _domain_interval, _value) in self._vars.items():
            if v.fixed != _fixed:
                vars_to_update.append(v)
                if self._treat_fixed_vars_as_params:
                    for c in self._referenced_variables[vid][0]:
                        cons_to_update[c] = None

            elif v._lb is not _lb:
                vars_to_update.append(v)
            elif v._ub is not _ub:
                vars_to_update.append(v)
            

    def update(self, timer: Optional[HierarchicalTimer] = None, **kwds):
        if timer is None:
            timer = HierarchicalTimer()
        config: AutoUpdateConfig = self.config(value=kwds, preserve_implicit=True)

        added_cons = set()
        added_sos = set()

        if config.check_for_new_or_removed_constraints:
            timer.start('sos')
            new_sos, old_sos = self._check_for_new_or_removed_sos()
            self._add_sos_constraints(new_sos)
            self._remove_sos_constraints(old_sos)
            added_sos.update(new_sos)
            timer.stop('cons')
            timer.start('cons')
            new_cons, old_cons = self._check_for_new_or_removed_constraints()
            self._add_constraints(new_cons)
            self._remove_constraints(old_cons)
            added_cons.update(new_cons)
            timer.stop('cons')

        if config.update_constraints:
            timer.start('cons')
            cons_to_update = self._check_for_modified_constraints()
            self._remove_constraints(cons_to_update)
            self._add_constraints(cons_to_update)
            added_cons.update(cons_to_update)
            timer.stop('cons')
            timer.start('sos')
            sos_to_update = self._check_for_modified_sos()
            self._remove_sos_constraints(sos_to_update)
            self._add_sos_constraints(sos_to_update)
            added_sos.update(sos_to_update)
            timer.stop('sos')

        need_to_set_objective = False

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

        if config.update_parameters:
            timer.start('params')
            modified_params = []
            for pid, (p, old_val) in self._params.items():
                if p.value != old_val:
                    modified_params.append(p)
            modified_vars = []
            for vid, (v, _lb, _ub, _fixed, _domain_interval, _val) in self._vars.items():
                if _fixed and _val != v.value:
                    modified_vars.append(v)
            self._update_parameters_and_fixed_variables(modified_params, modified_vars)
            timer.stop('params')


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
