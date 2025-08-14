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
from pyomo.contrib.solver.common.util import get_objective
from .component_collector import collect_components_from_expr
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
        self.update_parameters: bool = self.declare(
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
    def set_objective(self, obj: Optional[ObjectiveData]):
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
    def update_parameters(self, params: List[ParamData]):
        pass


class ModelChangeDetector:
    def __init__(self, observers: Sequence[Observer], **kwds):
        """
        Parameters
        ----------
        observers: Sequence[Observer]
            The objects to notify when changes are made to the model
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
        )  # param_id: [dict[constraints, None], dict[sos constraints, None], None or objective]
        self._vars_referenced_by_con = {}
        self._vars_referenced_by_obj = []
        self._params_referenced_by_con = {}
        self._params_referenced_by_obj = []
        self._expr_types = None
        self.config: AutoUpdateConfig = AutoUpdateConfig()(
            value=kwds, preserve_implicit=True
        )

    def set_instance(self, model):
        saved_config = self.config
        self.__init__(observers=self._observers)
        self.config = saved_config
        self._model = model
        self._add_block(model)

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
        for obs in self._observers:
            obs.add_variables(variables)

    def add_parameters(self, params: List[ParamData]):
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
        self.add_variables(list(new_vars.values()))

    def _check_to_remove_vars(self, variables: List[VarData]):
        vars_to_remove = {}
        for v in variables:
            v_id = id(v)
            ref_cons, ref_sos, ref_obj = self._referenced_variables[v_id]
            if len(ref_cons) == 0 and len(ref_sos) == 0 and ref_obj is None:
                vars_to_remove[v_id] = v
        self.remove_variables(list(vars_to_remove.values()))

    def _check_for_new_params(self, params: List[ParamData]):
        new_params = {}
        for p in params:
            pid = id(p)
            if pid not in self._referenced_params:
                new_params[pid] = p
        self.add_parameters(list(new_params.values()))

    def _check_to_remove_params(self, params: List[ParamData]):
        params_to_remove = {}
        for p in params:
            p_id = id(p)
            ref_cons, ref_sos, ref_obj = self._referenced_params[p_id]
            if len(ref_cons) == 0 and len(ref_sos) == 0 and ref_obj is None:
                params_to_remove[p_id] = p
        self.remove_parameters(list(params_to_remove.values()))

    def add_constraints(self, cons: List[ConstraintData]):
        vars_to_check = []
        params_to_check = []
        for con in cons:
            if con in self._active_constraints:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._active_constraints[con] = con.expr
            tmp = collect_components_from_expr(con.expr)
            named_exprs, variables, parameters, external_functions = tmp
            vars_to_check.extend(variables)
            params_to_check.extend(parameters)
            self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
            if len(external_functions) > 0:
                self._external_functions[con] = external_functions
            self._vars_referenced_by_con[con] = variables
            self._params_referenced_by_con[con] = parameters
        self._check_for_new_vars(vars_to_check)
        self._check_for_new_params(params_to_check)
        for con in cons:
            variables = self._vars_referenced_by_con[con]
            parameters = self._params_referenced_by_con[con]
            for v in variables:
                self._referenced_variables[id(v)][0][con] = None
            for p in parameters:
                self._referenced_params[id(p)][0][con] = None
        for obs in self._observers:
            obs.add_constraints(cons)

    def add_sos_constraints(self, cons: List[SOSConstraintData]):
        vars_to_check = []
        params_to_check = []
        for con in cons:
            if con in self._active_sos:
                raise ValueError(f'Constraint {con.name} has already been added')
            sos_items = list(con.get_items())
            self._active_sos[con] = (
                [i[0] for i in sos_items],
                [i[1] for i in sos_items],
            )
            variables = []
            params = []
            for v, p in sos_items:
                variables.append(v)
                if type(p) in native_numeric_types:
                    continue
                if p.is_parameter_type():
                    params.append(p)
            vars_to_check.extend(variables)
            params_to_check.extend(params)
            self._named_expressions[con] = []
            self._vars_referenced_by_con[con] = variables
            self._params_referenced_by_con[con] = params
        self._check_for_new_vars(vars_to_check)
        self._check_for_new_params(params_to_check)
        for con in cons:
            variables = self._vars_referenced_by_con[con]
            params = self._params_referenced_by_con[con]
            for v in variables:
                self._referenced_variables[id(v)][1][con] = None
            for p in params:
                self._referenced_params[id(p)][1][con] = None
        for obs in self._observers:
            obs.add_sos_constraints(cons)

    def set_objective(self, obj: Optional[ObjectiveData]):
        vars_to_remove_check = []
        params_to_remove_check = []
        if self._objective is not None:
            for v in self._vars_referenced_by_obj:
                self._referenced_variables[id(v)][2] = None
            for p in self._params_referenced_by_obj:
                self._referenced_params[id(p)][2] = None
            vars_to_remove_check.extend(self._vars_referenced_by_obj)
            params_to_remove_check.extend(self._params_referenced_by_obj)
            self._external_functions.pop(self._objective, None)
        if obj is not None:
            self._objective = obj
            self._objective_expr = obj.expr
            self._objective_sense = obj.sense
            tmp = collect_components_from_expr(obj.expr)
            named_exprs, variables, parameters, external_functions = tmp
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
        else:
            self._vars_referenced_by_obj = []
            self._params_referenced_by_obj = []
            self._objective = None
            self._objective_expr = None
            self._objective_sense = None
            self._obj_named_expressions = []
        for obs in self._observers:
            obs.set_objective(obj)
        self._check_to_remove_vars(vars_to_remove_check)
        self._check_to_remove_params(params_to_remove_check)

    def _add_block(self, block):
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
        self.set_objective(obj)

    def remove_constraints(self, cons: List[ConstraintData]):
        for obs in self._observers:
            obs.remove_constraints(cons)
        vars_to_check = []
        params_to_check = []
        for con in cons:
            if con not in self._active_constraints:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][0].pop(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[id(p)][0].pop(con)
            vars_to_check.extend(self._vars_referenced_by_con[con])
            params_to_check.extend(self._params_referenced_by_con[con])
            del self._active_constraints[con]
            del self._named_expressions[con]
            self._external_functions.pop(con, None)
            del self._vars_referenced_by_con[con]
            del self._params_referenced_by_con[con]
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

    def remove_sos_constraints(self, cons: List[SOSConstraintData]):
        for obs in self._observers:
            obs.remove_sos_constraints(cons)
        vars_to_check = []
        params_to_check = []
        for con in cons:
            if con not in self._active_sos:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[id(v)][1].pop(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[id(p)][1].pop(con)
            vars_to_check.extend(self._vars_referenced_by_con[con])
            params_to_check.extend(self._params_referenced_by_con[con])
            del self._active_sos[con]
            del self._named_expressions[con]
            del self._vars_referenced_by_con[con]
            del self._params_referenced_by_con[con]
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

    def remove_variables(self, variables: List[VarData]):
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

    def remove_parameters(self, params: List[ParamData]):
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
        for obs in self._observers:
            obs.update_variables(variables)

    def update_parameters(self, params):
        for p in params:
            self._params[id(p)] = (p, p.value)
        for obs in self._observers:
            obs.update_parameters(params)

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
                for c in self._referenced_variables[vid][0]:
                    cons_to_update[c] = None
                if self._referenced_variables[vid][2] is not None:
                    update_obj = True
            elif v._lb is not _lb:
                vars_to_update.append(v)
            elif v._ub is not _ub:
                vars_to_update.append(v)
            elif _domain_interval != v.domain.get_interval():
                vars_to_update.append(v)
            elif v.fixed and v.value != _value:
                vars_to_update.append(v)
        cons_to_update = list(cons_to_update.keys())
        return vars_to_update, cons_to_update, update_obj

    def _check_for_param_changes(self):
        params_to_update = []
        for pid, (p, val) in self._params.items():
            if p.value != val:
                params_to_update.append(p)
        return params_to_update

    def _check_for_named_expression_changes(self):
        cons_to_update = []
        for con, ne_list in self._named_expressions.items():
            for named_expr, old_expr in ne_list:
                if named_expr.expr is not old_expr:
                    cons_to_update.append(con)
                    break
        update_obj = False
        ne_list = self._obj_named_expressions
        for named_expr, old_expr in ne_list:
            if named_expr.expr is not old_expr:
                update_obj = True
                break
        return cons_to_update, update_obj

    def _check_for_new_objective(self):
        update_obj = False
        new_obj = get_objective(self._model)
        if new_obj is not self._objective:
            update_obj = True
        return new_obj, update_obj

    def _check_for_objective_changes(self):
        update_obj = False
        if self._objective is None:
            return update_obj
        if self._objective.expr is not self._objective_expr:
            update_obj = True
        elif self._objective.sense != self._objective_sense:
            # we can definitely do something faster here than resetting the whole objective
            update_obj = True
        return update_obj

    def update(self, timer: Optional[HierarchicalTimer] = None, **kwds):
        if timer is None:
            timer = HierarchicalTimer()
        config: AutoUpdateConfig = self.config(value=kwds, preserve_implicit=True)

        added_cons = set()
        added_sos = set()

        if config.check_for_new_or_removed_constraints:
            timer.start('sos')
            new_sos, old_sos = self._check_for_new_or_removed_sos()
            self.add_sos_constraints(new_sos)
            self.remove_sos_constraints(old_sos)
            added_sos.update(new_sos)
            timer.stop('sos')
            timer.start('cons')
            new_cons, old_cons = self._check_for_new_or_removed_constraints()
            self.add_constraints(new_cons)
            self.remove_constraints(old_cons)
            added_cons.update(new_cons)
            timer.stop('cons')

        if config.update_constraints:
            timer.start('cons')
            cons_to_update = self._check_for_modified_constraints()
            self.remove_constraints(cons_to_update)
            self.add_constraints(cons_to_update)
            added_cons.update(cons_to_update)
            timer.stop('cons')
            timer.start('sos')
            sos_to_update = self._check_for_modified_sos()
            self.remove_sos_constraints(sos_to_update)
            self.add_sos_constraints(sos_to_update)
            added_sos.update(sos_to_update)
            timer.stop('sos')

        need_to_set_objective = False

        if config.update_vars:
            timer.start('vars')
            vars_to_update, cons_to_update, update_obj = self._check_for_var_changes()
            self.update_variables(vars_to_update)
            cons_to_update = [i for i in cons_to_update if i not in added_cons]
            self.remove_constraints(cons_to_update)
            self.add_constraints(cons_to_update)
            added_cons.update(cons_to_update)
            if update_obj:
                need_to_set_objective = True
            timer.stop('vars')

        if config.update_named_expressions:
            timer.start('named expressions')
            cons_to_update, update_obj = self._check_for_named_expression_changes()
            cons_to_update = [i for i in cons_to_update if i not in added_cons]
            self.remove_constraints(cons_to_update)
            self.add_constraints(cons_to_update)
            added_cons.update(cons_to_update)
            if update_obj:
                need_to_set_objective = True
            timer.stop('named expressions')

        timer.start('objective')
        new_obj = self._objective
        if config.check_for_new_objective:
            new_obj, update_obj = self._check_for_new_objective()
            if update_obj:
                need_to_set_objective = True
        if config.update_objective:
            update_obj = self._check_for_objective_changes()
            if update_obj:
                need_to_set_objective = True

        if need_to_set_objective:
            self.set_objective(new_obj)
        timer.stop('objective')

        if config.update_parameters:
            timer.start('params')
            params_to_update = self._check_for_param_changes()
            self.update_parameters(params_to_update)
            timer.stop('params')
