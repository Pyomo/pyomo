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
from typing import List, Sequence, Optional

from pyomo.common.config import ConfigDict, ConfigValue, document_configdict
from pyomo.core.base.constraint import ConstraintData, Constraint
from pyomo.core.base.sos import SOSConstraintData, SOSConstraint
from pyomo.core.base.var import VarData
from pyomo.core.base.param import ParamData
from pyomo.core.base.objective import ObjectiveData, Objective
from pyomo.core.base.block import BlockData
from pyomo.core.base.component import ActiveComponent
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.util import get_objective
from pyomo.contrib.observer.component_collector import collect_components_from_expr
from pyomo.common.numeric_types import native_numeric_types
import gc


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


@document_configdict()
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

        # automatically detect new/removed constraints on subsequent solves
        self.check_for_new_or_removed_constraints: bool = self.declare(
            'check_for_new_or_removed_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old constraints will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver 
                with opt.add_constraints() and opt.remove_constraints() or when you 
                are certain constraints are not being added to/removed from the 
                model.""",
            ),
        )
        # automatically detect new/removed objectives on subsequent solves
        self.check_for_new_or_removed_objectives: bool = self.declare(
            'check_for_new_or_removed_objectives',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old objectives will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver 
                with opt.add_objectives() and opt.remove_objectives() or when you 
                are certain objectives are not being added to/removed from the 
                model.""",
            ),
        )
        # automatically detect changes to constraints on subsequent solves
        self.update_constraints: bool = self.declare(
            'update_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to existing constraints will not be automatically 
                detected on subsequent solves. This includes changes to the lower, 
                body, and upper attributes of constraints. Use False only when 
                manually updating the solver with opt.remove_constraints() and 
                opt.add_constraints() or when you are certain constraints are not 
                being modified.""",
            ),
        )
        # automatically detect changes to variables on subsequent solves
        self.update_vars: bool = self.declare(
            'update_vars',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to existing variables will not be automatically 
                detected on subsequent solves. This includes changes to the lb, ub, 
                domain, and fixed attributes of variables. Use False only when 
                manually updating the observer with opt.update_variables() or when 
                you are certain variables are not being modified. Note that changes 
                to values of fixed variables is handled by 
                update_parameters_and_fixed_vars.""",
            ),
        )
        # automatically detect changes to parameters on subsequent solves
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
        # automatically detect changes to named expressions on subsequent solves
        self.update_named_expressions: bool = self.declare(
            'update_named_expressions',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to Expressions will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver 
                with opt.remove_constraints() and opt.add_constraints() or when you 
                are certain Expressions are not being modified.""",
            ),
        )
        # automatically detect changes to objectives on subsequent solves
        self.update_objectives: bool = self.declare(
            'update_objectives',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to objectives will not be automatically detected on 
                subsequent solves. This includes the expr and sense attributes of 
                objectives. Use False only when manually updating the solver with 
                opt.set_objective() or when you are certain objectives are not being 
                modified.""",
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
    def add_objectives(self, objs: List[ObjectiveData]):
        pass

    @abc.abstractmethod
    def remove_objectives(self, objs: List[ObjectiveData]):
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
    def __init__(self, model: BlockData, observers: Sequence[Observer], **kwds):
        """
        Parameters
        ----------
        observers: Sequence[Observer]
            The objects to notify when changes are made to the model
        """
        self._known_active_ctypes = {Constraint, SOSConstraint, Objective}
        self._observers: List[Observer] = list(observers)
        self._active_constraints = {}  # maps constraint to expression
        self._active_sos = {}
        self._vars = {}  # maps var id to (var, lb, ub, fixed, domain, value)
        self._params = {}  # maps param id to param
        self._objectives = {}  # maps objective id to (objective, expression, sense)

        # maps constraints/objectives to list of tuples (named_expr, named_expr.expr)
        self._named_expressions = {}
        self._obj_named_expressions = {}

        self._external_functions = ComponentMap()

        # the dictionaries below are really just ordered sets, but we need to 
        # stick with built-in types for performance

        # var_id: (
        #     dict[constraints, None],
        #     dict[sos constraints, None], 
        #     dict[objectives, None],
        # )
        self._referenced_variables = {}

        # param_id: (
        #     dict[constraints, None], 
        #     dict[sos constraints, None],
        #     dict[objectives, None],
        # )
        self._referenced_params = {}

        self._vars_referenced_by_con = {}
        self._vars_referenced_by_obj = {}
        self._params_referenced_by_con = {}
        self._params_referenced_by_obj = {}

        self.config: AutoUpdateConfig = AutoUpdateConfig()(
            value=kwds, preserve_implicit=True
        )

        self._model = model
        self._set_instance()

    def _add_variables(self, variables: List[VarData]):
        for v in variables:
            if id(v) in self._referenced_variables:
                raise ValueError(f'Variable {v.name} has already been added')
            self._referenced_variables[id(v)] = ({}, {}, {})
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
            self._referenced_params[pid] = ({}, {}, {})
            self._params[id(p)] = (p, p.value)
        for obs in self._observers:
            obs.add_parameters(params)

    def _check_for_new_vars(self, variables: List[VarData]):
        new_vars = []
        for v in variables:
            if id(v) not in self._referenced_variables:
                new_vars.append(v)
        self._add_variables(new_vars)

    def _check_to_remove_vars(self, variables: List[VarData]):
        vars_to_remove = []
        for v in variables:
            v_id = id(v)
            ref_cons, ref_sos, ref_obj = self._referenced_variables[v_id]
            if not ref_cons and not ref_sos and not ref_obj:
                vars_to_remove.append(v)
        self._remove_variables(vars_to_remove)

    def _check_for_new_params(self, params: List[ParamData]):
        new_params = []
        for p in params:
            if id(p) not in self._referenced_params:
                new_params.append(p)
        self._add_parameters(new_params)

    def _check_to_remove_params(self, params: List[ParamData]):
        params_to_remove = []
        for p in params:
            p_id = id(p)
            ref_cons, ref_sos, ref_obj = self._referenced_params[p_id]
            if not ref_cons and not ref_sos and not ref_obj:
                params_to_remove.append(p)
        self._remove_parameters(params_to_remove)

    def _add_constraints(self, cons: List[ConstraintData]):
        vars_to_check = []
        params_to_check = []
        for con in cons:
            if con in self._active_constraints:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._active_constraints[con] = con.expr
            (
                named_exprs,
                variables,
                parameters,
                external_functions,
            ) = collect_components_from_expr(con.expr)
            vars_to_check.extend(variables)
            params_to_check.extend(parameters)
            if named_exprs:
                self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
            if external_functions:
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

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
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

    def _add_objectives(self, objs: List[ObjectiveData]):
        vars_to_check = []
        params_to_check = []
        for obj in objs:
            obj_id = id(obj)
            self._objectives[obj_id] = (obj, obj.expr, obj.sense)
            (
                named_exprs, 
                variables,
                parameters,
                external_functions,
            ) = collect_components_from_expr(obj.expr)
            vars_to_check.extend(variables)
            params_to_check.extend(parameters)
            if named_exprs:
                self._obj_named_expressions[obj_id] = [(e, e.expr) for e in named_exprs]
            if external_functions:
                self._external_functions[obj] = external_functions
            self._vars_referenced_by_obj[obj_id] = variables
            self._params_referenced_by_obj[obj_id] = parameters
        self._check_for_new_vars(vars_to_check)
        self._check_for_new_params(params_to_check)
        for obj in objs:
            obj_id = id(obj)
            variables = self._vars_referenced_by_obj[obj_id]
            parameters = self._params_referenced_by_obj[obj_id]
            for v in variables:
                self._referenced_variables[id(v)][2][obj_id] = None
            for p in parameters:
                self._referenced_params[id(p)][2][obj_id] = None
        for obs in self._observers:
            obs.add_objectives(objs)

    def _remove_objectives(self, objs: List[ObjectiveData]):
        for obs in self._observers:
            obs.remove_objectives(objs)

        vars_to_check = []
        params_to_check = []
        for obj in objs:
            obj_id = id(obj)
            if obj_id not in self._objectives:
                raise ValueError(
                    f'cannot remove objective {obj.name} - it was not added'
                )
            for v in self._vars_referenced_by_obj[obj_id]:
                self._referenced_variables[id(v)][2].pop(obj_id)
            for p in self._params_referenced_by_obj[obj_id]:
                self._referenced_params[id(p)][2].pop(obj_id)
            vars_to_check.extend(self._vars_referenced_by_obj[obj_id])
            params_to_check.extend(self._params_referenced_by_obj[obj_id])
            del self._objectives[obj_id]
            self._obj_named_expressions.pop(obj_id, None)
            self._external_functions.pop(obj, None)
            del self._vars_referenced_by_obj[obj_id]
            del self._params_referenced_by_obj[obj_id]
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

    def _check_for_unknown_active_components(self):
        for ctype in self._model.collect_ctypes():
            if not issubclass(ctype, ActiveComponent):
                continue
            if ctype in self._known_active_ctypes:
                continue
            for comp in self._model.component_data_objects(
                ctype, 
                active=True, 
                descend_into=True
            ):
                raise NotImplementedError(
                    f'ModelChangeDetector does not know how to '
                    'handle compents with ctype {ctype}'
                )

    def _set_instance(self):

        is_gc_enabled = gc.isenabled()
        gc.disable()

        try:
            self._check_for_unknown_active_components()

            self._add_constraints(
                list(
                    self._model.component_data_objects(Constraint, descend_into=True, active=True)
                )
            )
            self._add_sos_constraints(
                list(
                    self._model.component_data_objects(
                        SOSConstraint, descend_into=True, active=True,
                    )
                )
            )
            self._add_objectives(
                list(
                    self._model.component_data_objects(
                        Objective, descend_into=True, active=True,
                    )
                )
            )
        finally:
            if is_gc_enabled:
                gc.enable()

    def _remove_constraints(self, cons: List[ConstraintData]):
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
            self._named_expressions.pop(con, None)
            self._external_functions.pop(con, None)
            del self._vars_referenced_by_con[con]
            del self._params_referenced_by_con[con]
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
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
            del self._vars_referenced_by_con[con]
            del self._params_referenced_by_con[con]
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

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
            if cons_using or sos_using or obj_using:
                raise ValueError(
                    f'Cannot remove variable {v.name} - it is still being used by constraints/objectives'
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
            if cons_using or sos_using or obj_using:
                raise ValueError(
                    f'Cannot remove parameter {p.name} - it is still being used by constraints/objectives'
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

    def _update_parameters(self, params):
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
        objs_to_update = {}
        for vid, (v, _lb, _ub, _fixed, _domain_interval, _value) in self._vars.items():
            if v.fixed != _fixed:
                vars_to_update.append(v)
                for c in self._referenced_variables[vid][0]:
                    cons_to_update[c] = None
                for obj_id in self._referenced_variables[vid][2]:
                    objs_to_update[obj_id] = None
            elif _fixed and v.value != _value:
                vars_to_update.append(v)
            elif v._lb is not _lb:
                vars_to_update.append(v)
            elif v._ub is not _ub:
                vars_to_update.append(v)
            elif _domain_interval != v.domain.get_interval():
                vars_to_update.append(v)
        cons_to_update = list(cons_to_update.keys())
        objs_to_update = [self._objectives[obj_id][0] for obj_id in objs_to_update.keys()]
        return vars_to_update, cons_to_update, objs_to_update

    def _check_for_param_changes(self):
        params_to_update = []
        for _, (p, val) in self._params.items():
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
        objs_to_update = []
        for obj_id, ne_list in self._obj_named_expressions.items():
            for named_expr, old_expr in ne_list:
                if named_expr.expr is not old_expr:
                    objs_to_update.append(self._objectives[obj_id][0])
                    break
        return cons_to_update, objs_to_update

    def _check_for_new_or_removed_objectives(self):
        new_objs = []
        old_objs = []
        current_objs_dict = {
            id(obj): obj for obj in self._model.component_data_objects(
                Objective, descend_into=True, active=True
            )
        }
        for obj_id, obj in current_objs_dict.items():
            if obj_id not in self._objectives:
                new_objs.append(obj)
        for obj_id, (obj, obj_expr, obj_sense) in self._objectives.items():
            if obj_id not in current_objs_dict:
                old_objs.append(obj)
        return new_objs, old_objs

    def _check_for_modified_objectives(self):
        objs_to_update = []
        for obj_id, (obj, obj_expr, obj_sense) in self._objectives.items():
            if obj.expr is not obj_expr or obj.sense != obj_sense:
                objs_to_update.append(obj)
        return objs_to_update

    def update(self, timer: Optional[HierarchicalTimer] = None, **kwds):
        if timer is None:
            timer = HierarchicalTimer()
        config: AutoUpdateConfig = self.config(value=kwds, preserve_implicit=True)

        is_gc_enabled = gc.isenabled()
        gc.disable()

        try:
            self._check_for_unknown_active_components()

            added_cons = set()
            added_sos = set()
            added_objs = {}

            if config.check_for_new_or_removed_constraints:
                timer.start('sos')
                new_sos, old_sos = self._check_for_new_or_removed_sos()
                if new_sos:
                    self._add_sos_constraints(new_sos)
                if old_sos:
                    self._remove_sos_constraints(old_sos)
                added_sos.update(new_sos)
                timer.stop('sos')
                timer.start('cons')
                new_cons, old_cons = self._check_for_new_or_removed_constraints()
                if new_cons:
                    self._add_constraints(new_cons)
                if old_cons:
                    self._remove_constraints(old_cons)
                added_cons.update(new_cons)
                timer.stop('cons')

            if config.update_constraints:
                timer.start('cons')
                cons_to_update = self._check_for_modified_constraints()
                if cons_to_update:
                    self._remove_constraints(cons_to_update)
                    self._add_constraints(cons_to_update)
                added_cons.update(cons_to_update)
                timer.stop('cons')
                timer.start('sos')
                sos_to_update = self._check_for_modified_sos()
                if sos_to_update:
                    self._remove_sos_constraints(sos_to_update)
                    self._add_sos_constraints(sos_to_update)
                added_sos.update(sos_to_update)
                timer.stop('sos')

            if config.check_for_new_or_removed_objectives:
                timer.start('objective')
                new_objs, old_objs = self._check_for_new_or_removed_objectives()
                # many solvers require one objective, so we have to remove the 
                # old objective first
                if old_objs:
                    self._remove_objectives(old_objs)
                if new_objs:
                    self._add_objectives(new_objs)
                added_objs.update((id(i), i) for i in new_objs)
                timer.stop('objective')

            if config.update_objectives:
                timer.start('objective')
                objs_to_update = self._check_for_modified_objectives()
                if objs_to_update:
                    self._remove_objectives(objs_to_update)
                    self._add_objectives(objs_to_update)
                added_objs.update((id(i), i) for i in objs_to_update)
                timer.stop('objective')

            if config.update_vars:
                timer.start('vars')
                vars_to_update, cons_to_update, objs_to_update = self._check_for_var_changes()
                if vars_to_update:
                    self._update_variables(vars_to_update)
                cons_to_update = [i for i in cons_to_update if i not in added_cons]
                objs_to_update = [i for i in objs_to_update if id(i) not in added_objs]
                if cons_to_update:
                    self._remove_constraints(cons_to_update)
                    self._add_constraints(cons_to_update)
                added_cons.update(cons_to_update)
                if objs_to_update:
                    self._remove_objectives(objs_to_update)
                    self._add_objectives(objs_to_update)
                added_objs.update((id(i), i) for i in objs_to_update)
                timer.stop('vars')

            if config.update_named_expressions:
                timer.start('named expressions')
                cons_to_update, objs_to_update = self._check_for_named_expression_changes()
                cons_to_update = [i for i in cons_to_update if i not in added_cons]
                objs_to_update = [i for i in objs_to_update if id(i) not in added_objs]
                if cons_to_update:
                    self._remove_constraints(cons_to_update)
                    self._add_constraints(cons_to_update)
                added_cons.update(cons_to_update)
                if objs_to_update:
                    self._remove_objectives(objs_to_update)
                    self._add_objectives(objs_to_update)
                added_objs.update((id(i), i) for i in objs_to_update)
                timer.stop('named expressions')

            if config.update_parameters:
                timer.start('params')
                params_to_update = self._check_for_param_changes()
                if params_to_update:
                    self._update_parameters(params_to_update)
                timer.stop('params')
        finally:
            if is_gc_enabled:
                gc.enable()
