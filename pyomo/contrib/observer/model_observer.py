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
from typing import List, Sequence, Optional, Mapping, MutableMapping, MutableSet, Tuple, Collection, Union

from pyomo.common.enums import ObjectiveSense
from pyomo.common.config import ConfigDict, ConfigValue, document_configdict
from pyomo.core.base.constraint import ConstraintData, Constraint
from pyomo.core.base.sos import SOSConstraintData, SOSConstraint
from pyomo.core.base.var import VarData
from pyomo.core.base.param import ParamData, ScalarParam
from pyomo.core.base.expression import ExpressionData
from pyomo.core.base.objective import ObjectiveData, Objective
from pyomo.core.base.block import BlockData, Block
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.numeric_expr import NumericValue
from pyomo.core.expr.relational_expr import RelationalExpression
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet, DefaultComponentMap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.util import get_objective
from pyomo.contrib.observer.component_collector import collect_components_from_expr
from pyomo.common.numeric_types import native_numeric_types
import warnings
import enum
from collections import defaultdict


# The ModelChangeDetector is meant to be used to automatically identify changes
# in a Pyomo model or block. Here is a list of changes that will be detected. 
# Note that inactive components (e.g., constraints) are treated as "removed".
#   - new constraints that have been added to the model
#   - constraints that have been removed from the model
#   - new variables that have been detected in new or modified constraints/objectives
#   - old variables that are no longer used in any constraints/objectives
#   - new parameters that have been detected in new or modified constraints/objectives
#   - old parameters that are no longer used in any constraints/objectives
#   - new objectives that have been added to the model
#   - objectives that have been removed from the model
#   - modified constraint expressions (relies on expressions being immutable)
#   - modified objective expressions (relies on expressions being immutable)
#   - modified objective sense
#   - changes to variable bounds, domains, and "fixed" flags
#   - changes to named expressions (relies on expressions being immutable)
#   - changes to parameter values and fixed variable values


_param_types = {ParamData, ScalarParam}


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

        #: automatically detect new/removed constraints on subsequent solves
        self.check_for_new_or_removed_constraints: bool = self.declare(
            'check_for_new_or_removed_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old constraints will not be automatically detected on 
                subsequent solves. Use False only when manually updating the change
                detector with cd.add_constraints() and cd.remove_constraints() or 
                when you are certain constraints are not being added to/removed from the 
                model.""",
            ),
        )
        #: automatically detect new/removed objectives on subsequent solves
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
        #: automatically detect changes to constraints on subsequent solves
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
        #: automatically detect changes to variables on subsequent solves
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
                you are certain variables are not being modified.""",
            ),
        )
        #: automatically detect changes to parameters on subsequent solves
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
        #: automatically detect changes to named expressions on subsequent solves
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
        #: automatically detect changes to objectives on subsequent solves
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


class Reason(enum.Flag):
    no_change = 0
    bounds = 1
    fixed = 2
    domain = 4
    value = 8
    added = 16
    removed = 32
    expr = 64
    sense = 128
    sos_items = 256


class Observer(abc.ABC):
    @abc.abstractmethod
    def _update_variables(self, variables: Mapping[VarData, Reason]):
        """
        This method gets called by the ModelChangeDetector when there are 
        any modifications to the set of "active" variables in the model being
        observed. By "active" variables, we mean variables
        that are used within an active component such as a constraint or
        an objective. Changes include new variables being added to the model,
        variables being removed from the model, or changes to variables 
        already in the model

        Parameters
        ----------
        variables: Mapping[VarData, Reason]
            The variables and what changed about them
        """
        pass

    @abc.abstractmethod
    def _update_parameters(self, params: Mapping[ParamData, Reason]):
        """
        This method gets called by the ModelChangeDetector when there are any 
        modifications to the set of "active" parameters in the model being
        observed. By "active" parameters, we mean parameters that are used within 
        an active component such as a constraint or an objective. Changes include 
        parameters being added to the model, parameters being removed from the model,
        or changes to parameters already in the model 

        Parameters
        ----------
        params: Mapping[ParamData, Reason]
            The parameters and what changed about them
        """
        pass

    @abc.abstractmethod
    def _update_constraints(self, cons: Mapping[ConstraintData, Reason]):
        """
        This method gets called by the ModelChangeDetector when there are any 
        modifications to the set of active constraints in the model being observed.
        Changes include constraints being added to the model, constraints being 
        removed from the model, or changes to constraints already in the model.

        Parameters
        ----------
        cons: Mapping[ConstraintData, Reason]
            The constraints and what changed about them
        """
        pass

    @abc.abstractmethod
    def _update_sos_constraints(self, cons: Mapping[SOSConstraintData, Reason]):
        """
        This method gets called by the ModelChangeDetector when there are any 
        modifications to the set of active SOS constraints in the model being 
        observed. Changes include constraints being added to the model, constraints
        being removed from the model, or changes to constraints already in the model.

        Parameters
        ----------
        cons: Mapping[SOSConstraintData, Reason]
            The SOS constraints and what changed about them
        """
        pass

    @abc.abstractmethod
    def _update_objectives(self, objs: Mapping[ObjectiveData, Reason]):
        """
        This method gets called by the ModelChangeDetector when there are any 
        modifications to the set of active objectives in the model being observed.
        Changes include objectives being added to the model, objectives being 
        removed from the model, or changes to objectives already in the model.

        Parameters
        ----------
        objs: Mapping[ObjectiveData, Reason]
            The objectives and what changed about them
        """
        pass


def _default_reason():
    return Reason.no_change


class _Updates:
    def __init__(self) -> None:
        self.vars_to_update = DefaultComponentMap(_default_reason)
        self.params_to_update = DefaultComponentMap(_default_reason)
        self.cons_to_update = defaultdict(_default_reason)
        self.sos_to_update = defaultdict(_default_reason)
        self.objs_to_update = DefaultComponentMap(_default_reason)

    def clear(self):
        self.vars_to_update.clear()
        self.params_to_update.clear()
        self.cons_to_update.clear()
        self.sos_to_update.clear()
        self.objs_to_update.clear()


class ModelChangeDetector:
    """
    This class "watches" a pyomo model and notifies the observers when any
    changes to the model are made (but only when ModelChangeDetector.update
    is called). An example use case is for the persistent solver interfaces.

    The ModelChangeDetector considers the model to be defined by its set of
    active components and any components used by those active components. For
    example, the observers will not be notified of the addition of a variable
    if that variable is not used in any constraints.

    The Observer/ModelChangeDetector are most useful when a small number
    of changes are being made relative to the size of the model. For example,
    the persistent solver interfaces can be very efficient when repeatedly
    solving the same model but with different values for mutable parameters.

    If you know that certain changes will not be made to the model, the
    config can be modified to improve performance. For example, if you
    know that no constraints will be added to or removed from the model,
    then ``check_for_new_or_removed_constraints`` can be set to ``False``,
    which will save some time when ``update`` is called.

    We have discussed expanding the interface of the ``ModelChangeDetector``
    with methods to request extra information. For example, if the value
    of a fixed variable changes, an observer may want to know all of the
    constraints that use the variables. This class already has that
    information, so the observer should not have to waste time recomputing
    that. We have not yet added methods like this because we do not have
    an immediate use case or need, and it's not yet clear what those
    methods should look like. If a need arises, please create an issue or
    pull request.

    Here are some usage examples:

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib.observer.model_observer import (
    ...     AutoUpdateConfig,
    ...     Observer,
    ...     ModelChangeDetector,
    ... )
    >>> class PrintObserver(Observer):
    ...     def add_variables(self, variables):
    ...         for i in variables:
    ...             print(f'{i} was added to the model')
    ...     def add_parameters(self, params):
    ...         for i in params:
    ...             print(f'{i} was added to the model')
    ...     def add_constraints(self, cons):
    ...         for i in cons:
    ...             print(f'{i} was added to the model')
    ...     def add_sos_constraints(self, cons):
    ...         for i in cons:
    ...             print(f'{i} was added to the model')
    ...     def add_objectives(self, objs):
    ...         for i in objs:
    ...             print(f'{i} was added to the model')
    ...     def remove_objectives(self, objs):
    ...         for i in objs:
    ...             print(f'{i} was removed from the model')
    ...     def remove_constraints(self, cons):
    ...         for i in cons:
    ...             print(f'{i} was removed from the model')
    ...     def remove_sos_constraints(self, cons):
    ...         for i in cons:
    ...             print(f'{i} was removed from the model')
    ...     def remove_variables(self, variables):
    ...         for i in variables:
    ...             print(f'{i} was removed from the model')
    ...     def remove_parameters(self, params):
    ...         for i in params:
    ...             print(f'{i} was removed from the model')
    ...     def update_variables(self, variables, reasons):
    ...         for i in variables:
    ...             print(f'{i} was modified')
    ...     def update_parameters(self, params):
    ...         for i in params:
    ...             print(f'{i} was modified')
    >>> m = pyo.ConcreteModel()
    >>> obs = PrintObserver()
    >>> detector = ModelChangeDetector(m, [obs])
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()
    >>> detector.update()  # no output because the variables are not used
    >>> m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
    >>> detector.update()
    x was added to the model
    y was added to the model
    obj was added to the model
    >>> del m.obj
    >>> detector.update()
    obj was removed from the model
    x was removed from the model
    y was removed from the model
    >>> m.px = pyo.Param(mutable=True, initialize=1)
    >>> m.py = pyo.Param(mutable=True, initialize=1)
    >>> m.obj = pyo.Objective(expr=m.px*m.x + m.py*m.y)
    >>> detector.update()
    x was added to the model
    y was added to the model
    px was added to the model
    py was added to the model
    obj was added to the model
    >>> detector.config.check_for_new_or_removed_constraints = False
    >>> detector.config.check_for_new_or_removed_objectives = False
    >>> detector.config.update_constraints = False
    >>> detector.config.update_vars = False
    >>> detector.config.update_parameters = True
    >>> detector.config.update_named_expressions = False
    >>> detector.config.update_objectives = False
    >>> for i in range(10):
    ...     m.py.value = i
    ...     detector.update()  # this will be faster because it is only checking for changes to parameters
    py was modified
    py was modified
    py was modified
    py was modified
    py was modified
    py was modified
    py was modified
    py was modified
    py was modified
    py was modified
    >>> m.c = pyo.Constraint(expr=m.y >= pyo.exp(m.x))
    >>> detector.update()  # no output because we did not check for new constraints
    >>> detector.config.check_for_new_or_removed_constraints = True
    >>> detector.update()
    c was added to the model

    """

    def __init__(self, model: BlockData, observers: Sequence[Observer], **kwds):
        """
        Parameters
        ----------
        model: BlockData
            The model for which changes should be detected
        observers: Sequence[Observer]
            The objects to notify when changes are made to the model
        """
        self._known_active_ctypes = {Constraint, SOSConstraint, Objective, Block}
        self._observers: List[Observer] = list(observers)

        self._active_constraints: MutableMapping[
            ConstraintData, Union[RelationalExpression, None]
        ] = {}

        self._active_sos = {}

        # maps var to (lb, ub, fixed, domain, value)
        self._vars: MutableMapping[VarData, Tuple] = ComponentMap()

        # maps param to value
        self._params: MutableMapping[ParamData, float] = ComponentMap()

        self._objectives: MutableMapping[
            ObjectiveData, Tuple[Union[NumericValue, float, int, None], ObjectiveSense]
        ] = ComponentMap()  # maps objective to (expression, sense)

        # maps constraints/objectives to list of tuples (named_expr, named_expr.expr)
        self._named_expressions: MutableMapping[
            ConstraintData, List[Tuple[ExpressionData, Union[NumericValue, float, int, None]]]
        ] = {}
        self._obj_named_expressions: MutableMapping[
            ObjectiveData, List[Tuple[ExpressionData, Union[NumericValue, float, int, None]]]
        ] = ComponentMap()

        self._external_functions = ComponentMap()

        self._referenced_variables: MutableMapping[
            VarData, 
            Tuple[
                MutableSet[ConstraintData], 
                MutableSet[SOSConstraintData], 
                MutableSet[ObjectiveData]
            ]
        ] = ComponentMap()

        self._referenced_params: MutableMapping[
            ParamData,
            Tuple[
                MutableSet[ConstraintData],
                MutableSet[SOSConstraintData],
                MutableSet[ObjectiveData],
                MutableSet[VarData],
            ]
        ] = ComponentMap()

        self._vars_referenced_by_con: MutableMapping[
            Union[ConstraintData, SOSConstraintData], List[VarData]
        ] = {}
        self._vars_referenced_by_obj: MutableMapping[
            ObjectiveData, List[VarData]
        ] = ComponentMap()
        self._params_referenced_by_con: MutableMapping[
            Union[ConstraintData, SOSConstraintData], List[ParamData]
        ] = {}
        # for when parameters show up in variable bounds
        self._params_referenced_by_var: MutableMapping[
            VarData, List[ParamData]
        ] = ComponentMap()
        self._params_referenced_by_obj: MutableMapping[
            ObjectiveData, List[ParamData]
        ] = ComponentMap()

        self.config: AutoUpdateConfig = AutoUpdateConfig()(
            value=kwds, preserve_implicit=True
        )

        self._updates = _Updates()

        self._model: BlockData = model
        self._set_instance()

    def add_variables(self, variables: Collection[VarData]):
        params_to_check = ComponentSet()
        for v in variables:
            if v in self._referenced_variables:
                raise ValueError(f'Variable {v.name} has already been added')
            self._referenced_variables[v] = (OrderedSet(), OrderedSet(), ComponentSet())
            self._vars[v] = (
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
            ref_params = ComponentSet()
            for bnd in (v._lb, v._ub):
                if bnd is None or type(bnd) in native_numeric_types:
                    continue
                (named_exprs, _vars, parameters, external_functions) = (
                    collect_components_from_expr(bnd)
                )
                if _vars:
                    raise NotImplementedError(
                        'ModelChangeDetector does not support variables in the bounds of other variables'
                    )
                if named_exprs:
                    raise NotImplementedError(
                        'ModelChangeDetector does not support Expressions in the bounds of other variables'
                    )
                if external_functions:
                    raise NotImplementedError(
                        'ModelChangeDetector does not support external functions in the bounds of other variables'
                    )
                params_to_check.update(parameters)
                ref_params.update(parameters)
            if ref_params:
                self._params_referenced_by_var[v] = list(ref_params)
        self._check_for_new_params(params_to_check)
        for v in variables:
            if v not in self._params_referenced_by_var:
                continue
            parameters = self._params_referenced_by_var[v]
            for p in parameters:
                self._referenced_params[p][3].add(v)
        for obs in self._observers:
            obs._update_variables(ComponentMap((v, Reason.added) for v in variables))

    def add_parameters(self, params: Collection[ParamData]):
        for p in params:
            if p in self._referenced_params:
                raise ValueError(f'Parameter {p.name} has already been added')
            self._referenced_params[p] = (OrderedSet(), OrderedSet(), ComponentSet(), ComponentSet())
            self._params[p] = p.value
        for obs in self._observers:
            obs._update_parameters(ComponentMap((p, Reason.added) for p in params))

    def _check_for_new_vars(self, variables: Collection[VarData]):
        new_vars = ComponentSet(
            v for v in variables if v not in self._referenced_variables
        )
        self.add_variables(new_vars)

    def _check_to_remove_vars(self, variables: Collection[VarData]):
        vars_to_remove = ComponentSet()
        for v in variables:
            if not any(self._referenced_variables[v]):
                vars_to_remove.add(v)
        self.remove_variables(vars_to_remove)

    def _check_for_new_params(self, params: Collection[ParamData]):
        new_params = ComponentSet(
            p for p in params if p not in self._referenced_params
        )
        self.add_parameters(new_params)

    def _check_to_remove_params(self, params: Collection[ParamData]):
        params_to_remove = ComponentSet()
        for p in params:
            if not any(self._referenced_params[p]):
                params_to_remove.add(p)
        self.remove_parameters(params_to_remove)

    def add_constraints(self, cons: Collection[ConstraintData]):
        vars_to_check = ComponentSet()
        params_to_check = ComponentSet()
        for con in cons:
            if con in self._active_constraints:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._active_constraints[con] = con.expr
            (named_exprs, variables, parameters, external_functions) = (
                collect_components_from_expr(con.expr)
            )
            vars_to_check.update(variables)
            params_to_check.update(parameters)
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
                self._referenced_variables[v][0].add(con)
            for p in parameters:
                self._referenced_params[p][0].add(con)
        for obs in self._observers:
            obs._update_constraints({c: Reason.added for c in cons})

    def add_sos_constraints(self, cons: Collection[SOSConstraintData]):
        vars_to_check = ComponentSet()
        params_to_check = ComponentSet()
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
            vars_to_check.update(variables)
            params_to_check.update(params)
            self._vars_referenced_by_con[con] = variables
            self._params_referenced_by_con[con] = params
        self._check_for_new_vars(vars_to_check)
        self._check_for_new_params(params_to_check)
        for con in cons:
            variables = self._vars_referenced_by_con[con]
            params = self._params_referenced_by_con[con]
            for v in variables:
                self._referenced_variables[v][1].add(con)
            for p in params:
                self._referenced_params[p][1].add(con)
        for obs in self._observers:
            obs._update_sos_constraints(ComponentMap((c, Reason.added) for c in cons))

    def add_objectives(self, objs: Collection[ObjectiveData]):
        vars_to_check = ComponentSet()
        params_to_check = ComponentSet()
        for obj in objs:
            self._objectives[obj] = (obj.expr, obj.sense)
            (named_exprs, variables, parameters, external_functions) = (
                collect_components_from_expr(obj.expr)
            )
            vars_to_check.update(variables)
            params_to_check.update(parameters)
            if named_exprs:
                self._obj_named_expressions[obj] = [(e, e.expr) for e in named_exprs]
            if external_functions:
                self._external_functions[obj] = external_functions
            self._vars_referenced_by_obj[obj] = variables
            self._params_referenced_by_obj[obj] = parameters
        self._check_for_new_vars(vars_to_check)
        self._check_for_new_params(params_to_check)
        for obj in objs:
            variables = self._vars_referenced_by_obj[obj]
            parameters = self._params_referenced_by_obj[obj]
            for v in variables:
                self._referenced_variables[v][2].add(obj)
            for p in parameters:
                self._referenced_params[p][2].add(obj)
        for obs in self._observers:
            obs._update_objectives(ComponentMap((obj, Reason.added) for obj in objs))

    def remove_objectives(self, objs: Collection[ObjectiveData]):
        for obs in self._observers:
            obs._update_objectives(ComponentMap((obj, Reason.removed) for obj in objs))

        vars_to_check = ComponentSet()
        params_to_check = ComponentSet()
        for obj in objs:
            if obj not in self._objectives:
                raise ValueError(
                    f'cannot remove objective {obj.name} - it was not added'
                )
            for v in self._vars_referenced_by_obj[obj]:
                self._referenced_variables[v][2].remove(obj)
            for p in self._params_referenced_by_obj[obj]:
                self._referenced_params[p][2].remove(obj)
            vars_to_check.update(self._vars_referenced_by_obj[obj])
            params_to_check.update(self._params_referenced_by_obj[obj])
            del self._objectives[obj]
            self._obj_named_expressions.pop(obj, None)
            self._external_functions.pop(obj, None)
            self._vars_referenced_by_obj.pop(obj)
            self._params_referenced_by_obj.pop(obj)
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

    def _check_for_unknown_active_components(self):
        for ctype in self._model.collect_ctypes(active=True, descend_into=True):
            if ctype in self._known_active_ctypes:
                continue
            if ctype is Suffix:
                    warnings.warn(
                        'ModelChangeDetector does not detect changes to suffixes'
                    )
                    continue
            raise NotImplementedError(
                f'ModelChangeDetector does not know how to '
                'handle components with ctype {ctype}'
            )

    def _set_instance(self):

        with PauseGC() as pgc:
            self._check_for_unknown_active_components()

            self.add_constraints(
                list(
                    self._model.component_data_objects(
                        Constraint, descend_into=True, active=True
                    )
                )
            )
            self.add_sos_constraints(
                list(
                    self._model.component_data_objects(
                        SOSConstraint, descend_into=True, active=True
                    )
                )
            )
            self.add_objectives(
                list(
                    self._model.component_data_objects(
                        Objective, descend_into=True, active=True
                    )
                )
            )

    def remove_constraints(self, cons: Collection[ConstraintData]):
        for obs in self._observers:
            obs._update_constraints({c: Reason.removed for c in cons})
        vars_to_check = ComponentSet()
        params_to_check = ComponentSet()
        for con in cons:
            if con not in self._active_constraints:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[v][0].remove(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[p][0].remove(con)
            vars_to_check.update(self._vars_referenced_by_con[con])
            params_to_check.update(self._params_referenced_by_con[con])
            self._active_constraints.pop(con)
            self._named_expressions.pop(con, None)
            self._external_functions.pop(con, None)
            self._vars_referenced_by_con.pop(con)
            self._params_referenced_by_con.pop(con)
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

    def remove_sos_constraints(self, cons: Collection[SOSConstraintData]):
        for obs in self._observers:
            obs._update_sos_constraints({c: Reason.removed for c in cons})
        vars_to_check = ComponentSet()
        params_to_check = ComponentSet()
        for con in cons:
            if con not in self._active_sos:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[v][1].remove(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[p][1].remove(con)
            vars_to_check.update(self._vars_referenced_by_con[con])
            params_to_check.update(self._params_referenced_by_con[con])
            self._active_sos.pop(con)
            self._vars_referenced_by_con.pop(con)
            self._params_referenced_by_con.pop(con)
        self._check_to_remove_vars(vars_to_check)
        self._check_to_remove_params(params_to_check)

    def remove_variables(self, variables: Collection[VarData]):
        for obs in self._observers:
            obs._update_variables(ComponentMap((v, Reason.removed) for v in variables))
        params_to_check = ComponentSet()
        for v in variables:
            if v not in self._referenced_variables:
                raise ValueError(
                    f'Cannot remove variable {v.name} - it has not been added'
                )
            if v in self._params_referenced_by_var:
                for p in self._params_referenced_by_var[v]:
                    self._referenced_params[p][3].remove(v)
                params_to_check.update(self._params_referenced_by_var[v])
                self._params_referenced_by_var.pop(v)
            if any(self._referenced_variables[v]):
                raise ValueError(
                    f'Cannot remove variable {v.name} - it is still being used by constraints/objectives'
                )
            self._referenced_variables.pop(v)
            self._vars.pop(v)
        self._check_to_remove_params(params_to_check)

    def remove_parameters(self, params: Collection[ParamData]):
        for obs in self._observers:
            obs._update_parameters(ComponentMap((p, Reason.removed) for p in params))
        for p in params:
            if p not in self._referenced_params:
                raise ValueError(
                    f'Cannot remove parameter {p.name} - it has not been added'
                )
            if any(self._referenced_params[p]):
                raise ValueError(
                    f'Cannot remove parameter {p.name} - it is still being used by constraints/objectives'
                )
            self._referenced_params.pop(p)
            self._params.pop(p)

    def update_variables(self, variables: Mapping[VarData, Reason]):
        for v in variables:
            self._vars[v] = (
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
        for obs in self._observers:
            obs._update_variables(variables)

    def update_parameters(self, params: Mapping[ParamData, Reason]):
        for p in params:
            self._params[p] = p.value
        for obs in self._observers:
            obs._update_parameters(params)

    def _check_for_new_or_removed_sos(self, sos_to_update=None):
        if sos_to_update is None:
            sos_to_update = defaultdict(_default_reason)
        current_sos_set = OrderedSet(
            self._model.component_data_objects(
                SOSConstraint, descend_into=True, active=True
            )
        )
        for c in current_sos_set:
            if c not in self._active_sos:
                sos_to_update[c] |= Reason.added
        for c in self._active_sos:
            if c not in current_sos_set:
                sos_to_update[c] |= Reason.removed
        return sos_to_update

    def _check_for_new_or_removed_constraints(self, cons_to_update=None):
        if cons_to_update is None:
            cons_to_update = defaultdict(_default_reason)
        current_cons_set = OrderedSet(
            self._model.component_data_objects(
                Constraint, descend_into=True, active=True
            )
        )
        for c in current_cons_set:
            if c not in self._active_constraints:
                cons_to_update[c] |= Reason.added
        for c in self._active_constraints:
            if c not in current_cons_set:
                cons_to_update[c] |= Reason.removed
        return cons_to_update

    def _check_for_modified_sos(self, sos_to_update=None):
        if sos_to_update is None:
            sos_to_update = defaultdict(_default_reason)
        for c, (old_vlist, old_plist) in self._active_sos.items():
            sos_items = list(c.get_items())
            new_vlist = [i[0] for i in sos_items]
            new_plist = [i[1] for i in sos_items]
            if len(old_vlist) != len(new_vlist):
                sos_to_update[c] |= Reason.sos_items
            elif len(old_plist) != len(new_plist):
                sos_to_update[c] |= Reason.sos_items
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
                    sos_to_update[c] |= Reason.sos_items
        return sos_to_update

    def _check_for_modified_constraints(self, cons_to_update=None):
        if cons_to_update is None:
            cons_to_update = defaultdict(_default_reason)
        for c, expr in self._active_constraints.items():
            if c.expr is not expr:
                cons_to_update[c] |= Reason.expr
        return cons_to_update

    def _check_for_var_changes(self, vars_to_update=None):
        if vars_to_update is None:
            vars_to_update = DefaultComponentMap(_default_reason)
        for v, (_lb, _ub, _fixed, _domain_interval, _value) in self._vars.items():
            reason = Reason.no_change
            if _fixed != v.fixed:
                reason = reason | Reason.fixed
            elif _fixed and (v.value != _value):
                reason = reason | Reason.value
            if v._lb is not _lb or v._ub is not _ub:
                reason = reason | Reason.bounds
            if _domain_interval != v.domain.get_interval():
                reason = reason | Reason.domain
            if reason:
                vars_to_update[v] |= reason
        return vars_to_update

    def _check_for_param_changes(self, params_to_update=None):
        if params_to_update is None:
            params_to_update = DefaultComponentMap(_default_reason)
        for p, val in self._params.items():
            if p.value != val:
                params_to_update[p] |= Reason.value
        return params_to_update

    def _check_for_named_expression_changes(self, cons_to_update=None, objs_to_update=None):
        if cons_to_update is None:
            cons_to_update = defaultdict(_default_reason)
        if objs_to_update is None:
            objs_to_update = DefaultComponentMap(_default_reason)
        for con, ne_list in self._named_expressions.items():
            for named_expr, old_expr in ne_list:
                if named_expr.expr is not old_expr:
                    cons_to_update[con] |= Reason.expr
                    break
        for obj, ne_list in self._obj_named_expressions.items():
            for named_expr, old_expr in ne_list:
                if named_expr.expr is not old_expr:
                    objs_to_update[obj] |= Reason.expr
                    break
        return cons_to_update, objs_to_update

    def _check_for_new_or_removed_objectives(self, objs_to_update=None):
        if objs_to_update is None:
            objs_to_update = DefaultComponentMap(_default_reason)
        current_objs_set = ComponentSet(
            self._model.component_data_objects(
                Objective, descend_into=True, active=True
            )
        )
        for obj in current_objs_set:
            if obj not in self._objectives:
                objs_to_update[obj] |= Reason.added
        for obj, (obj_expr, obj_sense) in self._objectives.items():
            if obj not in current_objs_set:
                objs_to_update[obj] |= Reason.removed
        return objs_to_update

    def _check_for_modified_objectives(self, objs_to_update=None):
        if objs_to_update is None:
            objs_to_update = DefaultComponentMap(_default_reason)
        for obj, (obj_expr, obj_sense) in self._objectives.items():
            if obj.expr is not obj_expr:
                objs_to_update[obj] |= Reason.expr
            if obj.sense != obj_sense:
                objs_to_update[obj] |= Reason.sense
        return objs_to_update

    def update(self, timer: Optional[HierarchicalTimer] = None, **kwds):
        """
        Check for changes to the model and notify the observers.

        Parameters
        ----------
        timer: Optional[HierarchicalTimer]
            The timer to use for tracking how much time is spent detecting
            different kinds of changes
        """

        """
        When possible, it is better to add new constraints before removing old 
        constraints. This prevents unnecessarily removing and adding variables.
        If a constraint is removed, any variables that are used only by that 
        constraint will be removed. If there is a new constraint that uses 
        the same variable, then we don't actually need to remove the variable.
        This is hard to avoid when we are modifying a constraint or changing
        the objective. When the objective changes, we remove the old one 
        first just because most things don't handle multiple objectives.

        We check for changes to constraints/objectives before variables/parameters 
        so that we don't waste time updating a variable/parameter that is going to 
        get removed.
        """
        if timer is None:
            timer = HierarchicalTimer()
        config: AutoUpdateConfig = self.config(value=kwds, preserve_implicit=True)

        with PauseGC() as pgc:
            self._check_for_unknown_active_components()

            if config.update_vars:
                timer.start('vars')
                vars_to_update = self._check_for_var_changes()
                if vars_to_update:
                    self.update_variables(vars_to_update, reasons)
                timer.stop('vars')

            if config.update_parameters:
                timer.start('params')
                params_to_update = self._check_for_param_changes()
                if params_to_update:
                    self.update_parameters(params_to_update)
                timer.stop('params')

            if config.update_named_expressions:
                timer.start('named expressions')
                cons_to_update, objs_to_update = (
                    self._check_for_named_expression_changes()
                )
                if cons_to_update:
                    self.remove_constraints(cons_to_update)
                    self.add_constraints(cons_to_update)
                if objs_to_update:
                    self.remove_objectives(objs_to_update)
                    self.add_objectives(objs_to_update)
                timer.stop('named expressions')

            if config.update_constraints:
                timer.start('cons')
                cons_to_update = self._check_for_modified_constraints()
                if cons_to_update:
                    self.remove_constraints(cons_to_update)
                    self.add_constraints(cons_to_update)
                timer.stop('cons')
                timer.start('sos')
                sos_to_update = self._check_for_modified_sos()
                if sos_to_update:
                    self.remove_sos_constraints(sos_to_update)
                    self.add_sos_constraints(sos_to_update)
                timer.stop('sos')

            if config.update_objectives:
                timer.start('objective')
                objs_to_update = self._check_for_modified_objectives()
                if objs_to_update:
                    self.remove_objectives(objs_to_update)
                    self.add_objectives(objs_to_update)
                timer.stop('objective')

            if config.check_for_new_or_removed_constraints:
                timer.start('sos')
                new_sos, old_sos = self._check_for_new_or_removed_sos()
                if new_sos:
                    self.add_sos_constraints(new_sos)
                if old_sos:
                    self.remove_sos_constraints(old_sos)
                timer.stop('sos')
                timer.start('cons')
                new_cons, old_cons = self._check_for_new_or_removed_constraints()
                if new_cons:
                    self.add_constraints(new_cons)
                if old_cons:
                    self.remove_constraints(old_cons)
                timer.stop('cons')

            if config.check_for_new_or_removed_objectives:
                timer.start('objective')
                new_objs, old_objs = self._check_for_new_or_removed_objectives()
                # many solvers require one objective, so we have to remove the
                # old objective first
                if old_objs:
                    self.remove_objectives(old_objs)
                if new_objs:
                    self.add_objectives(new_objs)
                timer.stop('objective')

    def get_variables_impacted_by_param(self, p: ParamData):
        return [self._vars[vid][0] for vid in self._referenced_params[id(p)][3]]

    def get_constraints_impacted_by_param(self, p: ParamData):
        return list(self._referenced_params[id(p)][0])

    def get_constraints_impacted_by_var(self, v: VarData):
        return list(self._referenced_variables[id(v)][0])
