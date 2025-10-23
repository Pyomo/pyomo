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

from __future__ import annotations
import abc
from typing import (
    List,
    Sequence,
    Optional,
    Mapping,
    MutableMapping,
    MutableSet,
    Tuple,
    Collection,
    Union,
)

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
from pyomo.core.base.component import ActiveComponent
from pyomo.core.expr.numeric_expr import NumericValue
from pyomo.core.expr.relational_expr import RelationalExpression
from pyomo.common.collections import (
    ComponentMap,
    ComponentSet,
    OrderedSet,
    DefaultComponentMap,
)
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
#   - changes to variable bounds, domains, "fixed" flags, and values for fixed variables
#   - changes to named expressions (relies on expressions being immutable)
#   - changes to parameter values


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
    def __init__(self, observers: Collection[Observer]) -> None:
        self.vars_to_update = DefaultComponentMap(_default_reason)
        self.params_to_update = DefaultComponentMap(_default_reason)
        self.cons_to_update = defaultdict(_default_reason)
        self.sos_to_update = defaultdict(_default_reason)
        self.objs_to_update = DefaultComponentMap(_default_reason)
        self.observers = observers

    def run(self):
        # split up new, removed, and modified variables
        new_vars = ComponentMap(
            (k, v) for k, v in self.vars_to_update.items() if v & Reason.added
        )
        other_vars = ComponentMap(
            (k, v) for k, v in self.vars_to_update.items() if not (v & Reason.added)
        )

        new_params = ComponentMap(
            (k, v) for k, v in self.params_to_update.items() if v & Reason.added
        )
        other_params = ComponentMap(
            (k, v) for k, v in self.params_to_update.items() if not (v & Reason.added)
        )

        for obs in self.observers:
            if new_vars:
                obs._update_variables(new_vars)
            if new_params:
                obs._update_parameters(new_params)
            if self.cons_to_update:
                obs._update_constraints(self.cons_to_update)
            if self.sos_to_update:
                obs._update_sos_constraints(self.sos_to_update)
            if self.objs_to_update:
                obs._update_objectives(self.objs_to_update)
            if other_vars:
                obs._update_variables(other_vars)
            if other_params:
                obs._update_parameters(other_params)

        self.clear()

    def clear(self):
        self.vars_to_update.clear()
        self.params_to_update.clear()
        self.cons_to_update.clear()
        self.sos_to_update.clear()
        self.objs_to_update.clear()


"""
There are three stages:
- identification of differences between the model and the internal data structures of the Change Detector
- synchronization of the model with the internal data structures of the ChangeDetector
- notification of the observers

The first two really happen at the same time

Update order when notifying the observers:
  - add new variables
  - add new constraints
  - add new objectives
  - remove old constraints
  - remove old objectives
  - remove old variables
  - update modified constraints
  - update modified objectives
  - update modified variables
"""


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

    Here are some usage examples:

    >>> import pyomo.environ as pyo
    >>> from typing import Mapping
    >>> from pyomo.contrib.observer.model_observer import (
    ...     AutoUpdateConfig,
    ...     Observer,
    ...     ModelChangeDetector,
    ...     Reason,
    ... )
    >>> from pyomo.core.base import (
    ...     VarData,
    ...     ParamData,
    ...     ConstraintData,
    ...     SOSConstraintData,
    ...     ObjectiveData,
    ... )
    >>> class PrintObserver(Observer):
    ...     def _update_variables(self, vars: Mapping[VarData, Reason]):
    ...         for v, r in vars.items():
    ...             print(f'{v}: {r.name}')
    ...     def _update_parameters(self, params: Mapping[ParamData, Reason]):
    ...         for p, r in params.items():
    ...             print(f'{p}: {r.name}')
    ...     def _update_constraints(self, cons: Mapping[ConstraintData, Reason]):
    ...         for c, r in cons.items():
    ...             print(f'{c}: {r.name}')
    ...     def _update_sos_constraints(self, cons: Mapping[SOSConstraintData, Reason]):
    ...         for c, r in cons.items():
    ...             print(f'{c}: {r.name}')
    ...     def _update_objectives(self, objs: Mapping[ObjectiveData, Reason]):
    ...         for o, r in objs.items():
    ...             print(f'{o}: {r.name}')
    >>> m = pyo.ConcreteModel()
    >>> obs = PrintObserver()
    >>> detector = ModelChangeDetector(m, [obs])
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()
    >>> detector.update()  # no output because the variables are not used
    >>> m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
    >>> detector.update()
    x: added
    y: added
    obj: added
    >>> del m.obj
    >>> detector.update()
    obj: removed
    x: removed
    y: removed
    >>> m.px = pyo.Param(mutable=True, initialize=1)
    >>> m.py = pyo.Param(mutable=True, initialize=1)
    >>> m.obj = pyo.Objective(expr=m.px*m.x + m.py*m.y)
    >>> detector.update()
    x: added
    y: added
    px: added
    py: added
    obj: added
    >>> m.px.value = 2
    >>> detector.update()
    px: value
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
    py: value
    py: value
    py: value
    py: value
    py: value
    py: value
    py: value
    py: value
    py: value
    py: value
    >>> m.c = pyo.Constraint(expr=m.y >= pyo.exp(m.x))
    >>> detector.update()  # no output because we did not check for new constraints
    >>> detector.config.check_for_new_or_removed_constraints = True
    >>> detector.update()
    c: added

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
            ConstraintData,
            List[Tuple[ExpressionData, Union[NumericValue, float, int, None]]],
        ] = {}
        self._obj_named_expressions: MutableMapping[
            ObjectiveData,
            List[Tuple[ExpressionData, Union[NumericValue, float, int, None]]],
        ] = ComponentMap()

        self._external_functions = ComponentMap()

        self._referenced_variables: MutableMapping[
            VarData,
            Tuple[
                MutableSet[ConstraintData],
                MutableSet[SOSConstraintData],
                MutableSet[ObjectiveData],
            ],
        ] = ComponentMap()

        self._referenced_params: MutableMapping[
            ParamData,
            Tuple[
                MutableSet[ConstraintData],
                MutableSet[SOSConstraintData],
                MutableSet[ObjectiveData],
                MutableSet[VarData],
            ],
        ] = ComponentMap()

        self._vars_referenced_by_con: MutableMapping[
            Union[ConstraintData, SOSConstraintData], MutableSet[VarData]
        ] = {}
        self._vars_referenced_by_obj: MutableMapping[
            ObjectiveData, MutableSet[VarData]
        ] = ComponentMap()
        self._params_referenced_by_con: MutableMapping[
            Union[ConstraintData, SOSConstraintData], MutableSet[ParamData]
        ] = {}
        # for when parameters show up in variable bounds
        self._params_referenced_by_var: MutableMapping[
            VarData, MutableSet[ParamData]
        ] = ComponentMap()
        self._params_referenced_by_obj: MutableMapping[
            ObjectiveData, MutableSet[ParamData]
        ] = ComponentMap()

        self.config: AutoUpdateConfig = AutoUpdateConfig()(
            value=kwds, preserve_implicit=True
        )

        self._updates = _Updates(self._observers)

        self._model: BlockData = model
        self._set_instance()

    def _add_variables(self, variables: Collection[VarData]):
        for v in variables:
            if v in self._referenced_variables:
                raise ValueError(f'Variable {v.name} has already been added')
            self._updates.vars_to_update[v] |= Reason.added
            self._referenced_variables[v] = (OrderedSet(), OrderedSet(), ComponentSet())
            self._vars[v] = (v._lb, v._ub, v.fixed, v.domain.get_interval(), v.value)
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
                ref_params.update(parameters)
            self._params_referenced_by_var[v] = ref_params
            if ref_params:
                self._check_for_new_params(ref_params)
                for p in ref_params:
                    self._referenced_params[p][3].add(v)

    def _add_parameters(self, params: Collection[ParamData]):
        for p in params:
            if p in self._referenced_params:
                raise ValueError(f'Parameter {p.name} has already been added')
            self._updates.params_to_update[p] |= Reason.added
            self._referenced_params[p] = (
                OrderedSet(),
                OrderedSet(),
                ComponentSet(),
                ComponentSet(),
            )
            self._params[p] = p.value

    def _check_for_new_vars(self, variables: Collection[VarData]):
        new_vars = ComponentSet(
            v for v in variables if v not in self._referenced_variables
        )
        self._add_variables(new_vars)

    def _check_to_remove_vars(self, variables: Collection[VarData]):
        vars_to_remove = ComponentSet()
        for v in variables:
            if not any(self._referenced_variables[v]):
                vars_to_remove.add(v)
        self._remove_variables(vars_to_remove)

    def _check_for_new_params(self, params: Collection[ParamData]):
        new_params = ComponentSet(p for p in params if p not in self._referenced_params)
        self._add_parameters(new_params)

    def _check_to_remove_params(self, params: Collection[ParamData]):
        params_to_remove = ComponentSet()
        for p in params:
            if not any(self._referenced_params[p]):
                params_to_remove.add(p)
        self._remove_parameters(params_to_remove)

    def _add_constraints(self, cons: Collection[ConstraintData]):
        for con in cons:
            if con in self._active_constraints:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._updates.cons_to_update[con] |= Reason.added
            self._active_constraints[con] = con.expr
            (named_exprs, variables, parameters, external_functions) = (
                collect_components_from_expr(con.expr)
            )
            self._check_for_new_vars(variables)
            self._check_for_new_params(parameters)
            if named_exprs:
                self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
            if external_functions:
                self._external_functions[con] = external_functions
            self._vars_referenced_by_con[con] = variables
            self._params_referenced_by_con[con] = parameters
            for v in variables:
                self._referenced_variables[v][0].add(con)
            for p in parameters:
                self._referenced_params[p][0].add(con)

    def add_constraints(self, cons: Collection[ConstraintData]):
        self._add_constraints(cons)
        self._updates.run()

    def _add_sos_constraints(self, cons: Collection[SOSConstraintData]):
        for con in cons:
            if con in self._active_sos:
                raise ValueError(f'Constraint {con.name} has already been added')
            self._updates.sos_to_update[con] |= Reason.added
            sos_items = list(con.get_items())
            self._active_sos[con] = (
                [i[0] for i in sos_items],
                [i[1] for i in sos_items],
            )
            variables = ComponentSet()
            params = ComponentSet()
            for v, p in sos_items:
                variables.add(v)
                if type(p) in native_numeric_types:
                    continue
                if p.is_parameter_type():
                    params.add(p)
            self._check_for_new_vars(variables)
            self._check_for_new_params(params)
            self._vars_referenced_by_con[con] = variables
            self._params_referenced_by_con[con] = params
            for v in variables:
                self._referenced_variables[v][1].add(con)
            for p in params:
                self._referenced_params[p][1].add(con)

    def add_sos_constraints(self, cons: Collection[SOSConstraintData]):
        self._add_sos_constraints(cons)
        self._updates.run()

    def _add_objectives(self, objs: Collection[ObjectiveData]):
        for obj in objs:
            self._updates.objs_to_update[obj] |= Reason.added
            self._objectives[obj] = (obj.expr, obj.sense)
            (named_exprs, variables, parameters, external_functions) = (
                collect_components_from_expr(obj.expr)
            )
            self._check_for_new_vars(variables)
            self._check_for_new_params(parameters)
            if named_exprs:
                self._obj_named_expressions[obj] = [(e, e.expr) for e in named_exprs]
            if external_functions:
                self._external_functions[obj] = external_functions
            self._vars_referenced_by_obj[obj] = variables
            self._params_referenced_by_obj[obj] = parameters
            for v in variables:
                self._referenced_variables[v][2].add(obj)
            for p in parameters:
                self._referenced_params[p][2].add(obj)

    def add_objectives(self, objs: Collection[ObjectiveData]):
        self._add_objectives(objs)
        self._updates.run()

    def _remove_objectives(self, objs: Collection[ObjectiveData]):
        for obj in objs:
            if obj not in self._objectives:
                raise ValueError(
                    f'cannot remove objective {obj.name} - it was not added'
                )
            self._updates.objs_to_update[obj] |= Reason.removed
            for v in self._vars_referenced_by_obj[obj]:
                self._referenced_variables[v][2].remove(obj)
            for p in self._params_referenced_by_obj[obj]:
                self._referenced_params[p][2].remove(obj)
            self._check_to_remove_vars(self._vars_referenced_by_obj[obj])
            self._check_to_remove_params(self._params_referenced_by_obj[obj])
            del self._objectives[obj]
            self._obj_named_expressions.pop(obj, None)
            self._external_functions.pop(obj, None)
            self._vars_referenced_by_obj.pop(obj)
            self._params_referenced_by_obj.pop(obj)

    def remove_objectives(self, objs: Collection[ObjectiveData]):
        self._remove_objectives(objs)
        self._updates.run()

    def _check_for_unknown_active_components(self):
        for ctype in self._model.collect_ctypes(active=True, descend_into=True):
            if not issubclass(ctype, ActiveComponent):
                continue
            if ctype in self._known_active_ctypes:
                continue
            if ctype is Suffix:
                warnings.warn('ModelChangeDetector does not detect changes to suffixes')
                continue
            raise NotImplementedError(
                f'ModelChangeDetector does not know how to '
                f'handle components with ctype {ctype}'
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

    def _remove_constraints(self, cons: Collection[ConstraintData]):
        for con in cons:
            if con not in self._active_constraints:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            self._updates.cons_to_update[con] |= Reason.removed
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[v][0].remove(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[p][0].remove(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            self._check_to_remove_params(self._params_referenced_by_con[con])
            self._active_constraints.pop(con)
            self._named_expressions.pop(con, None)
            self._external_functions.pop(con, None)
            self._vars_referenced_by_con.pop(con)
            self._params_referenced_by_con.pop(con)

    def remove_constraints(self, cons: Collection[ConstraintData]):
        self._remove_constraints(cons)
        self._updates.run()

    def _remove_sos_constraints(self, cons: Collection[SOSConstraintData]):
        for con in cons:
            if con not in self._active_sos:
                raise ValueError(
                    f'Cannot remove constraint {con.name} - it was not added'
                )
            self._updates.sos_to_update[con] |= Reason.removed
            for v in self._vars_referenced_by_con[con]:
                self._referenced_variables[v][1].remove(con)
            for p in self._params_referenced_by_con[con]:
                self._referenced_params[p][1].remove(con)
            self._check_to_remove_vars(self._vars_referenced_by_con[con])
            self._check_to_remove_params(self._params_referenced_by_con[con])
            self._active_sos.pop(con)
            self._vars_referenced_by_con.pop(con)
            self._params_referenced_by_con.pop(con)

    def remove_sos_constraints(self, cons: Collection[SOSConstraintData]):
        self._remove_sos_constraints(cons)
        self._updates.run()

    def _remove_variables(self, variables: Collection[VarData]):
        for v in variables:
            if v not in self._referenced_variables:
                raise ValueError(
                    f'Cannot remove variable {v.name} - it has not been added'
                )
            self._updates.vars_to_update[v] |= Reason.removed
            for p in self._params_referenced_by_var[v]:
                self._referenced_params[p][3].remove(v)
            self._check_to_remove_params(self._params_referenced_by_var[v])
            self._params_referenced_by_var.pop(v)
            if any(self._referenced_variables[v]):
                raise ValueError(
                    f'Cannot remove variable {v.name} - it is still being used by constraints/objectives'
                )
            self._referenced_variables.pop(v)
            self._vars.pop(v)

    def _remove_parameters(self, params: Collection[ParamData]):
        for p in params:
            if p not in self._referenced_params:
                raise ValueError(
                    f'Cannot remove parameter {p.name} - it has not been added'
                )
            self._updates.params_to_update[p] |= Reason.removed
            if any(self._referenced_params[p]):
                raise ValueError(
                    f'Cannot remove parameter {p.name} - it is still being used by constraints/objectives'
                )
            self._referenced_params.pop(p)
            self._params.pop(p)

    def _update_var_bounds(self, v: VarData):
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
            ref_params.update(parameters)

        _ref_params = self._params_referenced_by_var[v]
        new_params = ref_params - _ref_params
        old_params = _ref_params - ref_params

        self._params_referenced_by_var[v] = ref_params

        if new_params:
            self._check_for_new_params(new_params)

        for p in new_params:
            self._referenced_params[p][3].add(v)
        for p in old_params:
            self._referenced_params[p][3].remove(v)

        if old_params:
            self._check_to_remove_params(old_params)

    def _update_variables(self, variables: Optional[Collection[VarData]] = None):
        if variables is None:
            variables = self._vars
        for v in variables:
            _lb, _ub, _fixed, _domain_interval, _value = self._vars[v]
            lb, ub, fixed, domain_interval, value = (
                v._lb,
                v._ub,
                v.fixed,
                v.domain.get_interval(),
                v.value,
            )
            reason = Reason.no_change
            if _fixed != fixed:
                reason |= Reason.fixed
            elif _fixed and (value != _value):
                reason |= Reason.value
            if lb is not _lb or ub is not _ub:
                reason |= Reason.bounds
            if _domain_interval != domain_interval:
                reason |= Reason.domain
            if reason:
                self._updates.vars_to_update[v] |= reason
                self._vars[v] = (lb, ub, fixed, domain_interval, value)
                if reason & Reason.bounds:
                    self._update_var_bounds(v)

    def update_variables(self, variables: Optional[Collection[VarData]] = None):
        self._update_variables(variables)
        self._updates.run()

    def _update_parameters(self, params: Optional[Collection[ParamData]] = None):
        if params is None:
            params = self._params
        for p in params:
            _val = self._params[p]
            val = p.value
            reason = Reason.no_change
            if _val != val:
                reason |= Reason.value
            if reason:
                self._updates.params_to_update[p] |= reason
                self._params[p] = val

    def update_parameters(self, params: Optional[Collection[ParamData]]):
        self._update_parameters(params)
        self._updates.run()

    def _update_con(self, con: ConstraintData):
        self._active_constraints[con] = con.expr
        (named_exprs, variables, parameters, external_functions) = (
            collect_components_from_expr(con.expr)
        )
        if named_exprs:
            self._named_expressions[con] = [(e, e.expr) for e in named_exprs]
        else:
            self._named_expressions.pop(con, None)
        if external_functions:
            self._external_functions[con] = external_functions
        else:
            self._external_functions.pop(con, None)

        _variables = self._vars_referenced_by_con[con]
        _parameters = self._params_referenced_by_con[con]
        new_vars = variables - _variables
        old_vars = _variables - variables
        new_params = parameters - _parameters
        old_params = _parameters - parameters

        self._vars_referenced_by_con[con] = variables
        self._params_referenced_by_con[con] = parameters

        if new_vars:
            self._check_for_new_vars(new_vars)
        if new_params:
            self._check_for_new_params(new_params)

        for v in new_vars:
            self._referenced_variables[v][0].add(con)
        for v in old_vars:
            self._referenced_variables[v][0].remove(con)
        for p in new_params:
            self._referenced_params[p][0].add(con)
        for p in old_params:
            self._referenced_params[p][0].remove(con)

        if old_vars:
            self._check_to_remove_vars(old_vars)
        if old_params:
            self._check_to_remove_params(old_params)

    def _update_constraints(self, cons: Optional[Collection[ConstraintData]] = None):
        if cons is None:
            cons = self._active_constraints
        for c in cons:
            reason = Reason.no_change
            if c.expr is not self._active_constraints[c]:
                reason |= Reason.expr
            if reason:
                self._updates.cons_to_update[c] |= reason
                self._update_con(c)

    def update_constraints(self, cons: Optional[Collection[ConstraintData]] = None):
        self._update_constraints(cons)
        self._updates.run()

    def _update_sos_con(self, con: SOSConstraintData):
        sos_items = list(con.get_items())
        self._active_sos[con] = ([i[0] for i in sos_items], [i[1] for i in sos_items])
        variables = ComponentSet()
        parameters = ComponentSet()
        for v, p in sos_items:
            variables.add(v)
            if type(p) in native_numeric_types:
                continue
            if p.is_parameter_type():
                parameters.add(p)

        _variables = self._vars_referenced_by_con[con]
        _parameters = self._params_referenced_by_con[con]
        new_vars = variables - _variables
        old_vars = _variables - variables
        new_params = parameters - _parameters
        old_params = _parameters - parameters

        self._vars_referenced_by_con[con] = variables
        self._params_referenced_by_con[con] = parameters

        if new_vars:
            self._check_for_new_vars(new_vars)
        if new_params:
            self._check_for_new_params(new_params)

        for v in new_vars:
            self._referenced_variables[v][1].add(con)
        for v in old_vars:
            self._referenced_variables[v][1].remove(con)
        for p in new_params:
            self._referenced_params[p][1].add(con)
        for p in old_params:
            self._referenced_params[p][1].remove(con)

        if old_vars:
            self._check_to_remove_vars(old_vars)
        if old_params:
            self._check_to_remove_params(old_params)

    def _update_sos_constraints(
        self, cons: Optional[Collection[SOSConstraintData]] = None
    ):
        if cons is None:
            cons = self._active_sos
        for c in cons:
            reason = Reason.no_change
            _vlist, _plist = self._active_sos[c]
            sos_items = list(c.get_items())
            vlist = [i[0] for i in sos_items]
            plist = [i[1] for i in sos_items]
            needs_update = False
            if len(_vlist) != len(vlist) or len(_plist) != len(plist):
                needs_update = True
            else:
                for v1, v2 in zip(_vlist, vlist):
                    if v1 is not v2:
                        needs_update = True
                        break
                for p1, p2 in zip(_plist, plist):
                    if p1 is not p2:
                        needs_update = True
                        break
            if needs_update:
                reason |= Reason.sos_items
                self._updates.sos_to_update[c] |= reason
                self._update_sos_con(c)

    def update_sos_constraints(
        self, cons: Optional[Collection[SOSConstraintData]] = None
    ):
        self._update_sos_constraints(cons)
        self._updates.run()

    def _update_obj_expr(self, obj: ObjectiveData):
        (named_exprs, variables, parameters, external_functions) = (
            collect_components_from_expr(obj.expr)
        )
        if named_exprs:
            self._obj_named_expressions[obj] = [(e, e.expr) for e in named_exprs]
        else:
            self._obj_named_expressions.pop(obj, None)
        if external_functions:
            self._external_functions[obj] = external_functions
        else:
            self._external_functions.pop(obj, None)

        _variables = self._vars_referenced_by_obj[obj]
        _parameters = self._params_referenced_by_obj[obj]
        new_vars = variables - _variables
        old_vars = _variables - variables
        new_params = parameters - _parameters
        old_params = _parameters - parameters

        self._vars_referenced_by_obj[obj] = variables
        self._params_referenced_by_obj[obj] = parameters

        if new_vars:
            self._check_for_new_vars(new_vars)
        if new_params:
            self._check_for_new_params(new_params)

        for v in new_vars:
            self._referenced_variables[v][2].add(obj)
        for v in old_vars:
            self._referenced_variables[v][2].remove(obj)
        for p in new_params:
            self._referenced_params[p][2].add(obj)
        for p in old_params:
            self._referenced_params[p][2].remove(obj)

        if old_vars:
            self._check_to_remove_vars(old_vars)
        if old_params:
            self._check_to_remove_params(old_params)

    def _update_objectives(self, objs: Optional[Collection[ObjectiveData]] = None):
        if objs is None:
            objs = self._objectives
        for obj in objs:
            reason = Reason.no_change
            _expr, _sense = self._objectives[obj]
            if _expr is not obj.expr:
                reason |= Reason.expr
            if _sense != obj.sense:
                reason |= Reason.sense
            if reason:
                self._updates.objs_to_update[obj] |= reason
                self._objectives[obj] = (obj.expr, obj.sense)
                if reason & Reason.expr:
                    self._update_obj_expr(obj)

    def update_objectives(self, objs: Optional[Collection[ObjectiveData]] = None):
        self._update_objectives(objs)
        self._updates.run()

    def _check_for_new_or_removed_sos(self):
        new_sos = []
        old_sos = []
        current_sos_set = OrderedSet(
            self._model.component_data_objects(
                SOSConstraint, descend_into=True, active=True
            )
        )
        for c in current_sos_set:
            if c not in self._active_sos:
                new_sos.append(c)
        for c in self._active_sos:
            if c not in current_sos_set:
                old_sos.append(c)
        return new_sos, old_sos

    def _check_for_new_or_removed_constraints(self):
        new_cons = []
        old_cons = []
        current_cons_set = OrderedSet(
            self._model.component_data_objects(
                Constraint, descend_into=True, active=True
            )
        )
        for c in current_cons_set:
            if c not in self._active_constraints:
                new_cons.append(c)
        for c in self._active_constraints:
            if c not in current_cons_set:
                old_cons.append(c)
        return new_cons, old_cons

    def _check_for_named_expression_changes(self):
        for con, ne_list in self._named_expressions.items():
            for named_expr, old_expr in ne_list:
                if named_expr.expr is not old_expr:
                    self._updates.cons_to_update[con] |= Reason.expr
                    self._update_con(con)
                    break
        for obj, ne_list in self._obj_named_expressions.items():
            for named_expr, old_expr in ne_list:
                if named_expr.expr is not old_expr:
                    self._updates.objs_to_update[obj] |= Reason.expr
                    self._update_obj_expr(obj)
                    break

    def _check_for_new_or_removed_objectives(self):
        new_objs = []
        old_objs = []
        current_objs_set = ComponentSet(
            self._model.component_data_objects(
                Objective, descend_into=True, active=True
            )
        )
        for obj in current_objs_set:
            if obj not in self._objectives:
                new_objs.append(obj)
        for obj in self._objectives.keys():
            if obj not in current_objs_set:
                old_objs.append(obj)
        return new_objs, old_objs

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

            if config.check_for_new_or_removed_constraints:
                new_cons, old_cons = self._check_for_new_or_removed_constraints()
                new_sos, old_sos = self._check_for_new_or_removed_sos()
            else:
                new_cons = []
                old_cons = []
                new_sos = []
                old_sos = []

            if config.check_for_new_or_removed_objectives:
                new_objs, old_objs = self._check_for_new_or_removed_objectives()
            else:
                new_objs = []
                old_objs = []

            if new_cons:
                self._add_constraints(new_cons)
            if new_sos:
                self._add_sos_constraints(new_sos)
            if new_objs:
                self._add_objectives(new_objs)

            if old_cons:
                self._remove_constraints(old_cons)
            if old_sos:
                self._remove_sos_constraints(old_sos)
            if old_objs:
                self._remove_objectives(old_objs)

            if config.update_constraints:
                self._update_constraints()
                self._update_sos_constraints()
            if config.update_objectives:
                self._update_objectives()

            if config.update_named_expressions:
                self._check_for_named_expression_changes()

            if config.update_vars:
                self._update_variables()

            if config.update_parameters:
                self._update_parameters()

            self._updates.run()

    def get_variables_impacted_by_param(self, p: ParamData):
        return list(self._referenced_params[p][3])

    def get_constraints_impacted_by_param(self, p: ParamData):
        return list(self._referenced_params[p][0])

    def get_constraints_impacted_by_var(self, v: VarData):
        return list(self._referenced_variables[v][0])

    def get_objectives_impacted_by_param(self, p: ParamData):
        return list(self._referenced_params[p][2])

    def get_objectives_impacted_by_var(self, v: VarData):
        return list(self._referenced_variables[v][2])
