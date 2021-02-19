#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.collections import ComponentMap
from pyomo.core.base.constraint import Constraint, _ConstraintData
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.expression import Expression, _ExpressionData
from pyomo.core.base.param import Param, _ParamData
from pyomo.core.base.objective import Objective, _ObjectiveData
from pyomo.core.base.block import Block, _BlockData

logger = logging.getLogger('pyomo.pysp')

def locate_annotations(model, annotation_type, max_allowed=None):
    """
    Return a list of all annotations of a given type on a Pyomo model.
    Setting the 'max_allowed' keyword will cause an exception to
    be raised if more than that many declarations of than annotation
    type are found on the model.
    """

    annotations = [(name, obj)
                   for block in model.block_data_objects(active=True,
                                                         descend_into=True)
                   for name, obj in vars(block).items()
                   if isinstance(obj, annotation_type)]
    if (max_allowed is not None) and (len(annotations) > max_allowed):
        raise ValueError("Too many annotations of type %s found on "
                         "model %s. The maximum allowed is %s."
                         % (annotation_type.__name__,
                            model.name,
                            max_allowed))
    return annotations

class PySP_Annotation(object):
    _ctypes = ()
    _ctypes_data = ()
    def __init__(self):
        assert len(self._ctypes) > 0
        assert len(self._ctypes) == len(self._ctypes_data)
        self._data = ComponentMap()
        self._default = None

    @property
    def default(self):
        return self._default

    @property
    def has_declarations(self):
        return bool(len(self._data) > 0)

    def declare(self, component, *args, **kwds):
        if isinstance(component, self._ctypes_data) or \
           isinstance(component, self._ctypes) or \
           isinstance(component, (Block, _BlockData)):
           self._declare_impl(component, *args, **kwds)
        else:
            raise TypeError(
                "Declarations in annotation type %s must be of types "
                "%s or Block. Invalid type: %s"
                % (self.__class__.__name__,
                   (",".join(ctype.__name__ for ctype in self._ctypes)),
                   type(component)))

    def expand_entries(self, expand_containers=True):
        """
        Translates the annotation into a flattened list of
        (component, annotation_value) pairs. The ctypes argument
        can be a single component type are a tuple of component
        types. If any components are found in the annotation
        not matching those types, an exception will be
        raised. If 'expand_containers' is set to False, then
        component containers will not be flattened into the set
        of components they contain.
        """
        items = []
        component_ids = set()
        def _append(component, val):
            items.append((component, val))
            if id(component) in component_ids:
                raise RuntimeError(
                    "Component %s was assigned multiple declarations "
                    "in annotation type %s. To correct this issue, ensure that "
                    "multiple container components under which the component might "
                    "be stored (such as a Block and an indexed Constraint) are not "
                    "simultaneously set in this annotation." % (component.name,
                                                                self.__class__.__name__))
            component_ids.add(id(component))

        for component in self._data:
            component_annotation_value = self._data[component]
            if not getattr(component, "active", True):
                continue
            if isinstance(component, self._ctypes_data):
                _append(component, component_annotation_value)
            elif isinstance(component, self._ctypes):
                if expand_containers:
                    for index in component:
                        obj = component[index]
                        if getattr(obj, "active", True):
                            _append(obj, component_annotation_value)
                else:
                    _append(component, component_annotation_value)
            else:
                if isinstance(component, _BlockData):
                    block_list = (component,)
                else:
                    isinstance(component, Block)
                    block_list = (component[index] for index in component)
                for block in block_list:
                    if block.active:
                        for ctype in self._ctypes:
                            if expand_containers:
                                for obj in block.component_data_objects(
                                        ctype,
                                        active=True,
                                        descend_into=True):
                                    _append(obj, component_annotation_value)
                            else:
                                for obj in block.component_objects(
                                        ctype,
                                        active=True,
                                        descend_into=True):
                                    _append(obj, component_annotation_value)

        return items

def PySP_StageCostAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'PySP_StageCostAnnotation' "
                   "has been renamed to 'StageCostAnnotation'. "
                   "Please update your model file.")
    return StageCostAnnotation(*args, **kwds)
class StageCostAnnotation(PySP_Annotation):
    """
    This annotation is used to identify the component
    representing the objective cost associated with
    a time stage.

    The declare method should be called with an additional
    argument that is an integer greater than or equal to
    one, which signifies the time-stage of the initial
    component argument.
    """
    _ctypes = (Var, Expression)
    _ctypes_data = (_VarData, _ExpressionData)

    def __init__(self):
        super(StageCostAnnotation, self).__init__()
        self._default = None

    def _declare_impl(self, component, stage):
        assert int(stage) == stage
        assert stage >= 1
        self._data[component] = stage

def PySP_VariableStageAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'PySP_VariableStageAnnotation' "
                   "has been renamed to 'VariableStageAnnotation'. "
                   "Please update your model file.")
    return VariableStageAnnotation(*args, **kwds)
class VariableStageAnnotation(PySP_Annotation):
    """
    This annotation is used to identify what time-stage a
    variable belongs to.

    The declare method should be called with an additional
    argument that is an integer greater than or equal to
    one, which signifies the time-stage of the initial
    component argument. The optional keyword 'derived' can
    be set to true to imply that a variable belongs to a
    particular time stage, but does not require
    non-anticipativity constraints be added when that time
    stage is not the last.
    """
    _ctypes = (Var, Expression)
    _ctypes_data = (_VarData, _ExpressionData)

    def __init__(self):
        super(VariableStageAnnotation, self).__init__()
        self._default = None

    def _declare_impl(self, component, stage, derived=False):
        assert int(stage) == stage
        assert stage >= 1
        self._data[component] = (stage, derived)

def PySP_ConstraintStageAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'PySP_ConstraintStageAnnotation' "
                   "is no longer recognized (constraint stages are "
                   "automatically inferred). It will be removed in "
                   "the future. Please update your model file.")
    return _ConstraintStageAnnotation(*args, **kwds)
def ConstraintStageAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'ConstraintStageAnnotation' "
                   "is no longer recognized (constraint stages are "
                   "automatically inferred). It will be removed in "
                   "the future. Please update your model file.")
    return _ConstraintStageAnnotation(*args, **kwds)
class _ConstraintStageAnnotation(PySP_Annotation):
    """
    This annotation is used to identify what time-stage a
    constraint belongs to.

    The declare method should be called with an additional
    argument that is an integer greater than or equal to
    one, which signifies the time-stage of the initial
    component argument.
    """
    _ctypes = (Constraint,)
    _ctypes_data = (_ConstraintData,)

    def __init__(self):
        super(_ConstraintStageAnnotation, self).__init__()
        self._default = None

    def _declare_impl(self, component, stage):
        assert int(stage) == stage
        assert stage >= 1
        self._data[component] = stage

def PySP_StochasticDataAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'PySP_StochasticDataAnnotation' "
                   "has been renamed to 'StochasticDataAnnotation'. "
                   "Please update your model file.")
    return StochasticDataAnnotation(*args, **kwds)
class StochasticDataAnnotation(PySP_Annotation):
    """
    This annotation is used to identify stochastic data
    locations in constraints or the objective.

    When calling declare, if the second argument is set
    to None, this implies that the constant part of the
    expression (e.g., the constant in the objective or the
    or rhs of a constraint) should be marked as stochastic.
    The optional keyword 'distribution' can be set to
    a data distribution.
    """
    _ctypes = (Param,)
    _ctypes_data = (_ParamData,)

    def __init__(self):
        super(StochasticDataAnnotation, self).__init__()
        self._default = None

    def _declare_impl(self, component, distribution=None):
        self._data[component] = distribution

def PySP_StochasticRHSAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'PySP_StochasticRHSAnnotation' "
                   "has been renamed to 'StochasticConstraintBoundsAnnotation'. "
                   "Please update your model file.")
    return StochasticConstraintBoundsAnnotation(*args, **kwds)
class StochasticConstraintBoundsAnnotation(PySP_Annotation):
    """
    This annotation is used to identify constraints that have
    stochastic bound data.

    When calling declare, at most one of the keywords 'lb' or
    'ub' can be set to False to disable the annotation on
    one side of any range type constraints.
    """
    _ctypes = (Constraint,)
    _ctypes_data = (_ConstraintData,)

    def __init__(self):
        super(StochasticConstraintBoundsAnnotation, self).__init__()
        self._default = True

    def _declare_impl(self, component, lb=True, ub=True):
        assert lb or ub
        assert (lb is True) or (lb is False)
        assert (ub is True) or (ub is False)
        if lb and ub:
            self._data[component] = True
        else:
            self._data[component] = (lb, ub)

def PySP_StochasticMatrixAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'PySP_StochasticMatrixAnnotation' "
                   "has been renamed to 'StochasticConstraintBodyAnnotation'. "
                   "Please update your model file.")
    return StochasticConstraintBodyAnnotation(*args, **kwds)
class StochasticConstraintBodyAnnotation(PySP_Annotation):
    """
    This annotation is used to identify constraints with
    stochastic data appearing in the body expression.

    When calling declare, the 'variables' keyword can be set to
    a list of variables whose coefficients should be treated
    as stochastic. Leaving 'variables' at its default of
    None signifies that coefficients of all variables
    appearing in the expression should be considered
    stochastic.
    """
    _ctypes = (Constraint,)
    _ctypes_data = (_ConstraintData,)

    def __init__(self):
        super(StochasticConstraintBodyAnnotation, self).__init__()
        self._default = None

    def _declare_impl(self, component, variables=None):
        self._data[component] = variables

def PySP_StochasticObjectiveAnnotation(*args, **kwds):
    logger.warning("DEPRECATED: 'PySP_StochasticObjectiveAnnotation' "
                   "has been renamed to 'StochasticObjectiveAnnotation'. "
                   "Please update your model file.")
    return StochasticObjectiveAnnotation(*args, **kwds)
class StochasticObjectiveAnnotation(PySP_Annotation):
    """
    This annotation is used to identify variable
    cost-coefficients that are stochastic.

    When calling declare, the 'variables' keyword can be set to
    a list of variables whose coefficients should be treated
    as stochastic. Leaving 'variables' at its default of
    None signifies that the coefficients of all variables
    appearing in the expression should be considered
    stochastic. The 'include_constant' keyword signifies
    whether or not the constant term in the cost expression
    should be treated as stochastic (default is True).
    """
    _ctypes = (Objective,)
    _ctypes_data = (_ObjectiveData,)

    def __init__(self):
        super(StochasticObjectiveAnnotation, self).__init__()
        self._default = (None, True)

    def _declare_impl(self, component, variables=None, include_constant=True):
        self._data[component] = (variables, include_constant)

class StochasticVariableBoundsAnnotation(PySP_Annotation):
    """
    This annotation is used to identify variable
    bounds that are stochastic.

    When calling declare, at most one of the keywords 'lb' or
    'ub' can be set to False to disable the annotation for
    one of the variable bounds.
    """
    _ctypes = (Var,)
    _ctypes_data = (_VarData,)

    def __init__(self):
        super(StochasticVariableBoundsAnnotation, self).__init__()
        self._default = (True, True)

    def _declare_impl(self, component,  lb=True, ub=True):
        assert lb or ub
        assert (lb is True) or (lb is False)
        assert (ub is True) or (ub is False)
        self._data[component] = (lb, ub)
