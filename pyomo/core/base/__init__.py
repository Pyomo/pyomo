#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# TODO: this import is for historical backwards compatibility and should
# probably be removed
from pyomo.common.collections import ComponentMap
from pyomo.common.enums import minimize, maximize

from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.expr.numvalue import (
    nonpyomo_leaf_types,
    native_types,
    native_numeric_types,
    value,
    is_constant,
    is_fixed,
    is_variable_type,
    is_potentially_variable,
    polynomial_degree,
    NumericValue,
    ZeroConstant,
)
from pyomo.core.expr.boolean_value import (
    as_boolean,
    BooleanConstant,
    BooleanValue,
    native_logical_values,
)

from pyomo.core.base.component import name, Component, ModelComponentFactory
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.label import (
    CuidLabeler,
    CounterLabeler,
    NumericLabeler,
    CNameLabeler,
    TextLabeler,
    AlphaNumericTextLabeler,
    NameLabeler,
    ShortNameLabeler,
)
from pyomo.core.base.misc import display
from pyomo.core.base.reference import Reference
from pyomo.core.base.symbol_map import symbol_map_from_instance
from pyomo.core.base.transformation import (
    Transformation,
    TransformationFactory,
    ReverseTransformationToken,
)

from pyomo.core.base.PyomoModel import (
    global_option,
    ModelSolution,
    ModelSolutions,
    Model,
    ConcreteModel,
    AbstractModel,
)

#
# Components
#
from pyomo.core.base.action import BuildAction
from pyomo.core.base.block import (
    Block,
    BlockData,
    ScalarBlock,
    active_components,
    components,
    active_components_data,
    components_data,
)
from pyomo.core.base.boolean_var import (
    BooleanVar,
    BooleanVarData,
    BooleanVarList,
    ScalarBooleanVar,
)
from pyomo.core.base.check import BuildCheck
from pyomo.core.base.connector import Connector, ConnectorData
from pyomo.core.base.constraint import (
    simple_constraint_rule,
    simple_constraintlist_rule,
    ConstraintList,
    Constraint,
    ConstraintData,
)
from pyomo.core.base.expression import Expression, NamedExpressionData, ExpressionData
from pyomo.core.base.external import ExternalFunction
from pyomo.core.base.logical_constraint import (
    LogicalConstraint,
    LogicalConstraintList,
    LogicalConstraintData,
)
from pyomo.core.base.objective import (
    simple_objective_rule,
    simple_objectivelist_rule,
    Objective,
    ObjectiveList,
    ObjectiveData,
)
from pyomo.core.base.param import Param, ParamData
from pyomo.core.base.piecewise import Piecewise, PiecewiseData
from pyomo.core.base.set import (
    Set,
    SetData,
    SetOf,
    RangeSet,
    Reals,
    PositiveReals,
    NonPositiveReals,
    NegativeReals,
    NonNegativeReals,
    Integers,
    PositiveIntegers,
    NonPositiveIntegers,
    NegativeIntegers,
    NonNegativeIntegers,
    Boolean,
    Binary,
    Any,
    AnyWithNone,
    EmptySet,
    UnitInterval,
    PercentFraction,
    RealInterval,
    IntegerInterval,
    simple_set_rule,
)
from pyomo.core.base.sos import SOSConstraint, SOSConstraintData
from pyomo.core.base.suffix import (
    active_export_suffix_generator,
    active_import_suffix_generator,
    Suffix,
)
from pyomo.core.base.var import Var, VarData, ScalarVar, VarList

from pyomo.core.base.instance2dat import instance2dat

#
# These APIs are deprecated and should be removed in the near future
#
from pyomo.core.base.set import set_options, RealSet, IntegerSet, BooleanSet

from pyomo.common.deprecation import relocated_module_attribute

relocated_module_attribute(
    'SimpleBlock', 'pyomo.core.base.block.SimpleBlock', version='6.0'
)
relocated_module_attribute('SimpleVar', 'pyomo.core.base.var.SimpleVar', version='6.0')
relocated_module_attribute(
    'SimpleBooleanVar', 'pyomo.core.base.boolean_var.SimpleBooleanVar', version='6.0'
)
# Historically, only a subset of "private" component data classes were imported here
relocated_module_attribute(
    f'_GeneralVarData', f'pyomo.core.base.VarData', version='6.7.2'
)
relocated_module_attribute(
    f'_GeneralBooleanVarData', f'pyomo.core.base.BooleanVarData', version='6.7.2'
)
relocated_module_attribute(
    f'_ExpressionData', f'pyomo.core.base.NamedExpressionData', version='6.7.2'
)
for _cdata in (
    'ConstraintData',
    'LogicalConstraintData',
    'VarData',
    'BooleanVarData',
    'ObjectiveData',
):
    relocated_module_attribute(
        f'_{_cdata}', f'pyomo.core.base.{_cdata}', version='6.7.2'
    )
del _cdata
del relocated_module_attribute
