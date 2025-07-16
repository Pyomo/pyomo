#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import (
    value,
    is_constant,
    is_fixed,
    is_variable_type,
    is_potentially_variable,
    NumericValue,
    ZeroConstant,
    native_numeric_types,
    native_types,
    polynomial_degree,
)

from pyomo.core.expr.boolean_value import BooleanValue

from pyomo.core.expr import (
    linear_expression,
    nonlinear_expression,
    land,
    lor,
    equivalent,
    exactly,
    atleast,
    atmost,
    all_different,
    count_if,
    implies,
    lnot,
    xor,
    inequality,
    log,
    log10,
    sin,
    cos,
    tan,
    cosh,
    sinh,
    tanh,
    asin,
    acos,
    atan,
    exp,
    sqrt,
    asinh,
    acosh,
    atanh,
    ceil,
    floor,
    Expr_if,
)

from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion

from pyomo.common.collections import ComponentMap
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.expr import (
    numvalue,
    numeric_expr,
    boolean_value,
    logical_expr,
    symbol_map,
    sympy_tools,
    taylor_series,
    visitor,
    expr_common,
    expr_errors,
    calculus,
)

from pyomo.core import expr, util, kernel

from pyomo.core.expr.numvalue import (
    nonpyomo_leaf_types,
    PyomoObject,
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
from pyomo.core.base import minimize, maximize
from pyomo.core.base.config import PyomoOptions

from pyomo.core.base.expression import Expression
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

#
# Components
#
from pyomo.core.base.component import name, Component, ModelComponentFactory
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.action import BuildAction
from pyomo.core.base.check import BuildCheck
from pyomo.core.base.set import Set, SetOf, simple_set_rule, RangeSet
from pyomo.core.base.param import Param
from pyomo.core.base.var import Var, ScalarVar, VarList
from pyomo.core.base.boolean_var import BooleanVar, BooleanVarList, ScalarBooleanVar
from pyomo.core.base.constraint import (
    simple_constraint_rule,
    simple_constraintlist_rule,
    ConstraintList,
    Constraint,
)
from pyomo.core.base.logical_constraint import LogicalConstraint, LogicalConstraintList
from pyomo.core.base.objective import (
    simple_objective_rule,
    simple_objectivelist_rule,
    Objective,
    ObjectiveList,
)
from pyomo.core.base.connector import Connector
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.piecewise import Piecewise
from pyomo.core.base.suffix import (
    active_export_suffix_generator,
    active_import_suffix_generator,
    Suffix,
)
from pyomo.core.base.external import ExternalFunction
from pyomo.core.base.symbol_map import symbol_map_from_instance
from pyomo.core.base.reference import Reference

from pyomo.core.base.set import (
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
)
from pyomo.core.base.misc import display
from pyomo.core.base.block import (
    SortComponents,
    TraversalStrategy,
    Block,
    ScalarBlock,
    active_components,
    components,
    active_components_data,
    components_data,
)
from pyomo.core.base.PyomoModel import (
    global_option,
    Model,
    ConcreteModel,
    AbstractModel,
)
from pyomo.core.base.transformation import (
    Transformation,
    TransformationFactory,
    ReverseTransformationToken,
)

from pyomo.core.base.instance2dat import instance2dat

from pyomo.core.util import (
    prod,
    quicksum,
    sum_product,
    dot_product,
    summation,
    sequence,
)

# These APIs are deprecated and should be removed in the near future
from pyomo.core.base.set import set_options, RealSet, IntegerSet, BooleanSet

from pyomo.common.deprecation import relocated_module_attribute

relocated_module_attribute(
    'SimpleBlock', 'pyomo.core.base.block.SimpleBlock', version='6.0'
)
relocated_module_attribute('SimpleVar', 'pyomo.core.base.var.SimpleVar', version='6.0')
relocated_module_attribute(
    'SimpleBooleanVar', 'pyomo.core.base.boolean_var.SimpleBooleanVar', version='6.0'
)
del relocated_module_attribute
