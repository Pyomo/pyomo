#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import iteritems, iterkeys
from pyomo.core.expr.numvalue import (
    value, is_constant, is_fixed, is_variable_type,
    is_potentially_variable, NumericValue, ZeroConstant,
    native_numeric_types, native_types, polynomial_degree,
)

from pyomo.core.expr.boolean_value import BooleanValue

from pyomo.core.expr.numeric_expr import linear_expression, nonlinear_expression
from pyomo.core.expr.logical_expr import (land, lor, equivalent, exactly,
                                          atleast, atmost, implies, lnot,
                                          xor, inequality)

from pyomo.core.expr.current import (
    log, log10, sin, cos, tan, cosh, sinh, tanh,
    asin, acos, atan, exp, sqrt, asinh, acosh,
    atanh, ceil, floor,
    Expr_if,
)

from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion

import pyomo.core.kernel
import pyomo.core.base._pyomo

from pyomo.common.collections import ComponentMap
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.expr import (numvalue, numeric_expr, boolean_value,
                             logical_expr, current, symbol_map, sympy_tools, 
                             taylor_series, visitor, expr_common, expr_errors,
                             calculus)
from pyomo.core import expr, preprocess, util, kernel

from pyomo.core.expr.numvalue import (nonpyomo_leaf_types,
                                      PyomoObject,
                                      native_numeric_types,
                                      value, is_constant,
                                      is_fixed, is_variable_type,
                                      is_potentially_variable,
                                      polynomial_degree,
                                      NumericValue,
                                      ZeroConstant)
from pyomo.core.expr.boolean_value import (
    as_boolean, BooleanConstant, BooleanValue,
    native_logical_values)
from pyomo.core.kernel.objective import (minimize,
                                         maximize)
from pyomo.core.base.config import PyomoOptions

from pyomo.core.base.expression import Expression
from pyomo.core.base.label import (CuidLabeler,
                                   CounterLabeler, NumericLabeler,
                                   CNameLabeler, TextLabeler,
                                   AlphaNumericTextLabeler, NameLabeler,
                                   ShortNameLabeler)

#
# Components
#
from pyomo.core.base.component import (name, Component)
from pyomo.core.base.componentuid import ComponentUID
import pyomo.core.base.indexed_component
from pyomo.core.base.action import BuildAction
from pyomo.core.base.check import BuildCheck
from pyomo.core.base.set import (
    Set, SetOf, simple_set_rule, RangeSet,
)
from pyomo.core.base.param import Param
from pyomo.core.base.var import (Var, SimpleVar, VarList)
from pyomo.core.base.boolean_var import (
    BooleanVar, BooleanVarList, SimpleBooleanVar)
from pyomo.core.base.constraint import (logical_expr,
                                        simple_constraint_rule,
                                        simple_constraintlist_rule,
                                        ConstraintList, Constraint)
from pyomo.core.base.logical_constraint import (
    LogicalConstraint, LogicalConstraintList)
from pyomo.core.base.objective import (simple_objective_rule,
                                       simple_objectivelist_rule,
                                       Objective, ObjectiveList)
from pyomo.core.base.connector import Connector
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.piecewise import Piecewise
from pyomo.core.base.suffix import (active_export_suffix_generator,
                                    active_import_suffix_generator,
                                    Suffix)
from pyomo.core.base.external import ExternalFunction
from pyomo.core.base.symbol_map import symbol_map_from_instance
from pyomo.core.base.reference import Reference

from pyomo.core.base.set import (Reals, PositiveReals, NonPositiveReals,
                                 NegativeReals, NonNegativeReals, Integers,
                                 PositiveIntegers, NonPositiveIntegers,
                                 NegativeIntegers, NonNegativeIntegers,
                                 Boolean, Binary, Any, AnyWithNone, EmptySet,
                                 UnitInterval, PercentFraction, RealInterval,
                                 IntegerInterval)
from pyomo.core.base.misc import display
from pyomo.core.base.block import (SortComponents, TraversalStrategy,
                                   Block, SimpleBlock,
                                   active_components,
                                   components, active_components_data,
                                   components_data)
from pyomo.core.base.PyomoModel import (global_option,
                                        Model, ConcreteModel,
                                        AbstractModel)
from pyomo.core.base.plugin import (pyomo_callback,
                                    IPyomoExpression, ExpressionFactory,
                                    ExpressionRegistration, IPyomoPresolver,
                                    IPyomoPresolveAction,
                                    IParamRepresentation,
                                    ParamRepresentationFactory,
                                    IPyomoScriptPreprocess,
                                    IPyomoScriptCreateModel,
                                    IPyomoScriptCreateDataPortal,
                                    IPyomoScriptModifyInstance,
                                    IPyomoScriptPrintModel,
                                    IPyomoScriptPrintInstance,
                                    IPyomoScriptSaveInstance,
                                    IPyomoScriptPrintResults,
                                    IPyomoScriptSaveResults,
                                    IPyomoScriptPostprocess,
                                    ModelComponentFactory, Transformation,
                                    TransformationFactory)
#
import pyomo.core.base._pyomo
#
from pyomo.core.base import util

from pyomo.core.base.instance2dat import instance2dat

# These APIs are deprecated and should be removed in the near future
from pyomo.core.base.set import (
    set_options, RealSet, IntegerSet, BooleanSet,
)

import pyomo.core.preprocess

from pyomo.core.util import (prod, quicksum, sum_product, dot_product,
                             summation, sequence)

from weakref import ref as weakref_ref
