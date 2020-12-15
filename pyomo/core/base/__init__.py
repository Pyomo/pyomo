#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# TODO: this import is for historical backwards compatibility and should
# probably be removed

from six import iteritems, iterkeys
import pyomo.core.expr.numvalue
import pyomo.core.expr.logical_expr
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.symbol_map import SymbolMap
import pyomo.core.base.action
import pyomo.core.base.boolean_var
import pyomo.core.base.check
import pyomo.core.base.component
import pyomo.core.base.config
import pyomo.core.base.constraint
import pyomo.core.base.expression
import pyomo.core.base.global_set
import pyomo.core.base.indexed_component
import pyomo.core.base.indexed_component_slice
import pyomo.core.base.label
import pyomo.core.base.logical_constraint
import pyomo.core.base.misc
import pyomo.core.base.param
import pyomo.core.base.plugin
import pyomo.core.base.range
import pyomo.core.base.set_types
import pyomo.core.base.set
import pyomo.core.base.units_container
import pyomo.core.base.util
import pyomo.core.base.var

from pyomo.core.expr.numvalue import (nonpyomo_leaf_types,
                                      native_types,
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

from pyomo.core.base.expression import (Expression, _ExpressionData)
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
from pyomo.core.base.action import BuildAction
from pyomo.core.base.check import BuildCheck
from pyomo.core.base.set import (
    Set, SetOf, simple_set_rule, RangeSet,
)
from pyomo.core.base.param import Param
from pyomo.core.base.var import (Var, _VarData, _GeneralVarData,
                                 SimpleVar, VarList)
from pyomo.core.base.boolean_var import (
    BooleanVar,  _BooleanVarData,  _GeneralBooleanVarData,
    BooleanVarList, SimpleBooleanVar)
from pyomo.core.base.constraint import (simple_constraint_rule,
                                        simple_constraintlist_rule,
                                        ConstraintList, Constraint,
                                        _ConstraintData)
from pyomo.core.base.logical_constraint import (
    LogicalConstraint, LogicalConstraintList, _LogicalConstraintData)
from pyomo.core.base.objective import (simple_objective_rule,
                                       simple_objectivelist_rule,
                                       Objective, ObjectiveList,
                                       _ObjectiveData)
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
                                   Block, SimpleBlock, active_components,
                                   components, active_components_data,
                                   components_data)
from pyomo.core.base.PyomoModel import (global_option,
                                        ModelSolution,
                                        ModelSolutions, Model, ConcreteModel,
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
import pyomo.core.base.util

from pyomo.core.base.instance2dat import instance2dat

# These APIs are deprecated and should be removed in the near future
from pyomo.core.base.set import (
    set_options, RealSet, IntegerSet, BooleanSet,
)

from weakref import ref as weakref_ref

#
# This is a hack to strip out modules, which shouldn't have been included in these imports
#
import types
_locals = locals()
__all__ = [__name for __name in _locals.keys() if (not __name.startswith('_') and not isinstance(_locals[__name],types.ModuleType)) or __name == '_' ]
__all__.append('pyomo')
