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
from pyomo.common.collections import ComponentMap

from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.symbol_map import symbol_map_from_instance

from pyomo.core.expr.boolean_value import (
    as_boolean, BooleanConstant, BooleanValue,
    native_logical_values)
from pyomo.core.kernel.objective import (minimize,
                                         maximize)
from pyomo.core.base.config import PyomoOptions

from pyomo.core.base.expression import Expression, SimpleExpression,\
    IndexedExpression, _ExpressionData
from pyomo.core.base.label import (alphanum_label_from_name, CuidLabeler,
                                   CounterLabeler, NumericLabeler,
                                   CNameLabeler, TextLabeler,
                                   AlphaNumericTextLabeler, NameLabeler,
                                   ShortNameLabeler)

#
# Components
#
from pyomo.core.base.component import (name, cname, CloneError, Component,
                                       ActiveComponent, ComponentData,
                                       ActiveComponentData, ComponentUID)
import pyomo.core.base.indexed_component
from pyomo.core.base.action import BuildAction
from pyomo.core.base.check import BuildCheck
from pyomo.core.base.set import (
    Set, SetOf, simple_set_rule, RangeSet,
)
from pyomo.core.base.param import Param, SimpleParam, IndexedParam
from pyomo.core.base.var import Var, VarList, SimpleVar, IndexedVar, _VarData
from pyomo.core.base.boolean_var import (
    BooleanVar, _BooleanVarData, _GeneralBooleanVarData,
    BooleanVarList, SimpleBooleanVar)
from pyomo.core.base.constraint import (simple_constraint_rule,
                                        simple_constraintlist_rule, Constraint,
                                        SimpleConstraint, IndexedConstraint,
                                        ConstraintList, _ConstraintData)
from pyomo.core.base.logical_constraint import (
    LogicalConstraint, LogicalConstraintList, _LogicalConstraintData)
from pyomo.core.base.objective import (simple_objective_rule,
                                       simple_objectivelist_rule, Objective,
                                       SimpleObjective, IndexedObjective,
                                       ObjectiveList, _ObjectiveData)
from pyomo.core.base.connector import (ConnectorExpander, IndexedConnector,
                                       SimpleConnector, Connector)
from pyomo.core.base.sos import (SOSConstraint, SimpleSOSConstraint,
                                 IndexedSOSConstraint)
from pyomo.core.base.piecewise import (PWRepn, Bound, Piecewise,
                                       SimplePiecewise, IndexedPiecewise)
from pyomo.core.base.suffix import (active_export_suffix_generator,
                                    export_suffix_generator,
                                    active_import_suffix_generator,
                                    import_suffix_generator,
                                    active_local_suffix_generator,
                                    local_suffix_generator,
                                    active_suffix_generator, suffix_generator,
                                    Suffix)
from pyomo.core.base.external import (ExternalFunction, AMPLExternalFunction,
                                      PythonCallbackFunction)
from pyomo.core.base.symbol_map import symbol_map_from_instance
from pyomo.core.base.reference import Reference
#
from pyomo.core.base.set import (
    Reals, PositiveReals, NonPositiveReals, NegativeReals, NonNegativeReals,
    Integers, PositiveIntegers, NonPositiveIntegers,
    NegativeIntegers, NonNegativeIntegers,
    Boolean, Binary,
    Any, AnyWithNone, EmptySet, UnitInterval, PercentFraction,
    RealInterval, IntegerInterval,
)
from pyomo.core.base.misc import (display, create_name, apply_indexed_rule,
                                  apply_parameterized_indexed_rule,
                                  sorted_robust, tabular_writer)
from pyomo.core.base.block import (SubclassOf, SortComponents, TraversalStrategy,
                                   PseudoMap, Block, SimpleBlock, IndexedBlock,
                                   generate_cuid_names, active_components,
                                   components, active_components_data,
                                   components_data, CustomBlock,
                                   declare_custom_block, )
from pyomo.core.base.PyomoModel import (global_option, PyomoConfig,
                                        ModelSolution, ModelSolutions, Model,
                                        ConcreteModel, AbstractModel)
from pyomo.core.base.plugin import (pyomo_callback, IParamRepresentation, 
                                    IPyomoScriptPreprocess,
                                    IPyomoScriptCreateModel,
                                    IPyomoScriptModifyInstance,
                                    IPyomoScriptCreateDataPortal,
                                    IPyomoScriptPrintModel,
                                    IPyomoScriptPrintInstance,
                                    IPyomoScriptSaveInstance,
                                    IPyomoScriptPrintResults,
                                    IPyomoScriptSaveResults,
                                    IPyomoScriptPostprocess, IPyomoPresolver,
                                    IPyomoPresolveAction, IPyomoExpression,
                                    ExpressionRegistration, ExpressionFactory,
                                    ModelComponentFactory,
                                    ParamRepresentationFactory,
                                    TransformationInfo, TransformationData,
                                    Transformation, TransformationFactory, 
                                    apply_transformation)
#
import pyomo.core.base._pyomo
#


from pyomo.core.base.instance2dat import instance2dat

# These APIs are deprecated and should be removed in the near future
from pyomo.core.base.set import (
    set_options, RealSet, IntegerSet, BooleanSet,
)

#
# This is a hack to strip out modules, which shouldn't have been included in these imports
#
import types
_locals = locals()
__all__ = [__name for __name in _locals.keys() if (not __name.startswith('_') and not isinstance(_locals[__name],types.ModuleType)) or __name == '_' ]
__all__.append('pyomo')

import pyomo.core.kernel
import pyomo.core.preprocess
import pyomo.core.base._pyomo
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.util import (prod, quicksum, sum_product, sequence,
                             summation, dot_product)
from pyomo.core.expr import (numvalue, numeric_expr, boolean_value, logical_expr,
                             current, value, is_constant, is_fixed,
                             is_variable_type, nonpyomo_leaf_types,
                             is_potentially_variable, NumericValue, ZeroConstant,
                             native_numeric_types, native_types,
                             polynomial_degree, BooleanValue,
                             linear_expression, nonlinear_expression, inequality,
                             land, lor, equivalent, exactly, atleast, atmost,
                             implies, lnot, xor, log, log10, sin, cos, tan, cosh,
                             sinh, tanh, asin, acos, atan, exp, sqrt, asinh,
                             acosh, atanh, ceil, floor, Expr_if, differentiate, 
                             taylor_series_expansion)
