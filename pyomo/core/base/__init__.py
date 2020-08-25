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
from pyomo.core.expr import current
from ctypes import (
    Structure, POINTER, CFUNCTYPE, cdll, byref,
    c_int, c_long, c_ulong, c_double, c_byte, c_char_p, c_void_p )

from pyomo.core.expr.numvalue import (_add, _sub, _mul, _div, _pow, _neg, _abs,
                                      _radd, _rsub, _rmul, _rdiv, _rpow, _iadd,
                                      _isub, _imul, _idiv, _ipow, _lt, _le,
                                      _eq, PyomoObject, TemplateExpressionError,
                                      _generate_sum_expression,
                                      _generate_mul_expression,
                                      _generate_relational_expression,
                                      _generate_other_expression,
                                      NonNumericValue, nonpyomo_leaf_types,
                                      native_numeric_types,
                                      native_integer_types, native_boolean_types,
                                      native_logical_types, pyomo_constant_types,
                                      RegisterNumericType, RegisterIntegerType,
                                      RegisterBooleanType, value, is_constant,
                                      is_fixed, is_variable_type,
                                      is_potentially_variable, is_numeric_data,
                                      polynomial_degree, as_numeric,
                                      check_if_numeric_type_and_cache,
                                      NumericValue, NumericConstant,
                                      ZeroConstant)
from pyomo.core.expr.boolean_value import (
    as_boolean, BooleanConstant, BooleanValue,
    native_logical_values)
from pyomo.core.kernel.objective import (minimize,
                                         maximize)
from pyomo.core.base.config import PyomoOptions

from pyomo.core.base.expression import (ConstructionTimer,
                                        IndexedComponent,
                                        UnindexedComponent_set,
                                        is_functor, _ExpressionData,
                                        _GeneralExpressionDataImpl,
                                        _GeneralExpressionData, Expression,
                                        SimpleExpression, IndexedExpression)
from pyomo.core.base.label import (_CharMapper, cpxlp_label_from_name,
                                   alphanum_label_from_name, CuidLabeler,
                                   CounterLabeler, NumericLabeler,
                                   CNameLabeler, TextLabeler,
                                   AlphaNumericTextLabeler, NameLabeler,
                                   ShortNameLabeler)

#
# Components
#
from pyomo.core.base.component import (_name_index_generator,
                                       name, cname, CloneError, _ComponentBase,
                                       Component, ActiveComponent,
                                       ComponentData, ActiveComponentData,
                                       ComponentUID)
import pyomo.core.base.indexed_component
from pyomo.core.base.action import BuildAction
from pyomo.core.base.check import BuildCheck
from pyomo.core.base.set import (
    Set, SetOf, simple_set_rule, RangeSet,
)
from pyomo.core.base.param import (NoArgumentGiven,
                                   _raise_modifying_immutable_error,
                                   _ImplicitAny, _NotValid, _ParamData,
                                   Param, SimpleParam, IndexedParam)
from pyomo.core.base.var import (_VarData, _GeneralVarData, Var,
                                 SimpleVar, IndexedVar, VarList)
from pyomo.core.base.boolean_var import (
    BooleanVar, _BooleanVarData, _GeneralBooleanVarData,
    BooleanVarList, SimpleBooleanVar)
from pyomo.core.base.constraint import (logical_expr,
                                        _simple_constraint_rule_types,
                                        _rule_returned_none_error,
                                        simple_constraint_rule,
                                        simple_constraintlist_rule,
                                        _ConstraintData,
                                        ConstraintList
                                        _GeneralConstraintData, Constraint,
                                        SimpleConstraint, IndexedConstraint)
from pyomo.core.base.logical_constraint import (
    LogicalConstraint, LogicalConstraintList, _LogicalConstraintData)
from pyomo.core.base.objective import (simple_objective_rule,
                                       simple_objectivelist_rule,
                                       _ObjectiveData, _GeneralObjectiveData,
                                       Objective, SimpleObjective,
                                       IndexedObjective, ObjectiveList)
from pyomo.core.base.connector import (_ConnectorData,
                                       Connector, SimpleConnector,
                                       IndexedConnector, ConnectorExpander,
                                       transform)
from pyomo.core.base.sos import (_SOSConstraintData, SOSConstraint,
                                 SimpleSOSConstraint, IndexedSOSConstraint)
from pyomo.core.base.piecewise import (PWRepn, Bound, _isNonDecreasing,
                                       _isNonIncreasing, _isPowerOfTwo,
                                       _GrayCode, _characterize_function,
                                       _PiecewiseData, _SimpleSinglePiecewise,
                                       _SimplifiedPiecewise, _SOS2Piecewise,
                                       _DCCPiecewise, _DLOGPiecewise,
                                       _CCPiecewise, _LOGPiecewise,
                                       _MCPiecewise, _INCPiecewise,
                                       _BIGMPiecewise, Piecewise,
                                       SimplePiecewise, IndexedPiecewise)
from pyomo.core.base.suffix import (active_export_suffix_generator,
                                    export_suffix_generator,
                                    active_import_suffix_generator,
                                    import_suffix_generator,
                                    active_local_suffix_generator,
                                    local_suffix_generator,
                                    active_suffix_generator,
                                    suffix_generator, Suffix)
from pyomo.core.base.external import (ExternalFunction, AMPLExternalFunction,
                                      PythonCallbackFunction, _ARGLIST,
                                      _AMPLEXPORTS)
from pyomo.core.base.symbol_map import symbol_map_from_instance
from pyomo.core.base.reference import Reference

from pyomo.core.base.set import (Reals, PositiveReals, NonPositiveReals,
                                 NegativeReals, NonNegativeReals, Integers,
                                 PositiveIntegers, NonPositiveIntegers,
                                 NegativeIntegers, NonNegativeIntegers,
                                 Boolean, Binary, Any, AnyWithNone, EmptySet,
                                 UnitInterval, PercentFraction, RealInterval,
                                 IntegerInterval)
from pyomo.core.base.misc import (display, create_name, apply_indexed_rule,
                                  apply_parameterized_indexed_rule,
                                  _robust_sort_keyfcn, sorted_robust,
                                  _to_ustr, tabular_writer)
from pyomo.core.base.block import (ProblemFormat, guess_format, WriterFactory,
                                   _generic_component_decorator,
                                   _component_decorator, SubclassOf,
                                   SortComponents, TraversalStrategy,
                                   _sortingLevelWalker, _levelWalker,
                                   _BlockConstruction, PseudoMap, _BlockData,
                                   Block, SimpleBlock, IndexedBlock,
                                   generate_cuid_names, active_components,
                                   components, active_components_data,
                                   components_data, _IndexedCustomBlockMeta,
                                   _ScalarCustomBlockMeta, CustomBlock,
                                   declare_custom_block)
from pyomo.core.base.PyomoModel import (SolverResults, Solution, SolverStatus,
                                        UndefinedData, id_func, global_option,
                                        PyomoConfig, ModelSolution,
                                        ModelSolutions, Model, ConcreteModel,
                                        AbstractModel)
from pyomo.core.base.plugin import (pyomo_callback, registered_callback,
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
                                    TransformationFactory,
                                    apply_transformation)
#
import pyomo.core.base._pyomo
#
import pyomo.core.base.util

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
