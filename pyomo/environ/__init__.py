#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys as _sys

if _sys.version_info[0] >= 3:
    import importlib


    def _do_import(pkg_name):
        importlib.import_module(pkg_name)
else:
    def _do_import(pkg_name):
        __import__(pkg_name, globals(), locals(), [], -1)

#
# These packages contain plugins that need to be loaded
#
_packages = [
    'pyomo.common',
    'pyomo.core',
    'pyomo.opt',
    'pyomo.dataportal',
    'pyomo.duality',
    'pyomo.checker',
    'pyomo.repn',
    'pyomo.pysp',
    'pyomo.neos',
    'pyomo.solvers',
    'pyomo.gdp',
    'pyomo.mpec',
    'pyomo.dae',
    'pyomo.bilevel',
    'pyomo.scripting',
    'pyomo.network',
]
#
#
# These packages also contain plugins that need to be loaded, but
# we silently ignore any import errors because these
# packages are optional and/or under development.
#
_optional_packages = {
    'pyomo.contrib.example',
    'pyomo.contrib.fme',
    'pyomo.contrib.gdpbb',
    'pyomo.contrib.gdpopt',
    'pyomo.contrib.gdp_bounds',
    'pyomo.contrib.mcpp',
    'pyomo.contrib.mindtpy',
    'pyomo.contrib.multistart',
    'pyomo.contrib.petsc',
    'pyomo.contrib.preprocessing',
    'pyomo.contrib.pynumero',
    'pyomo.contrib.trustregion',
    'pyomo.contrib.community_detection',
}


def _import_packages():
    #
    # Import required packages
    #
    for _package in _packages:
        pname = _package + '.plugins'
        try:
            _do_import(pname)
        except ImportError:
            exctype, err, tb = _sys.exc_info()  # BUG?
            import traceback
            msg = "pyomo.environ failed to import %s:\nOriginal %s: %s\n" \
                  "Traceback:\n%s" \
                  % (pname, exctype.__name__, err,
                     ''.join(traceback.format_tb(tb)),)
            # clear local variables to remove circular references
            exctype = err = tb = None
            # TODO: Should this just log an error and re-raise the
            # original exception?
            raise ImportError(msg)

        pkg = _sys.modules[pname]
        pkg.load()
    #
    # Import optional packages
    #
    for _package in _optional_packages:
        pname = _package + '.plugins'
        try:
            _do_import(pname)
        except ImportError:
            continue
        pkg = _sys.modules[pname]
        pkg.load()


_import_packages()

#
# Expose the symbols from pyomo.core
#
from pyomo.dataportal import DataPortal
import pyomo.core.kernel
import pyomo.core.base._pyomo
from pyomo.common.collections import ComponentMap
import pyomo.core.base.indexed_component
import pyomo.core.base._pyomo
from six import iterkeys, iteritems
import pyomo.core.base.util
from pyomo.core import expr, base, beta, kernel, plugins, preprocess
from pyomo.core.base import util
import pyomo.core.preprocess

from pyomo.core import (numvalue, numeric_expr, boolean_value,
                             current, symbol_map, sympy_tools, 
                             taylor_series, visitor, expr_common, expr_errors,
                             calculus, native_types,
                             linear_expression, nonlinear_expression,
                             land, lor, equivalent, exactly,
                             atleast, atmost, implies, lnot,
                             xor, inequality, log, log10, sin, cos, tan, cosh,
                             sinh, tanh, asin, acos, atan, exp, sqrt, asinh, acosh,
                             atanh, ceil, floor, Expr_if, differentiate,
                             taylor_series_expansion, SymbolMap, PyomoObject,
                             nonpyomo_leaf_types, native_numeric_types,
                             value, is_constant, is_fixed, is_variable_type,
                             is_potentially_variable, polynomial_degree,
                             NumericValue, ZeroConstant, as_boolean, BooleanConstant,
                             BooleanValue, native_logical_values, minimize,
                             maximize, PyomoOptions, Expression, CuidLabeler,
                             CounterLabeler, NumericLabeler,
                             CNameLabeler, TextLabeler,
                             AlphaNumericTextLabeler, NameLabeler, ShortNameLabeler, 
                             name, Component, ComponentUID, BuildAction, 
                             BuildCheck, Set, SetOf, simple_set_rule, RangeSet,
                             Param, Var, VarList, SimpleVar, 
                             BooleanVar, BooleanVarList, SimpleBooleanVar, 
                             logical_expr, simple_constraint_rule,
                             simple_constraintlist_rule, ConstraintList,
                             Constraint, LogicalConstraint, 
                             LogicalConstraintList, simple_objective_rule,
                             simple_objectivelist_rule, Objective,
                             ObjectiveList, Connector, SOSConstraint,
                             Piecewise, active_export_suffix_generator,
                             active_import_suffix_generator, Suffix, 
                             ExternalFunction, symbol_map_from_instance, 
                             Reference, Reals, PositiveReals, NonPositiveReals,
                             NegativeReals, NonNegativeReals, Integers,
                             PositiveIntegers, NonPositiveIntegers,
                             NegativeIntegers, NonNegativeIntegers,
                             Boolean, Binary, Any, AnyWithNone, EmptySet,
                             UnitInterval, PercentFraction, RealInterval,
                             IntegerInterval, display, SortComponents,
                             TraversalStrategy, Block, SimpleBlock,
                             active_components, components, 
                             active_components_data, components_data, 
                             global_option, Model, ConcreteModel,
                             AbstractModel, pyomo_callback,
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
                             TransformationFactory, instance2dat, 
                             set_options, RealSet, IntegerSet, BooleanSet,
                             prod, quicksum, sum_product, dot_product,
                             summation, sequence)

from pyomo.opt import (
    SolverFactory, SolverManagerFactory, UnknownSolver,
    TerminationCondition, SolverStatus, check_optimal_termination,
    assert_optimal_termination
    )
from pyomo.core.base.units_container import units
from weakref import ref as weakref_ref
