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
_optional_packages = set([
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
])


def _import_packages():
    #
    # Import required packages
    #
    for name in _packages:
        pname = name+'.plugins'
        try:
            _do_import(pname)
        except ImportError:
            exctype, err, tb = _sys.exc_info()  # BUG?
            import traceback
            msg = "pyomo.environ failed to import %s:\nOriginal %s: %s\n"\
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
    for name in _optional_packages:
        pname = name+'.plugins'
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
from pyomo.core import (AbstractModel, AlphaNumericTextLabeler, Any, 
                        AnyWithNone, Binary, Block, Boolean, BooleanConstant,
                        BooleanSet, BooleanValue, BooleanVar, BooleanVarList,
                        BuildAction, BuildCheck, CNameLabeler, Component,
                        ComponentMap, ComponentUID, ConcreteModel, Connector,
                        Constraint, ConstraintList, CounterLabeler, CuidLabeler,
                        EmptySet, Expr_if, Expression, ExpressionFactory,
                        ExpressionRegistration, ExternalFunction,
                        IParamRepresentation, IPyomoExpression,
                        IPyomoPresolveAction, IPyomoPresolver,
                        IPyomoScriptCreateDataPortal, IPyomoScriptCreateModel,
                        IPyomoScriptModifyInstance, IPyomoScriptPostprocess,
                        IPyomoScriptPreprocess, IPyomoScriptPrintInstance,
                        IPyomoScriptPrintModel, IPyomoScriptPrintResults,
                        IPyomoScriptSaveInstance, IPyomoScriptSaveResults,
                        IntegerInterval, IntegerSet, Integers, LogicalConstraint,
                        LogicalConstraintList, Model, ModelComponentFactory,
                        NameLabeler, NegativeIntegers, NegativeReals,
                        NonNegativeIntegers, NonNegativeReals,
                        NonPositiveIntegers, NonPositiveReals, NumericLabeler,
                        NumericValue, Objective, ObjectiveList, Param,
                        ParamRepresentationFactory, PercentFraction, Piecewise,
                        PositiveIntegers, PositiveReals, PyomoOptions, RangeSet,
                        RealInterval, RealSet, Reals, Reference, SOSConstraint,
                        Set, SetOf, ShortNameLabeler, SimpleBlock,
                        SimpleBooleanVar, SimpleVar, SortComponents, Suffix,
                        SymbolMap, TextLabeler, Transformation,
                        TransformationFactory, TraversalStrategy, UnitInterval,
                        Var, VarList, ZeroConstant, acos, acosh,
                        active_components, active_components_data,
                        active_export_suffix_generator,
                        active_import_suffix_generator, as_boolean, asin, asinh,
                        atan, atanh, atleast, atmost, base, boolean_value,
                        ceil, components, components_data, cos, cosh,
                        current, differentiate, display, dot_product, equivalent,
                        exactly, exp, expr, floor,
                        global_option, implies, inequality, instance2dat,
                        is_constant, is_fixed, is_potentially_variable,
                        is_variable_type, kernel, land,
                        linear_expression, lnot, log, log10, logical_expr, lor,
                        maximize, minimize, name, native_logical_values,
                        native_numeric_types, native_types, nonlinear_expression,
                        nonpyomo_leaf_types, numeric_expr, numvalue,
                        polynomial_degree, preprocess, prod, pyomo,
                        pyomo_callback, pyomoobject, quicksum, sequence,
                        set_options, simple_constraint_rule,
                        simple_constraintlist_rule, simple_objective_rule,
                        simple_objectivelist_rule, simple_set_rule, sin, sinh,
                        sqrt, sum_product, summation,
                        symbol_map_from_instance, tan, tanh,
                        taylor_series_expansion, 
                        util, value, xor)


from pyomo.opt import (
    SolverFactory, SolverManagerFactory, UnknownSolver,
    TerminationCondition, SolverStatus, check_optimal_termination,
    assert_optimal_termination
    )
from pyomo.core.base.units_container import units
