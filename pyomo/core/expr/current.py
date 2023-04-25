#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
import math

#
# Common intrinsic functions
#
from pyomo.core.expr import expr_common as common


#
# Provide a global value that indicates which expression system is being used
#
class Mode(enum.IntEnum):
    # coopr: Original Coopr/Pyomo expression system
    coopr_trees = 1
    # coopr3: leverage reference counts to reduce the amount of required
    # expression cloning to ensure independent expression trees.
    coopr3_trees = 3
    # pyomo4: rework the expression system to remove reliance on
    # reference counting.  This enables pypy support (which doesn't have
    # reference counting).  This version never became the default.
    pyomo4_trees = 4
    # pyomo5: refinement of pyomo4.  Expressions are now immutable by
    # contract, which tolerates "entangled" expression trees.  Added
    # specialized classes for NPV expressions and LinearExpressions.
    pyomo5_trees = 5
    # pyomo6: refinement of pyomo5 expression generation to leverage
    # multiple dispatch.  Standardized expression storage and argument
    # handling (significant rework of the LinearExpression structure).
    pyomo6_trees = 6


_mode = Mode.pyomo6_trees
# We no longer support concurrent expression systems.  _mode is left
# primarily so we can support expression system-specific baselines
assert _mode == Mode.pyomo6_trees

#
# Pull symbols from the appropriate expression system
#
from pyomo.core.expr import numvalue as _numvalue
from pyomo.core.expr import boolean_value as _logicalvalue

from pyomo.core.expr import numeric_expr as _numeric_expr
from .base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
    _add,
    _sub,
    _mul,
    _div,
    _pow,
    _neg,
    _abs,
    _inplace,
    _unary,
    NumericExpression,
    NumericValue,
    native_types,
    nonpyomo_leaf_types,
    native_numeric_types,
    as_numeric,
    value,
    evaluate_expression,
    expression_to_string,
    polynomial_degree,
    clone_expression,
    sizeof_expression,
    _expression_is_fixed,
    clone_counter,
    nonlinear_expression,
    linear_expression,
    NegationExpression,
    NPV_NegationExpression,
    ExternalFunctionExpression,
    NPV_ExternalFunctionExpression,
    PowExpression,
    NPV_PowExpression,
    ProductExpression,
    NPV_ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    NPV_DivisionExpression,
    SumExpressionBase,
    NPV_SumExpression,
    SumExpression,
    _MutableSumExpression,
    Expr_ifExpression,
    NPV_Expr_ifExpression,
    UnaryFunctionExpression,
    NPV_UnaryFunctionExpression,
    AbsExpression,
    NPV_AbsExpression,
    LinearExpression,
    _MutableLinearExpression,
    decompose_term,
    LinearDecompositionError,
    _decompose_linear_terms,
    _balanced_parens,
    NPV_expression_types,
    _fcn_dispatcher,
)
from pyomo.core.expr import logical_expr as _logical_expr
from pyomo.core.expr.logical_expr import (
    native_logical_types,
    BooleanValue,
    BooleanConstant,
    _and,
    _or,
    _equiv,
    _inv,
    _xor,
    _impl,
    _generate_logical_proposition,
    BooleanExpressionBase,
    lnot,
    equivalent,
    xor,
    implies,
    _flattened,
    land,
    lor,
    exactly,
    atmost,
    atleast,
    UnaryBooleanExpression,
    NotExpression,
    BinaryBooleanExpression,
    EquivalenceExpression,
    XorExpression,
    ImplicationExpression,
    NaryBooleanExpression,
    _add_to_and_or_expression,
    AndExpression,
    OrExpression,
    ExactlyExpression,
    AtMostExpression,
    AtLeastExpression,
    special_boolean_atom_types,
)
from pyomo.core.expr.relational_expr import (
    RelationalExpression,
    RangedExpression,
    InequalityExpression,
    EqualityExpression,
    inequality,
)
from pyomo.core.expr.template_expr import (
    TemplateExpressionError,
    _NotSpecified,
    GetItemExpression,
    Numeric_GetItemExpression,
    Boolean_GetItemExpression,
    Structural_GetItemExpression,
    NPV_Numeric_GetItemExpression,
    NPV_Boolean_GetItemExpression,
    NPV_Structural_GetItemExpression,
    GetAttrExpression,
    Numeric_GetAttrExpression,
    Boolean_GetAttrExpression,
    Structural_GetAttrExpression,
    NPV_Numeric_GetAttrExpression,
    NPV_Boolean_GetAttrExpression,
    NPV_Structural_GetAttrExpression,
    CallExpression,
    _TemplateSumExpression_argList,
    TemplateSumExpression,
    IndexTemplate,
    resolve_template,
    ReplaceTemplateExpression,
    substitute_template_expression,
    _GetItemIndexer,
    substitute_getitem_with_param,
    substitute_template_with_value,
    _set_iterator_template_generator,
    _template_iter_context,
    templatize_rule,
    templatize_constraint,
)
from pyomo.core.expr import visitor as _visitor
from pyomo.core.expr.visitor import (
    SymbolMap,
    StreamBasedExpressionVisitor,
    SimpleExpressionVisitor,
    ExpressionValueVisitor,
    replace_expressions,
    ExpressionReplacementVisitor,
    _EvaluationVisitor,
    FixedExpressionError,
    NonConstantExpressionError,
    _EvaluateConstantExpressionVisitor,
    _ComponentVisitor,
    identify_components,
    _VariableVisitor,
    identify_variables,
    _MutableParamVisitor,
    identify_mutable_parameters,
    _PolynomialDegreeVisitor,
    _IsFixedVisitor,
    _ToStringVisitor,
)


def Expr_if(IF=None, THEN=None, ELSE=None):
    """
    Function used to construct a logical conditional expression.
    """
    if _numvalue.is_constant(IF):
        return THEN if value(IF) else ELSE
    if not any(map(_numvalue.is_potentially_variable, (IF, THEN, ELSE))):
        return NPV_Expr_ifExpression((IF, THEN, ELSE))
    return Expr_ifExpression((IF, THEN, ELSE))


#
# NOTE: abs() and pow() are not defined here, because they are
# Python operators.
#
def ceil(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'ceil', math.ceil)


def floor(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'floor', math.floor)


# e ** x
def exp(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'exp', math.exp)


def log(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'log', math.log)


def log10(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'log10', math.log10)


# FIXME: this is nominally the same as x ** 0.5, but follows a different
# path and produces a different NL file!
def sqrt(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'sqrt', math.sqrt)
    # return _pow_dispatcher[arg.__class__, float](arg, 0.5)


def sin(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'sin', math.sin)


def cos(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'cos', math.cos)


def tan(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'tan', math.tan)


def sinh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'sinh', math.sinh)


def cosh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'cosh', math.cosh)


def tanh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'tanh', math.tanh)


def asin(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'asin', math.asin)


def acos(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'acos', math.acos)


def atan(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'atan', math.atan)


def asinh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'asinh', math.asinh)


def acosh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'acosh', math.acosh)


def atanh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'atanh', math.atanh)
