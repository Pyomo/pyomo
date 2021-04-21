#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division
import math

#
# Common intrinsic functions
#
from pyomo.core.expr import expr_common as common

#
# Provide a global value that indicates which expression system is being used
#
class Mode(object):
    pyomo5_trees = (3,)
_mode = Mode.pyomo5_trees

#
# Pull symbols from the appropriate expression system
#
from pyomo.core.expr import numvalue as _numvalue
from pyomo.core.expr import boolean_value as _logicalvalue

# Pyomo5
if _mode == Mode.pyomo5_trees:
    from pyomo.core.expr import numeric_expr as _numeric_expr
    from pyomo.core.expr.numeric_expr import (_add, _sub, _mul, _div, _pow,
                                              _neg, _abs, _inplace, _unary,
                                              NumericValue, native_types,
                                              nonpyomo_leaf_types, 
                                              native_numeric_types,
                                              as_numeric, value,
                                              evaluate_expression,
                                              expression_to_string,
                                              polynomial_degree,
                                              clone_expression,
                                              sizeof_expression,
                                              _expression_is_fixed,
                                              clone_counter,
                                              nonlinear_expression,
                                              linear_expression, ExpressionBase,
                                              NegationExpression,
                                              NPV_NegationExpression,
                                              ExternalFunctionExpression,
                                              NPV_ExternalFunctionExpression,
                                              PowExpression, NPV_PowExpression,
                                              ProductExpression,
                                              NPV_ProductExpression,
                                              MonomialTermExpression,
                                              DivisionExpression,
                                              NPV_DivisionExpression,
                                              ReciprocalExpression,
                                              NPV_ReciprocalExpression,
                                              _LinearOperatorExpression,
                                              SumExpressionBase,
                                              NPV_SumExpression, SumExpression,
                                              _MutableSumExpression,
                                              Expr_ifExpression,
                                              UnaryFunctionExpression,
                                              NPV_UnaryFunctionExpression,
                                              AbsExpression, NPV_AbsExpression,
                                              LinearExpression,
                                              _MutableLinearExpression,
                                              decompose_term,
                                              LinearDecompositionError,
                                              _decompose_linear_terms,
                                              _process_arg,
                                              _generate_sum_expression,
                                              _generate_mul_expression,
                                              _generate_other_expression,
                                              _generate_intrinsic_function_expression,
                                              _balanced_parens,
                                              NPV_expression_types)
    from pyomo.core.expr import logical_expr as _logical_expr
    from pyomo.core.expr.logical_expr import (native_logical_types, BooleanValue,
                                              BooleanConstant, _lt, _le, _eq,
                                              _and, _or, _equiv, _inv, _xor,
                                              _impl, _chainedInequality,
                                              RangedExpression,
                                              InequalityExpression, inequality,
                                              EqualityExpression,
                                              _generate_relational_expression,
                                              _generate_logical_proposition,
                                              BooleanExpressionBase, lnot,
                                              equivalent, xor, implies,
                                              _flattened, land, lor, exactly,
                                              atmost, atleast,
                                              UnaryBooleanExpression,
                                              NotExpression,
                                              BinaryBooleanExpression,
                                              EquivalenceExpression,
                                              XorExpression,
                                              ImplicationExpression,
                                              NaryBooleanExpression,
                                              _add_to_and_or_expression,
                                              AndExpression, OrExpression,
                                              ExactlyExpression,
                                              AtMostExpression,
                                              AtLeastExpression,
                                              special_boolean_atom_types)
    from pyomo.core.expr.template_expr import (TemplateExpressionError,
                                               _NotSpecified, GetItemExpression,
                                               GetAttrExpression,
                                               _TemplateSumExpression_argList,
                                               TemplateSumExpression,
                                               IndexTemplate, resolve_template,
                                               ReplaceTemplateExpression,
                                               substitute_template_expression,
                                               _GetItemIndexer,
                                               substitute_getitem_with_param,
                                               substitute_template_with_value,
                                               _set_iterator_template_generator,
                                               _template_iter_context,
                                               templatize_rule,
                                               templatize_constraint)
    from pyomo.core.expr import visitor as _visitor
    from pyomo.core.expr.visitor import (SymbolMap, StreamBasedExpressionVisitor,
                                         SimpleExpressionVisitor,
                                         ExpressionValueVisitor,
                                         replace_expressions,
                                         ExpressionReplacementVisitor,
                                         _EvaluationVisitor,
                                         FixedExpressionError,
                                         NonConstantExpressionError,
                                         _EvaluateConstantExpressionVisitor,
                                         _ComponentVisitor, identify_components,
                                         _VariableVisitor, identify_variables,
                                         _MutableParamVisitor,
                                         identify_mutable_parameters,
                                         _PolynomialDegreeVisitor,
                                         _IsFixedVisitor, _ToStringVisitor)
    # FIXME: we shouldn't need circular dependencies between modules
    _visitor.LinearExpression = _numeric_expr.LinearExpression
    _visitor.MonomialTermExpression = _numeric_expr.MonomialTermExpression
    _visitor.NPV_expression_types = _numeric_expr.NPV_expression_types
    _visitor.clone_counter = _numeric_expr.clone_counter

    # Initialize numvalue functions
    _numvalue._generate_sum_expression \
        = _numeric_expr._generate_sum_expression
    _numvalue._generate_mul_expression \
        = _numeric_expr._generate_mul_expression
    _numvalue._generate_other_expression \
        = _numeric_expr._generate_other_expression
    _numvalue._generate_relational_expression \
        = _logical_expr._generate_relational_expression

    # Initialize logicalvalue functions
    _logicalvalue._generate_logical_proposition = _logical_expr._generate_logical_proposition
else:
    raise ValueError("No other expression systems are supported in Pyomo right now.")    #pragma: no cover


def Expr_if(IF=None, THEN=None, ELSE=None):
    """
    Function used to construct a logical conditional expression.
    """
    return Expr_ifExpression(IF_=IF, THEN_=THEN, ELSE_=ELSE)

#
# NOTE: abs() and pow() are not defined here, because they are
# Python operators.
#
def ceil(arg):
    return _generate_intrinsic_function_expression(arg, 'ceil', math.ceil)

def floor(arg):
    return _generate_intrinsic_function_expression(arg, 'floor', math.floor)

# e ** x
def exp(arg):
    return _generate_intrinsic_function_expression(arg, 'exp', math.exp)

def log(arg):
    return _generate_intrinsic_function_expression(arg, 'log', math.log)

def log10(arg):
    return _generate_intrinsic_function_expression(arg, 'log10', math.log10)

# FIXME: this is nominally the same as x ** 0.5, but follows a different
# path and produces a different NL file!
def sqrt(arg):
    return _generate_intrinsic_function_expression(arg, 'sqrt', math.sqrt)
#    return _generate_expression(common._pow, arg, 0.5)


def sin(arg):
    return _generate_intrinsic_function_expression(arg, 'sin', math.sin)

def cos(arg):
    return _generate_intrinsic_function_expression(arg, 'cos', math.cos)

def tan(arg):
    return _generate_intrinsic_function_expression(arg, 'tan', math.tan)

def sinh(arg):
    return _generate_intrinsic_function_expression(arg, 'sinh', math.sinh)

def cosh(arg):
    return _generate_intrinsic_function_expression(arg, 'cosh', math.cosh)

def tanh(arg):
    return _generate_intrinsic_function_expression(arg, 'tanh', math.tanh)


def asin(arg):
    return _generate_intrinsic_function_expression(arg, 'asin', math.asin)

def acos(arg):
    return _generate_intrinsic_function_expression(arg, 'acos', math.acos)

def atan(arg):
    return _generate_intrinsic_function_expression(arg, 'atan', math.atan)

def asinh(arg):
    return _generate_intrinsic_function_expression(arg, 'asinh', math.asinh)

def acosh(arg):
    return _generate_intrinsic_function_expression(arg, 'acosh', math.acosh)

def atanh(arg):
    return _generate_intrinsic_function_expression(arg, 'atanh', math.atanh)
