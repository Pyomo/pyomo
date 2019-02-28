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
# Provide a global value that indicates which expression system is being used
#
class Mode(object):
    pyomo5_trees = (3,)
_mode = Mode.pyomo5_trees

#
# Common intrinsic functions
#
from pyomo.core.expr import expr_common as common
#
# Pull symbols from the appropriate expression system
#
from pyomo.core.expr import numvalue as _numvalue

# Pyomo5
if _mode == Mode.pyomo5_trees:
    from pyomo.core.expr import numeric_expr as _numeric_expr
    from pyomo.core.expr.numeric_expr import *
    from pyomo.core.expr.numeric_expr import (
        _generate_sum_expression,
        _generate_mul_expression,
        _generate_other_expression,
        _generate_intrinsic_function_expression,
    )
    from pyomo.core.expr import logical_expr as _logical_expr
    from pyomo.core.expr.logical_expr import *
    from pyomo.core.expr.logical_expr import (
        _generate_relational_expression,
        _chainedInequality,
    )
    from pyomo.core.expr import visitor as _visitor
    from pyomo.core.expr.visitor import *
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
