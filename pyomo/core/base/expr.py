#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from __future__ import division

__all__ = ( 'log', 'log10', 'sin', 'cos', 'tan', 'cosh', 'sinh', 'tanh',
            'asin', 'acos', 'atan', 'exp', 'sqrt', 'asinh', 'acosh', 
            'atanh', 'ceil', 'floor' )

from pyomo.core.base import expr_common as common

def generate_expression(etype, _self,_other):
    raise RuntimeError("incomplete import of Pyomo expression system")
def generate_relational_expression(etype, lhs, rhs):
    raise RuntimeError("incomplete import of Pyomo expression system")
def generate_intrinsic_function_expression(etype, name, arg):
    raise RuntimeError("incomplete import of Pyomo expression system")


from pyomo.core.base import numvalue

if common.mode is common.Mode.coopr3_trees:
    from pyomo.core.base.expr_coopr3 import *
    from pyomo.core.base.expr_coopr3 import \
        _EqualityExpression, _InequalityExpression, _ExpressionBase, \
        _ProductExpression, _SumExpression, _PowExpression, _AbsExpression, \
        _IntrinsicFunctionExpression, _ExternalFunctionExpression
elif common.mode is common.Mode.pyomo4_trees:
    from pyomo.core.base.expr_pyomo4 import *
    from pyomo.core.base.expr_pyomo4 import \
        _EqualityExpression, _InequalityExpression, _ExpressionBase, \
        _ProductExpression, _SumExpression, _PowExpression, _AbsExpression, \
        _UnaryFunctionExpression as _IntrinsicFunctionExpression, \
        _ExternalFunctionExpression
else:
    raise RuntimeError("Unrecognized expression tree mode")

# FIXME: This should eventually not be needed
from pyomo.core.base.expr_common import _pow

numvalue.generate_expression = generate_expression
numvalue.generate_relational_expression = generate_relational_expression


def fabs(arg):
    # FIXME: We need to switch this over from generate_expression to
    # just use generate_intrinsic_function_expression
    #
    #return generate_intrinsic_function_expression(arg, 'fabs', math.fabs)
    return generate_expression(_abs, arg, None)

def ceil(arg):
    return generate_intrinsic_function_expression(arg, 'ceil', math.ceil)

def floor(arg):
    return generate_intrinsic_function_expression(arg, 'floor', math.floor)

# e ** x
def exp(arg):
    return generate_intrinsic_function_expression(arg, 'exp', math.exp)

def log(arg):
    return generate_intrinsic_function_expression(arg, 'log', math.log)

def log10(arg):
    return generate_intrinsic_function_expression(arg, 'log10', math.log10)

def pow(*args):
    return generate_expression(_pow, *args)

# FIXME: this is nominally the same as x ** 0.5, but follows a different
# path and produces a different NL file!
def sqrt(arg):
    return generate_intrinsic_function_expression(arg, 'sqrt', math.sqrt)
#    return generate_expression(_pow, arg, 0.5)


def sin(arg):
    return generate_intrinsic_function_expression(arg, 'sin', math.sin)

def cos(arg):
    return generate_intrinsic_function_expression(arg, 'cos', math.cos)

def tan(arg):
    return generate_intrinsic_function_expression(arg, 'tan', math.tan)

def sinh(arg):
    return generate_intrinsic_function_expression(arg, 'sinh', math.sinh)

def cosh(arg):
    return generate_intrinsic_function_expression(arg, 'cosh', math.cosh)

def tanh(arg):
    return generate_intrinsic_function_expression(arg, 'tanh', math.tanh)


def asin(arg):
    return generate_intrinsic_function_expression(arg, 'asin', math.asin)

def acos(arg):
    return generate_intrinsic_function_expression(arg, 'acos', math.acos)

def atan(arg):
    return generate_intrinsic_function_expression(arg, 'atan', math.atan)

def asinh(arg):
    return generate_intrinsic_function_expression(arg, 'asinh', math.asinh)

def acosh(arg):
    return generate_intrinsic_function_expression(arg, 'acosh', math.acosh)

def atanh(arg):
    return generate_intrinsic_function_expression(arg, 'atanh', math.atanh)
