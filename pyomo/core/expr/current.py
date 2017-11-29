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
import copy

#
# Data and methods that are exposed when importing pyomo.core.expr
#
__public__ = ['log', 'log10', 'sin', 'cos', 'tan', 'cosh', 'sinh', 'tanh',
           'asin', 'acos', 'atan', 'exp', 'sqrt', 'asinh', 'acosh',
           'atanh', 'ceil', 'floor']

#
# Data and methods that are exposed when importing pyomo.core.expr.current
#
__all__ = copy.copy(__public__)

#
# Provide a global value that indicates which expression system is being used
#
class Mode(object):
    coopr3_trees = (1,)
    pyomo4_trees = (2,)
    pyomo5_trees = (3,)
mode = Mode.pyomo5_trees

#
# Pull symbols from the appropriate expression system
#
# Pyomo5
if mode == Mode.pyomo5_trees:
    from pyomo.core.expr import expr_pyomo5 as curr
    __public__.extend(curr.__public__)
    for obj in curr.__all__:
        globals()[obj] = getattr(curr, obj)
else:
    raise ValueError("WEH - Other expression systems aren't working right now.")    #pragma: no cover



# Initialize numvalue functions
from pyomo.core.expr import numvalue
numvalue._generate_sum_expression = _generate_sum_expression
numvalue._generate_mul_expression = _generate_mul_expression
numvalue._generate_other_expression = _generate_other_expression
numvalue._generate_relational_expression = _generate_relational_expression


#
# Common intrinsic functions
#
from pyomo.core.expr import expr_common as common

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
