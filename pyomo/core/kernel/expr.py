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
import builtins

__all__ = ('fabs', 'log', 'log10', 'sin', 'cos', 'tan', 'cosh', 'sinh', 'tanh',
           'asin', 'acos', 'atan', 'exp', 'sqrt', 'asinh', 'acosh',
           'atanh', 'ceil', 'floor', 'sum')

from pyomo.core.kernel import expr_common as common

def generate_expression(etype, _self,_other):
    raise RuntimeError("incomplete import of Pyomo expression system")
def generate_relational_expression(etype, lhs, rhs):
    raise RuntimeError("incomplete import of Pyomo expression system")
def generate_intrinsic_function_expression(arg, name, fcn):
    raise RuntimeError("incomplete import of Pyomo expression system")
def compress_expression(expr, verbose=False, dive=False, multiprod=False):
    return expr
sum = builtins.sum

from pyomo.core.kernel import numvalue

# Import global methods that are common to all expression systems
_common_module_members = [
    'clone_expression',
    'identify_variables',
    'generate_expression',
    'generate_intrinsic_function_expression',
    'generate_relational_expression',
    'chainedInequalityErrorMessage',
    '_ExpressionBase',
    '_EqualityExpression',
    '_InequalityExpression',
    '_ProductExpression',
    '_SumExpression',
    '_AbsExpression',
    '_PowExpression',
    '_ExternalFunctionExpression',
    '_GetItemExpression',
    'clone_counter',
    'Expr_if',
]
_coopr3_module_members = [
    '_IntrinsicFunctionExpression',
    'ignore_entangled_expressions',
]
_pyomo4_module_members = [
    'ignore_entangled_expressions',
    '_LinearExpression',
    '_DivisionExpression',
    '_NegationExpression',
    'EntangledExpressionError',
    '_IntrinsicFunctionExpression',
]
_pyomo5_module_members = [
    '_LinearExpression',
    '_StaticLinearExpression',
    'clone_counter',
    'linear_expression',
    'nonlinear_expression',
    'evaluate_expression',
    '_ReciprocalExpression',
    '_NegationExpression',
    '_ViewSumExpression',
    '_UnaryFunctionExpression',
    '_NPV_NegationExpression',
    '_NPV_ExternalFunctionExpression',
    '_NPV_PowExpression',
    '_NPV_ProductExpression',
    '_NPV_ReciprocalExpression',
    '_NPV_SumExpression',
    '_NPV_UnaryFunctionExpression',
    '_NPV_AbsExpression',
    '_Constant_NegationExpression',
    '_Constant_ExternalFunctionExpression',
    '_Constant_PowExpression',
    '_Constant_ProductExpression',
    '_Constant_ReciprocalExpression',
    '_Constant_SumExpression',
    '_Constant_UnaryFunctionExpression',
    '_Constant_AbsExpression',
]

def set_expression_tree_format(mode):
    if mode is common.Mode.coopr3_trees:
        from pyomo.core.kernel import expr_coopr3 as expr3
        for obj in _common_module_members:
            globals()[obj] = getattr(expr3, obj)
        for obj in _pyomo4_module_members:
            if obj in globals():
                del globals()[obj]
        for obj in _pyomo5_module_members:
            if obj in globals():
                del globals()[obj]
        for obj in _coopr3_module_members:
            globals()[obj] = getattr(expr3, obj)

    elif mode is common.Mode.pyomo4_trees:
        from pyomo.core.kernel import expr_pyomo4 as expr4
        for obj in _common_module_members:
            globals()[obj] = getattr(expr4, obj)
        for obj in _coopr3_module_members:
            if obj in globals():
                del globals()[obj]
        for obj in _pyomo5_module_members:
            if obj in globals():
                del globals()[obj]
        for obj in _pyomo4_module_members:
            globals()[obj] = getattr(expr4, obj)

    elif mode is common.Mode.pyomo5_trees:
        from pyomo.core.kernel import expr_pyomo5 as expr5
        for obj in _common_module_members:
            globals()[obj] = getattr(expr5, obj)
        for obj in _coopr3_module_members:
            if obj in globals():
                del globals()[obj]
        for obj in _pyomo4_module_members:
            if obj in globals():
                del globals()[obj]
        for obj in _pyomo5_module_members:
            globals()[obj] = getattr(expr5, obj)

    else:
        raise RuntimeError("Unrecognized expression tree mode: %s\n"
                           "Must be one of [%s, %s]"
                           % (mode,
                              common.Mode.coopr3_trees,
                              common.Mode.pyomo4_trees,
                              common.Mode.pyomo5_trees))
    #
    # Propagate the generate_expression functions to the numvalue namespace
    numvalue.generate_expression = generate_expression
    numvalue.generate_relational_expression = generate_relational_expression
    #
    common.mode = mode

set_expression_tree_format(common.mode)

def fabs(arg):
    # FIXME: We need to switch this over from generate_expression to
    # just use generate_intrinsic_function_expression
    #
    #return generate_intrinsic_function_expression(arg, 'fabs', math.fabs)
    return generate_expression(common._abs, arg, None)

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
    return generate_expression(common._pow, *args)

# FIXME: this is nominally the same as x ** 0.5, but follows a different
# path and produces a different NL file!
def sqrt(arg):
    return generate_intrinsic_function_expression(arg, 'sqrt', math.sqrt)
#    return generate_expression(common._pow, arg, 0.5)


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
