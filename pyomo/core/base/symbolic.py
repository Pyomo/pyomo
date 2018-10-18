#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import StringIO, iterkeys
import pyutilib.misc
from pyomo import core
from pyomo.common import DeveloperError
from pyomo.core.expr import current as EXPR, native_types
from pyomo.core.expr.numvalue import value
from pyomo.core.kernel.component_map import ComponentMap

_sympy_available = True
try:
    import sympy

    def _prod(*x):
        ans = x[0]
        for i in x[1:]:
            ans *= i
        return ans

    def _sum(*x):
        return sum(x_ for x_ in x)

    def _nondifferentiable(*x):
        if type(x[1]) is tuple:
            # sympy >= 1.3 returns tuples (var, order)
            wrt = x[1][0]
        else:
            # early versions of sympy returned the bare var
            wrt = x[1]
        raise NondifferentiableError(
            "The sub-expression '%s' is not differentiable with respect to %s"
            % (x[0], wrt) )

    _operatorMap = {
        sympy.Add: _sum,
        sympy.Mul: _prod,
        sympy.Pow: lambda x, y: x**y,
        sympy.exp: lambda x: core.exp(x),
        sympy.log: lambda x: core.log(x),
        sympy.sin: lambda x: core.sin(x),
        sympy.asin: lambda x: core.asin(x),
        sympy.sinh: lambda x: core.sinh(x),
        sympy.asinh: lambda x: core.asinh(x),
        sympy.cos: lambda x: core.cos(x),
        sympy.acos: lambda x: core.acos(x),
        sympy.cosh: lambda x: core.cosh(x),
        sympy.acosh: lambda x: core.acosh(x),
        sympy.tan: lambda x: core.tan(x),
        sympy.atan: lambda x: core.atan(x),
        sympy.tanh: lambda x: core.tanh(x),
        sympy.atanh: lambda x: core.atanh(x),
        sympy.ceiling: lambda x: core.ceil(x),
        sympy.floor: lambda x: core.floor(x),
        sympy.sqrt: lambda x: core.sqrt(x),
        sympy.Derivative: _nondifferentiable,
        sympy.Tuple: lambda *x: x,
    }

    _pyomo_operator_map = {
        EXPR.SumExpression: sympy.Add,
        EXPR.ProductExpression: sympy.Mul,
        EXPR.NPV_ProductExpression: sympy.Mul,
        EXPR.MonomialTermExpression: sympy.Mul,
    }

    _functionMap = {
        'exp': sympy.exp,
        'log': sympy.log,
        'log10': lambda x: sympy.log(x)/sympy.log(10),
        'sin': sympy.sin,
        'asin': sympy.asin,
        'sinh': sympy.sinh,
        'asinh': sympy.asinh,
        'cos': sympy.cos,
        'acos': sympy.acos,
        'cosh': sympy.cosh,
        'acosh': sympy.acosh,
        'tan': sympy.tan,
        'atan': sympy.atan,
        'tanh': sympy.tanh,
        'atanh': sympy.atanh,
        'ceil': sympy.ceiling,
        'floor': sympy.floor,
        'sqrt': sympy.sqrt,
    }
except ImportError:
    _sympy_available = False

# A "public" attribute indicating that differentiate() can be called
# ... this provides a bit of future-proofing for alternative approaches
# to symbolic differentiation.
differentiate_available = _sympy_available

class NondifferentiableError(ValueError):
    """A Pyomo-specific ValueError raised for non-differentiable expressions"""
    pass

class PyomoSympyBimap(object):
    def __init__(self):
        self.pyomo2sympy = ComponentMap()
        self.sympy2pyomo = {}
        self.i = 0

    def getPyomoSymbol(self, sympy_object, default=None):
        return self.sympy2pyomo.get(sympy_object, default)

    def getSympySymbol(self, pyomo_object):
        if pyomo_object in self.pyomo2sympy:
            return self.pyomo2sympy[pyomo_object]
        sympy_obj = sympy.Symbol("x%d" % self.i)
        self.i += 1
        self.pyomo2sympy[pyomo_object] = sympy_obj
        self.sympy2pyomo[sympy_obj] = pyomo_object
        return sympy_obj

    def sympyVars(self):
        return iterkeys(self.sympy2pyomo)

def differentiate(expr, wrt=None, wrt_list=None):
    """Return derivative of expression.

    This function returns an expression or list of expression objects
    corresponding to the derivative of the passed expression 'expr' with
    respect to a variable 'wrt' or list of variables 'wrt_list'

    Args:
        expr (Expression): Pyomo expression
        wrt (Var): Pyomo variable
        wrt_list (list): list of Pyomo variables

    Returns:
        Expression or list of Expression objects

    """
    if not _sympy_available:
        raise RuntimeError(
            "The sympy module is not available.\n\t"
            "Cannot perform automatic symbolic differentiation.")
    if not (( wrt is None ) ^ ( wrt_list is None )):
        raise ValueError(
            "differentiate(): Must specify exactly one of wrt and wrt_list")
    #
    # Convert the Pyomo expression to a sympy expression
    #
    objectMap, sympy_expr = sympify_expression(expr)
    #
    # The partial_derivs dict holds intermediate sympy expressions that
    # we can re-use.  We will prepopulate it with None for all vars that
    # appear in the expression (so that we can detect wrt combinations
    # that are, by definition, 0)
    #
    partial_derivs = dict((x,None) for x in objectMap.sympyVars())
    #
    # Setup the WRT list
    #
    if wrt is not None:
        wrt_list = [ wrt ]
    else:
        # Copy the list because we will normalize things in place below
        wrt_list = list(wrt_list)
    #
    # Convert WRT vars into sympy vars
    #
    ans = [None]*len(wrt_list)
    for i, target in enumerate(wrt_list):
        if target.__class__ is not tuple:
            target = (target,)
        wrt_list[i] = tuple(objectMap.getSympySymbol(x) for x in target)
        for x in wrt_list[i]:
            if x not in partial_derivs:
                ans[i] = 0.
                break
    #
    # We assume that users will not request duplicate derivatives.  We
    # will only cache up to the next-to last partial, and if a user
    # requests the exact same derivative twice, then we will just
    # re-calculate it.
    #
    last_partial_idx = max(len(x) for x in wrt_list) - 1
    #
    # Calculate all the derivatives
    #
    for i, target in enumerate(wrt_list):
        if ans[i] is not None:
            continue
        part = sympy_expr
        for j, wrt_var in enumerate(target):
            if j == last_partial_idx:
                part = sympy.diff(part, wrt_var)
            else:
                partial_target = target[:j+1]
                if partial_target in partial_derivs:
                    part = partial_derivs[partial_target]
                else:
                    part = sympy.diff(part, wrt_var)
                    partial_derivs[partial_target] = part
        ans[i] = sympy2pyomo_expression(part, objectMap)
    #
    # Return the answer
    #
    return ans if wrt is None else ans[0]


# =====================================================
# sympify_expression
# =====================================================

class Pyomo2SympyVisitor(EXPR.StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        super(Pyomo2SympyVisitor, self).__init__()
        self.object_map = object_map

    def exitNode(self, node, values):
        if node.__class__ is EXPR.UnaryFunctionExpression:
            return _functionMap[node._name](values[0])
        _op = _pyomo_operator_map.get(node.__class__, None)
        if _op is None:
            return node._apply_operation(values)
        else:
            return _op(*tuple(values))

    def beforeChild(self, node, child):
        #
        # Don't replace native or sympy types
        #
        if type(child) in native_types:
            return False, child
        #
        # We will descend into all expressions...
        #
        if child.is_expression_type():
            return True, None
        #
        # Replace pyomo variables with sympy variables
        #
        if child.is_potentially_variable():
            return False, self.object_map.getSympySymbol(child)
        #
        # Everything else is a constant...
        #
        return False, value(child)

class Sympy2PyomoVisitor(EXPR.StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        super(Sympy2PyomoVisitor, self).__init__()
        self.object_map = object_map

    def enterNode(self, node):
        return (node._args, [])

    def exitNode(self, node, values):
        """ Visit nodes that have been expanded """
        _sympyOp = node
        _op = _operatorMap.get( type(_sympyOp), None )
        if _op is None:
            raise DeveloperError(
                "sympy expression type '%s' not found in the operator "
                "map" % type(_sympyOp) )
        return _op(*tuple(values))

    def beforeChild(self, node, child):
        if not child._args:
            item = self.object_map.getPyomoSymbol(child, None)
            if item is None:
                item = float(child.evalf())
            return False, item
        return True, None

def sympify_expression(expr):
    """Convert a Pyomo expression to a Sympy expression"""
    #
    # Create the visitor and call it.
    #
    object_map = PyomoSympyBimap()
    visitor = Pyomo2SympyVisitor(object_map)
    is_expr, ans = visitor.beforeChild(None, expr)
    if not is_expr:
        return object_map, ans

    return object_map, visitor.walk_expression(expr)


def sympy2pyomo_expression(expr, object_map):
    visitor = Sympy2PyomoVisitor(object_map)
    is_expr, ans = visitor.beforeChild(None, expr)
    if not is_expr:
        return ans
    return visitor.walk_expression(expr)
