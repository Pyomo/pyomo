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
from pyomo.core.expr import current
from pyomo.common import DeveloperError
from pyomo.core.expr import current as EXPR, native_types
from pyomo.core.expr.numvalue import value
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.common.errors import NondifferentiableError

sympy_available = True
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
        sympy.exp: lambda x: current.exp(x),
        sympy.log: lambda x: current.log(x),
        sympy.sin: lambda x: current.sin(x),
        sympy.asin: lambda x: current.asin(x),
        sympy.sinh: lambda x: current.sinh(x),
        sympy.asinh: lambda x: current.asinh(x),
        sympy.cos: lambda x: current.cos(x),
        sympy.acos: lambda x: current.acos(x),
        sympy.cosh: lambda x: current.cosh(x),
        sympy.acosh: lambda x: current.acosh(x),
        sympy.tan: lambda x: current.tan(x),
        sympy.atan: lambda x: current.atan(x),
        sympy.tanh: lambda x: current.tanh(x),
        sympy.atanh: lambda x: current.atanh(x),
        sympy.ceiling: lambda x: current.ceil(x),
        sympy.floor: lambda x: current.floor(x),
        sympy.sqrt: lambda x: current.sqrt(x),
        sympy.Abs: lambda x: abs(x),
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
    sympy_available = False

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
        # Pyomo currently ONLY supports Real variables (not complex
        # variables).  If that ever changes, then we will need to
        # revisit hard-coding the symbol type here
        sympy_obj = sympy.Symbol("x%d" % self.i, real=True)
        self.i += 1
        self.pyomo2sympy[pyomo_object] = sympy_obj
        self.sympy2pyomo[sympy_obj] = pyomo_object
        return sympy_obj

    def sympyVars(self):
        return iterkeys(self.sympy2pyomo)

# =====================================================
# sympyify_expression
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

def sympyify_expression(expr):
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
