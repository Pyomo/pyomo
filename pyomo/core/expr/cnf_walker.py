from six import iterkeys

from pyomo.common import DeveloperError
from pyomo.core.expr.numvalue import native_types, value
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.expr.logical_expr import (
    LogicalExpressionBase,
    NotExpression, BinaryLogicalExpression, MultiArgsExpression,
    AndExpression, OrExpression, ImplicationExpression, EquivalenceExpression,
    XorExpression,
    ExactlyExpression, AtMostExpression, AtLeastExpression, Not, Equivalent,
    LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor,
)

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor

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

    _operatorMap = {
        sympy.Or: LogicalOr,
        sympy.And: LogicalAnd,
        sympy.Implies: Implies,
        sympy.Equivalent: Equivalent,
        sympy.Not: Not,
    }

    _pyomo_operator_map = {
        AndExpression: sympy.And,
        OrExpression: sympy.Or,
        ImplicationExpression: sympy.Implies,
        EquivalenceExpression: sympy.Equivalent,
        XorExpression: sympy.Xor,
        NotExpression: sympy.Not,
    }
except ImportError:
    sympy_available = False


class PyomoSympyLogicalBimap(object):
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


class Pyomo2SympyVisitor(StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        super(Pyomo2SympyVisitor, self).__init__()
        self.object_map = object_map

    def exitNode(self, node, values):
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
            if child.__class__ in (ExactlyExpression, AtMostExpression, AtLeastExpression):
                return False, self.object_map.getSympySymbol(child)
            else:
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


class Sympy2PyomoVisitor(StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        super(Sympy2PyomoVisitor, self).__init__()
        self.object_map = object_map

    def enterNode(self, node):
        return (node.args, [])

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
        if not child.args:
            item = self.object_map.getPyomoSymbol(child, None)
            if item is None:
                item = float(child.evalf())
            return False, item
        return True, None


def to_cnf(expr):
    symbol_map, sympy_expr = sympyify_expression(expr)
    cnf_form = sympy.to_cnf(sympy_expr)
    return sympy2pyomo_expression(cnf_form, symbol_map)


def sympyify_expression(expr):
    """Convert a Pyomo expression to a Sympy expression"""
    #
    # Create the visitor and call it.
    #
    object_map = PyomoSympyLogicalBimap()
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