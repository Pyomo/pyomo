#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import StringIO
import pyutilib.misc
from pyomo import core
from pyomo.core.expr import current as EXPR
from pyomo.core.expr import native_types
from pyomo.common import DeveloperError
from pyomo.core.expr.numvalue import value

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
        raise NondifferentiableError(
            "The sub-expression '%s' is not differentiable with respect to %s"
            % (x[0],x[1]) )

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
        sympy.Derivative: _nondifferentiable,
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
    }
except ImportError: #pragma:nocover
    _sympy_available = False

# A "public" attribute indicating that differentiate() can be called
# ... this provides a bit of future-proofing for alternative approaches
# to symbolic differentiation.
differentiate_available = _sympy_available

class NondifferentiableError(ValueError):
    """A Pyomo-specific ValueError raised for non-differentiable expressions"""
    pass


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
            "The sympy module is not available.  "
            "Cannot perform automatic symbolic differentiation.")
    if not (( wrt is None ) ^ ( wrt_list is None )):
        raise ValueError(
            "differentiate(): Must specify exactly one of wrt and wrt_list")
    #
    # Setup the WRT list
    #
    if wrt is not None:
        wrt_list = [ wrt ]
    else:
        # Copy the list because we will normalize things in place below
        wrt_list = list(wrt_list)
    #
    # Setup mapping dictionaries
    #
    pyomo_vars = list(EXPR.identify_variables(expr))
    pyomo_vars = sorted(pyomo_vars, key=lambda x: str(x))
    sympy_vars = [sympy.var('x%s'% i) for i in range(len(pyomo_vars))]
    sympy2pyomo = dict( zip(sympy_vars, pyomo_vars) )
    pyomo2sympy = dict( (id(pyomo_vars[i]), sympy_vars[i])
                         for i in range(len(pyomo_vars)) )
    #
    # Process WRT information
    #
    ans = []
    for i, target in enumerate(wrt_list):
        if target.__class__ is not tuple:
            wrt_list[i] = target = (target,)
        mismatch_target = False
        for var in target:
            if id(var) not in pyomo2sympy:
                mismatch_target = True
                break
        wrt_list[i] = tuple( pyomo2sympy.get(id(var),None) for var in target )
        ans.append(0 if mismatch_target else None)
    #
    # If there is nothing to do, do nothing
    #
    if all(i is not None for i in ans):
        return ans if wrt is None else ans[0]
    #
    # Create sympy expression
    #
    sympy_expr = sympify_expression(expr, sympy2pyomo, pyomo2sympy)
    #
    # Differentiate for each WRT variable, and map the
    # result back into a Pyomo expression tree.
    #
    for i, target in enumerate(wrt_list):
        if ans[i] is None:
            sympy_ans = sympy_expr.diff(*target)
            ans[i] = _map_sympy2pyomo(sympy_ans, sympy2pyomo)
    #
    # Return the answer
    #
    return ans if wrt is None else ans[0]


# =====================================================
# sympify_expression
# =====================================================

class SympifyVisitor(EXPR.ExpressionValueVisitor):

    def __init__(self, native_or_sympy_types, pyomo2sympy):
        self.native_or_sympy_types = native_or_sympy_types
        self.pyomo2sympy = pyomo2sympy

    def visit(self, node, values):
        if node.__class__ is EXPR.UnaryFunctionExpression:
            return _functionMap[node._name](values[0])
        else:
            return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        #
        # Don't replace native or sympy types
        #
        if type(node) in self.native_or_sympy_types:
            return True, node
        #
        # Replace pyomo variables with sympy variables
        #
        if id(node) in self.pyomo2sympy:
            return True, self.pyomo2sympy[id(node)]
        #
        # Replace constants
        #
        if not node.is_potentially_variable():
            return True, value(node)
        #
        # Don't replace anything else
        #
        return False, None

    def finalize(self, ans):
        return ans

def sympify_expression(expr, sympySymbols, pyomo2sympy):
    #
    # Handle simple cases
    #
    if expr.__class__ in native_types:
        return expr
    if not expr.is_expression_type():
        if id(expr) in pyomo2sympy:
            return pyomo2sympy[id(expr)]
        return expr
    #
    # Create the visitor and call it.
    #
    native_or_sympy_types = set(native_types)
    native_or_sympy_types.add( type(list(sympySymbols)[0]) )
    visitor = SympifyVisitor(native_or_sympy_types, pyomo2sympy)
    return visitor.dfs_postorder_stack(expr)


# =====================================================
# _map_sympy2pyomo
# =====================================================

class Sympy2PyomoVisitor(pyutilib.misc.ValueVisitor):

    def __init__(self, sympy2pyomo):
        self.sympy2pyomo = sympy2pyomo

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        _sympyOp = node
        _op = _operatorMap.get( type(_sympyOp), None )
        if _op is None:
            raise DeveloperError(
                "sympy expression type '%s' not found in the operator "
                "map" % type(_sympyOp) )
        return _op(*tuple(values))

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if not node._args:
            if node in self.sympy2pyomo:
                return True, self.sympy2pyomo[node]
            else:
                return True, float(node.evalf())
        return False, None

    def children(self, node):
        return list(node._args)

    def finalize(self, ans):
        return ans

def _map_sympy2pyomo(expr, sympy2pyomo):
    if not expr._args:
        if expr in sympy2pyomo:
            return sympy2pyomo[expr]
        else:
            return float(expr.evalf())
    visitor = Sympy2PyomoVisitor(sympy2pyomo)
    return visitor.dfs_postorder_stack(expr)
