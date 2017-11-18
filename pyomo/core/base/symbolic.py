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
from pyomo import core
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import native_types
from pyomo.util import DeveloperError

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
except ImportError: #pragma:nocover
    _sympy_available = False


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
    # Clone the expression and map intrinsic functions
    #
    #tmp_expr = EXPR.clone_expression( expr, memo=pyomo2sympy )
    tmp_expr = _map_intrinsic_functions(expr, sympy2pyomo, pyomo2sympy)
    #
    # Call to_string() to create a string version of this expression
    #
    buf = StringIO()
    if isinstance(tmp_expr, EXPR._ExpressionBase):
        tmp_expr.to_string(buf)
        tmp_expr = buf.getvalue()
        buf.close()
    else:
        tmp_expr = str(tmp_expr)
    #
    # Sympify the string representation of the expression
    #
    sympy_expr = sympy.sympify(
        tmp_expr, locals=dict((str(x), x) for x in sympy_vars) )
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


class ReplaceIntrinsicFunctions(EXPR.ExpressionReplacementVisitor):

    def __init__(self, native_or_sympy_types, pyomo2sympy):
        self.native_or_sympy_types = native_or_sympy_types
        self.pyomo2sympy = pyomo2sympy

    def visit_potential_leaf(self, node):
        #
        # Don't replace native or sympy types
        #
        if type(node) in self.native_or_sympy_types:
            return False, None
        #
        # Replace pyomo variables with sympy variables
        #
        if id(node) in self.pyomo2sympy:
            return True, self.pyomo2sympy[id(node)]
        #
        # Replace some intrinsic functions whose name or
        # definition change.
        #
        if isinstance(node, EXPR._UnaryFunctionExpression):
            if node._name == 'ceil':
                node_ = node.__class__(list(node.args))
                node_._name = 'ceiling'
                return True, node_
                
            if node._name == 'log10':
                node_ = node.__class__([node/EXPR.log(10)])
                node_._name = 'log'
                return True, node_
        #
        # Don't replace anything else
        #
        return False, None


def _map_intrinsic_functions(expr, sympySymbols, pyomo2sympy):
    #
    # Handle simple cases
    #
    if expr.__class__ in native_numeric_types:
        return expr
    if not expr.is_expression():
        return expr
    #
    # Create the visitor and call it.
    #
    visitor = ReplaceIntrinsicFunctions(sympySymbols, pyomo2sympy)
    return visitor.dfs_postorder_stack(expr)


def X_map_intrinsic_functions(expr, sympySymbols):
    coopr3_mode = expr_common.mode is expr_common.Mode.coopr3_trees
    native_or_sympy_types = set(native_types)
    native_or_sympy_types.add( type(list(sympySymbols)[0]) )

    _stack = [ ([expr], 0, 1) ]
    while _stack:
        _argList, _idx, _len = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if type(_sub) in native_or_sympy_types:
                pass
            elif _sub.is_expression():
                # Substitute intrinsic function names
                if isinstance(_sub, EXPR._UnaryFunctionExpression):
                    if _sub._name == 'ceil':
                        _sub._name = 'ceiling'
                    if _sub._name == 'log10':
                        _sub._name = 'log'
                        _sub = _sub / EXPR.log(10)
                        if _argList.__class__ is tuple:
                            # Scary: this assumes the args are ONLY
                            # stored in the _args attribute of the
                            # parent node.  This is true for everything
                            # except Coopr3 _ProductExpression.
                            # Fortunately, those arguments are lists
                            # (and not tuples), so we will hit the
                            # in-place modification below.  This is also
                            # not the case for the root node -- but we
                            # also force that to be a list above.
                            tmp = list(_argList)
                            tmp[_idx-1] = _sub
                            _argList = tuple(tmp)
                            _stack[-1][0][ _stack[-1][1]-1 ]._args = _argList
                        else:
                            _argList[_idx-1] = _sub

                _stack.append(( _argList, _idx, _len ))
                if coopr3_mode and type(_sub) is EXPR._ProductExpression:
                    if _sub._denominator:
                        _stack.append(
                            (_sub._denominator, 0, len(_sub._denominator)) )
                    _argList = _sub._numerator
                else:
                    _argList = _sub._args
                _idx = 0
                _len = _sub.nargs()
    return _argList[0]


# =====================================================
# _map_sympy2pyomo
# =====================================================

class Sympy2PyomoVisitor(EXPR.ExpressionValueVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        _sympyOp = node
        _op = _operatorMap.get( type(_sympyOp), None )
        if _op is None:
            raise DeveloperError(
                "sympy expression type '%s' not found in the operator "
                "map" % type(_sympyOp) )
        return _op(*tuple(values))

    def visiting_potential_leaf(self, node, _values):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.nargs() > 0:
            return False
        if node in sympy2pyomo:
            _values.append( sympy2pyomo[node] )
        else:
            _values.append( node.evalf() )
        return True

    def finalize(self, ans):
        return ans


def _map_sympy2pyomo(expr, sympy2pyomo):
    _stack = [ ([expr], 0, 1) ]
    while 1:
        _argList, _idx, _len = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if not _sub._args:
                if _sub in sympy2pyomo:
                    _sub = _argList[_idx-1] = sympy2pyomo[_sub]
                else:
                    _sub = _argList[_idx-1] = float(_sub.evalf())
                continue

            _stack.append(( _argList, _idx, _len ))
            _argList = list(_sub._args)
            _idx = 0
            _len = _sub.nargs()

        # Substitute the operator
        if not _stack:
            return _argList[0]
        else:
            _sympyOp = _stack[-1][0][ _stack[-1][1]-1 ]
            _op = _operatorMap.get( type(_sympyOp), None )
            if _op is None:
                raise DeveloperError(
                    "sympy expression type '%s' not found in the operator "
                    "map for expression %s"
                    % (type(_sympyOp), expr) )
            _stack[-1][0][ _stack[-1][1]-1 ] = _op(*tuple(_argList))
    # No return
