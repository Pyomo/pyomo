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

__public__ = ['linear_expression', 'nonlinear_expression']
__all__ = (
'linear_expression',
'nonlinear_expression',
'decompose_term',
'clone_counter',
'clone_expression',
'evaluate_expression',
'identify_variables',
'generate_sum_expression',
'generate_mul_expression',
'generate_other_expression',
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
'_NPV_ExternalFunctionExpression',
'_GetItemExpression',
'Expr_if',
'_LinearExpression',
'_StaticLinearExpression',
'_ReciprocalExpression',
'_NegationExpression',
'_ViewSumExpression',
'_MutableViewSumExpression',
'_UnaryFunctionExpression',
'compress_expression',
'_NPV_NegationExpression',
'_NPV_ExternalFunctionExpression',
'_NPV_PowExpression',
'_NPV_ProductExpression',
'_NPV_ReciprocalExpression',
'_NPV_SumExpression',
'_NPV_UnaryFunctionExpression',
'_NPV_AbsExpression',
)

import math
import logging
import sys
import traceback
from copy import deepcopy
from collections import deque
from itertools import islice
from six import StringIO, next, string_types, itervalues
from six.moves import xrange, builtins
from weakref import ref

logger = logging.getLogger('pyomo.core')

from pyutilib.misc.visitor import SimpleVisitor, ValueVisitor
from pyutilib.math.util import isclose

from pyomo.core.expr.numvalue import \
    (NumericValue,
     NumericConstant,
     native_types,
     native_numeric_types,
     as_numeric,
     value)
from pyomo.core.expr.expr_common import \
    (_add, _sub, _mul, _div,
     _pow, _neg, _abs, _inplace,
     _unary, _radd, _rsub, _rmul,
     _rdiv, _rpow, _iadd, _isub,
     _imul, _idiv, _ipow, _lt, _le,
     _eq) 
from pyomo.core.expr import expr_common as common


def chainedInequalityErrorMessage(msg=None):
    if msg is None:
        msg = "Relational expression used in an unexpected Boolean context."
    buf = StringIO()
    _InequalityExpression.chainedInequality.to_string(buf)
    # We are about to raise an exception, so it's OK to reset chainedInequality
    info = _InequalityExpression.call_info
    _InequalityExpression.chainedInequality = None
    _InequalityExpression.call_info = None

    args = ( str(msg).strip(), buf.getvalue().strip(), info[0], info[1],
             ':\n    %s' % info[3] if info[3] is not None else '.' )
    return """%s

The inequality expression:
    %s
contains non-constant terms (variables) that were evaluated in an
unexpected Boolean context at
  File '%s', line %s%s

Evaluating Pyomo variables in a Boolean context, e.g.
    if expression <= 5:
is generally invalid.  If you want to obtain the Boolean value of the
expression based on the current variable values, explicitly evaluate the
expression using the value() function:
    if value(expression) <= 5:
or
    if value(expression <= 5):
""" % args


_ParamData = None
SimpleParam = None
TemplateExpressionError = None
def initialize_expression_data():
    global pyomo5_variable_types
    if pyomo5_variable_types is None:
        from pyomo.core.base import _VarData, _GeneralVarData, SimpleVar
        from pyomo.core.kernel.component_variable import IVariable, variable
        pyomo5_variable_types = set([_VarData, _GeneralVarData, IVariable, variable, SimpleVar])
        _LinearExpression.vtypes = pyomo5_variable_types
        #
        global _ParamData
        global SimpleParam
        global TemplateExpressionError
        from pyomo.core.base.param import _ParamData, SimpleParam
        from pyomo.core.base.template_expr import TemplateExpressionError
    # [functionality] chainedInequality allows us to generate symbolic
    # expressions of the type "a < b < c".  This provides a buffer to hold
    # the first inequality so the second inequality can access it later.
    _InequalityExpression.chainedInequality = None
    _InequalityExpression.call_info = None



def compress_expression(expr):
    return expr


class clone_counter_context(object):
    _count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def count(self):
        return clone_counter_context._count

clone_counter = clone_counter_context()


class mutable_sum_context(object):

    def __enter__(self):
        self.e = _MutableViewSumExpression([])
        return self.e

    def __exit__(self, *args):
        pass
        if self.e.__class__ == _MutableViewSumExpression:
            self.e.__class__ = _ViewSumExpression

nonlinear_expression = mutable_sum_context()


class mutable_linear_context(object):

    def __enter__(self):
        self.e = _LinearExpression()
        return self.e

    def __exit__(self, *args):
        if self.e.__class__ == _LinearExpression:
            self.e.__class__ = _StaticLinearExpression

linear_expression = mutable_linear_context()


#-------------------------------------------------------
#
# Visitor Logic
#
#-------------------------------------------------------

class SimpleExpressionVisitor(SimpleVisitor):

    def xbfs(self, node):
        """
        Breadth-first search, except that 
        leaf nodes are immediately visited.
        """
        dq = deque([node])
        while dq:
            current = dq.popleft()
            self.visit(current)
            #for c in self.children(current):
            for c in current.args:
                #if self.is_leaf(c):
                if c.__class__ in native_numeric_types or not c.is_expression() or c.nargs() == 0:
                    self.visit(c)
                else:
                    dq.append(c)
        return self.finalize()

    def xbfs_yield_leaves(self, node):
        """
        Breadth-first search, except that 
        leaf nodes are immediately visited.
        """
        #
        # If we start with a leaf, then yield it and stop iteration
        #
        if not node.__class__ in pyomo5_expression_types or node.nargs() == 0:
            ans = self.visit(node)
            if not ans is None:
                yield ans
            raise StopIteration
        #
        # Iterate through the tree.
        #
        dq = deque([node])
        while dq:
            current = dq.popleft()
            #self.visit(current)
            #for c in self.children(current):
            for c in current.args:
                #if self.is_leaf(c):
                if c.__class__ in pyomo5_expression_types and c.nargs() > 0:
                    dq.append(c)
                else:
                    ans = self.visit(c)
                    if not ans is None:
                        yield ans


class ValueExpressionVisitor(ValueVisitor):

    #
    # NOTE: This is not currently being used, so coverage testing is disabled
    #
    def dfs_postorder_deque(self, node):            #pragma: no cover
        """
        Depth-first search - postorder

        This function is slightly optimized from 
        ValueVisitor.dfs_postorder_deque
        """
        _values = [[]]
        expanded = set()
        dq = deque([node])
        while dq:
            current = dq[-1]
            if id(current) in expanded:
                dq.pop()
                values = _values.pop()
                _values[-1].append( self.visit(current, values) )
            elif self.visiting_potential_leaf(current, _values[-1]):
                dq.pop()
            else:
                #for c in reversed(self.children(current)):
                for c in reversed(current.args):
                    dq.append(c)
                expanded.add(id(current))
                _values.append( [] )
        return self.finalize(_values[-1][0])

    def dfs_postorder_stack(self, node):
        """
        Depth-first search - postorder

        This function is slightly optimized from 
        ValueVisitor.dfs_postorder_stack
        """
        #_stack = [ (node, self.children(node), 0, len(self.children(node)), [])]
        _stack = [ (node, node._args, 0, node.nargs(), [])]
        #
        # Iterate until the stack is empty
        #
        # Note: 1 is faster than True for Python 2.x
        #
        while 1:
            #
            # Get the top of the stack
            #   _obj        Current expression object
            #   _argList    The arguments for this expression objet
            #   _idx        The current argument being considered
            #   _len        The number of arguments
            #   _result     The return values
            #
            _obj, _argList, _idx, _len, _result = _stack.pop()
            #
            # Iterate through the arguments
            #
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                if not self.visiting_potential_leaf(_sub, _result):
                    #
                    # Push an expression onto the stack
                    #
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj                    = _sub
                    #_argList                = self.children(_sub)
                    _argList                = _sub._args
                    _idx                    = 0
                    _len                    = _sub.nargs()
                    _result                 = []
            #
            # Process the current node
            #
            ans = self.visit(_obj, _result)
            if _stack:
                #
                # "return" the recursion by putting the return value on the end of the results stack
                #
                _stack[-1][-1].append( ans )
            else:
                return self.finalize(ans)



#-------------------------------------------------------
#
# Functions used to process expression trees
#
#-------------------------------------------------------

# =====================================================
#  clone_expression
# =====================================================

class CloneVisitor(ValueExpressionVisitor):

    def __init__(self, clone_leaves=False, memo=None):
        self.clone_leaves = clone_leaves
        self.memo = memo

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._clone( tuple(values), self.memo )

    def visiting_potential_leaf(self, node, _values):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_numeric_types:
            #
            # Store a native or numeric object
            #
            _values.append( deepcopy(node, self.memo) )
            return True
        elif node.__class__ not in pyomo5_expression_types:
            #
            # Store a kernel object that is cloned
            #
            if self.clone_leaves:
                _values.append( deepcopy(node, self.memo) )
            else:
                _values.append( node )
            return True
        return False

    def finalize(self, ans):
        return ans


def clone_expression(expr, memo=None, verbose=False, clone_leaves=True):
    clone_counter_context._count += 1
    if not memo:
        memo = {'__block_scope__': { id(None): False }}
    #
    if expr.__class__ in native_numeric_types:
        return expr
    if not expr.is_expression():
        return deepcopy(expr, memo)
    #
    visitor = CloneVisitor(clone_leaves=clone_leaves, memo=memo)
    return visitor.dfs_postorder_stack(expr)


# =====================================================
#  sizeof_expression
# =====================================================

class SizeVisitor(SimpleExpressionVisitor):

    def __init__(self):
        self.counter = 0

    def visit(self, node):
        self.counter += 1

    #def children(self, node):
    #    return node._args

    #def is_leaf(self, node):
    #    return node.__class__ in native_numeric_types or not node.is_expression() or node.nargs() == 0

    def finalize(self):
        return self.counter


def sizeof_expression(expr, verbose=False):
    #if expr.__class__ in native_numeric_types or not expr.is_expression():
    #    return 1
    visitor = SizeVisitor()
    return visitor.xbfs(expr)
    
 
# =====================================================
#  evaluate_expression
# =====================================================

class EvaluationVisitor(ValueExpressionVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node, _values):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_numeric_types:
            _values.append( node )
            return True

        elif node.__class__ in pyomo5_variable_types:
            _values.append( value(node) )
            return True

        elif not node.is_expression():
            _values.append( value(node) )
            return True

        return False

    def finalize(self, ans):
        return ans


def evaluate_expression(exp, exception=True):
    try:
        if exp.__class__ in pyomo5_variable_types:
            return exp.value
        elif exp.__class__ in native_numeric_types:
            return exp
        elif not exp.is_expression():
            return exp()
        visitor = EvaluationVisitor()
        return visitor.dfs_postorder_stack(exp)

    except TemplateExpressionError:
        if exception:
            raise
        return None
    except ValueError:
        if exception:
            raise
        return None


# =====================================================
#  identify_variables
# =====================================================

class VariableVisitor(SimpleExpressionVisitor):

    def __init__(self, include_fixed=True, allow_duplicates=False, include_potentially_variable=False):
        self.seen = set()
        self.include_fixed = include_fixed
        self.allow_duplicates = allow_duplicates
        self.include_potentially_variable = include_potentially_variable
        
    def visit(self, node):
        if node.__class__ in pyomo5_variable_types:
            if ( self.include_fixed
                 or not node.is_fixed()
                 or self.include_potentially_variable ):
                if not self.allow_duplicates:
                    if id(node) in self.seen:
                        return
                    self.seen.add(id(node))
                return node
        elif self.include_potentially_variable and node.__class__ not in native_numeric_types and node._potentially_variable():
            if not self.allow_duplicates:
                if id(node) in self.seen:
                    return
                self.seen.add(id(node))
            return node


def identify_variables(expr,
                       include_fixed=True,
                       allow_duplicates=False,
                       include_potentially_variable=False):
    # OPTIONS:
    # include_fixed - list includes fixed variables
    # allow_duplicates - list of each variable encountered, even if it's a duplicate of one earlier
    # include_potentially_variable - list expression and other PV components, but not their variables
    #
    # WEH - it's not clear why this function shouldn't list all variables in the
    # contained expressions.  The include_potentially_variable option is confusing.
    #
    #
    # TODO: The allow_duplicates option isn't used anywhere, so we should remove it.
    #
    visitor = VariableVisitor(include_fixed, allow_duplicates, include_potentially_variable)
    for v in visitor.xbfs_yield_leaves(expr):
        yield v


# =====================================================
#  polynomial_degree
# =====================================================

class PolyDegreeVisitor(ValueExpressionVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._polynomial_degree(values)

    def visiting_potential_leaf(self, node, _values):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_types or not node._potentially_variable():
            _values.append( 0 )
            return True
        elif not node.is_expression():
            _values.append( 0 if node.is_fixed() else 1 )
            return True
        return False

    def finalize(self, ans):
        return ans


def polynomial_degree(node):
    visitor = PolyDegreeVisitor()
    return visitor.dfs_postorder_stack(node)


# =====================================================
#  expression_is_fixed
# =====================================================

class IsFixedVisitor(ValueExpressionVisitor):
    """
    NOTE: This doesn't check if combiner logic is 
    all or any and short-circuit the test.  It's
    not clear that that is an important optimization.
    """

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._is_fixed(values)

    def visiting_potential_leaf(self, node, _values):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_numeric_types or not node._potentially_variable():
            _values.append( True )
            return True

        elif not node.__class__ in pyomo5_expression_types:
            _values.append( node.is_fixed() ) 
            return True

        return False

    def finalize(self, ans):
        return ans


def expression_is_fixed(node):
    if not node._potentially_variable():
        return True
    visitor = IsFixedVisitor()
    return visitor.dfs_postorder_stack(node)


# =====================================================
#  expression_to_string
# =====================================================

"""
class StringVisitor(SimpleExpressionVisitor):

    def __init__(self, verbose=None, precedence=None):
        self.buf = StringIO()
        self.verbose = None
        self.precedence = None
        self._infix = False
        self._bypass_prefix = False

    def visit(self, node):
        self.counter += 1

    def children(self, node):
        return node._args

    def is_leaf(self, node):
        return node.__class__ in native_numeric_types or not node.is_expression() or node.nargs() == 0

    def finalize(self):
        ans = self.buf.getvalue()
        buf.close()
        return ans

    def dfs_inorder(self, node):
        # Depth-first search - inorder
        expanded = set()
        dq = deque([node])
        while dq:
            current = dq.pop()
            if id(current) in expanded or self.is_leaf(current):
                self.visit(current)
            else:
                first = True
                for c in reversed(self.children(current)):
                    if first:
                        first = False
                    else:
                        dq.append(current)
                    dq.append(c)
                expanded.add(id(current))
        return self.finalize()

def NEW_expression_to_string(expr, verbose=None, precedence=None):
    visitor = StringVisitor(verbose=verbose, precedence=precedence)
    return visitor.dfs_inorder(expr)
"""


def expression_to_string(expr, ostream=None, verbose=None, precedence=None):
    _name_buffer = {}
    if ostream is None:
        ostream = sys.stdout
    if expr.__class__ in native_numeric_types:
        ostream.write(str(expr))
        return
    elif expr.__class__ in pyomo5_variable_types:
        ostream.write(str(expr))
        return
    verbose = common.TO_STRING_VERBOSE if verbose is None else verbose

    _infix = False
    _bypass_prefix = False
    argList = expr._args
    _stack = [ [ expr, argList, 0, expr.nargs(), precedence if precedence is not None else expr._precedence() ] ]
    while _stack:
        _parent, _args, _idx, _len, _prec = _stack[-1]
        _my_precedence = _parent._precedence()
        if _idx < _len:
            _sub = _args[_idx]
            _stack[-1][2] += 1
            if _parent._to_string_skip(_idx):
                continue
            if _infix:
                _bypass_prefix = _parent._to_string_infix(ostream, _idx, verbose)
            else:
                if not _bypass_prefix:
                    _parent._to_string_prefix(ostream, verbose)
                else:
                    _bypass_prefix = False
                if ((_len-_parent._to_string_skip(0) > 1) and _my_precedence > _prec) or not _my_precedence or verbose:
                    ostream.write("( ")
                    #ostream.write("%s %s %s %s %s %s( " % (str(_len), str(_idx), str(_my_precedence), str(_prec), str(verbose), str(type(_parent))))
                    #if _len == 2 and skip == 0:
                    #    ostream.write(" % ")
                    #    ostream.write(" %s " % str(_parent._to_string_skip(0)))
                    #    ostream.write(" % ")
                    #    ostream.write(str(_args[0]))
                    #    ostream.write(str(type(_args[0])))
                    #    ostream.write(" % ")
                    #    ostream.write(str(_args[1]))
                    #    ostream.write(str(type(_args[1])))
                    #    ostream.write(" % ")
                _infix = True
            if _sub.__class__ in pyomo5_expression_types:
                argList = _sub._args
                _stack.append([ _sub, argList, 0, _sub.nargs(), _my_precedence ])
                _infix = False
            elif hasattr(_parent, '_to_string_term'):
                _parent._to_string_term(ostream, _idx, _sub, _name_buffer, verbose)
            else:
                expr._to_string_term(ostream, _idx, _sub, _name_buffer, verbose)
        else:
            _parent._to_string_suffix(ostream, verbose)
            _stack.pop()
            if ((_len-_parent._to_string_skip(0) > 1) and _my_precedence > _prec) or not _my_precedence or verbose:
                ostream.write(" )")


#-------------------------------------------------------
#
# Expression classes
#
#-------------------------------------------------------


class _ExpressionBase(NumericValue):
    """
    An object that defines a mathematical expression that can be evaluated

    m.p = Param(default=10, mutable=False)
    m.q = Param(default=10, mutable=True)
    m.x = var()
    m.y = var(initialize=1)
    m.y.fixed = True

                            m.p     m.q     m.x     m.y
    constant                T       F       F       F
    potentially_variable    F       F       T       T
    npv                     T       T       F       F
    fixed                   T       T       F       T
    """

    __slots__ =  ('_args','_is_owned')
    PRECEDENCE = 0

    def __init__(self, args):
        self._args = args
        self._is_owned = False
        for arg in args:
            if arg.__class__ in pyomo5_expression_types:
                arg._is_owned = True

    def arg(self, index):
        if index < 0 or index >= self.nargs():
            raise KeyError("Invalid index for expression argument: %d" % index)
        return self._args[index]

    @property
    def args(self):
        return islice(self._args, self.nargs())

    def nargs(self):
        return 2

    def __getstate__(self):
        state = super(_ExpressionBase, self).__getstate__()
        for i in _ExpressionBase.__slots__:
           state[i] = getattr(self,i)
        return state

    def __nonzero__(self):      #pragma: no cover
        return bool(self())

    __bool__ = __nonzero__

    def __str__(self):
        from pyomo.repn import generate_standard_repn
        #if True:
        try:
            #
            # Try to factor the constant and linear terms when printing NONVERBOSE
            #
            if common.TO_STRING_VERBOSE:
                expr = self
            elif self.__class__ is _InequalityExpression:
                expr = self
                # TODO: chained inequalities
                #if self._args[0].__class__ is _InequalityExpression:
                #    repn0a = generate_standard_repn(self._args[0]._args[0], compress=False, quadratic=False, compute_values=False)
                #    repn0b = generate_standard_repn(self._args[0]._args[1], compress=False, quadratic=False, compute_values=False)
                #    lhs = _InequalityExpression( (repn0a.to_expression(), repn0b.to_expression()), self._args[0]._strict, self._args[0]._cloned_from)
                #    repn1 = generate_standard_repn(self._args[1], compress=False, quadratic=False, compute_values=False)
                #    expr = _InequalityExpression( (lhs, repn1.to_expression()), self._strict, self._cloned_from)
                #elif self._args[0].__class__ is _InequalityExpression:
                #    repn0 = generate_standard_repn(self._args[0], compress=False, quadratic=False, compute_values=False)
                #    repn1a = generate_standard_repn(self._args[1]._args[0], compress=False, quadratic=False, compute_values=False)
                #    repn1b = generate_standard_repn(self._args[1]._args[1], compress=False, quadratic=False, compute_values=False)
                #    rhs = _InequalityExpression( (repn1a.to_expression(), repn1b.to_expression()), self._args[1]._strict, self._args[1]._cloned_from)
                #    expr = _InequalityExpression( (repn0.to_expression(), rhs), self._strict, self._cloned_from)
                #else:
                #    repn0 = generate_standard_repn(self._args[0], compress=False, quadratic=False, compute_values=False)
                #    repn1 = generate_standard_repn(self._args[1], compress=False, quadratic=False, compute_values=False)
                #    expr = _InequalityExpression( (repn0.to_expression(), repn1.to_expression()), self._strict, self._cloned_from)
            elif self.__class__ is _EqualityExpression:
                repn0 = generate_standard_repn(self._args[0], quadratic=False, compute_values=False)
                repn1 = generate_standard_repn(self._args[1], quadratic=False, compute_values=False)
                expr = _EqualityExpression( (repn0.to_expression(), repn1.to_expression()) )
            else:
                repn = generate_standard_repn(self, quadratic=False, compute_values=False)
                expr = repn.to_expression()
        #else:
        except Exception as e:
            #print(str(e))
            #
            # Fall back to simply printing the expression in an
            # unfactored form.
            #
            expr = self
        #
        # Output the string
        #
        buf = StringIO()
        self.to_string(buf, expr=expr)
        ans = buf.getvalue()
        buf.close()
        return ans

    def __call__(self, exception=True):
        return evaluate_expression(self, exception)

    def clone(self, substitute=None, verbose=False):
        return clone_expression(self, memo=substitute, verbose=verbose, clone_leaves=False)

    def size(self, verbose=False):
        return sizeof_expression(self, verbose=verbose)

    def __deepcopy__(self, memo):
        return clone_expression(self, memo=memo)

    def _clone(self, args, memo):
        return self.__class__(args)

    def getname(self, *args, **kwds):                       #pragma: no cover
        """The text name of this Expression function"""
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement getname()" % ( str(self.__class__), ))

    # TODO: what if test was a lambda function?  Would that be faster?
    def is_constant(self):
        """Return True if this expression is an atomic constant

        This method contrasts with the is_fixed() method.  This method
        returns True if the expression is an atomic constant, that is it
        is composed exclusively of constants and immutable parameters.
        NumericValue objects returning is_constant() == True may be
        simplified to their numeric value at any point without warning.

        Note:  This defaults to False, but gets redefined in sub-classes.
        """
        return False

    def is_fixed(self):
        """Return True if this expression contains no free variables.

        The is_fixed() method returns True iff there are no free
        variables within this expression (i.e., all arguments are
        constants, params, and fixed variables).  The parameter values
        can of course change over time, but at any point in time, they
        are "fixed". hence, the name.

        """
        return expression_is_fixed(self)

    def _is_fixed(self, values):
        """Private method to be overridden by derived classes requiring special
        handling for computing is_fixed()

        This method should return a function that takes a list of the
        results of the is_fixed() for each of the arguments and
        returns True/False for this expression.

        """
        return all(values)

    def _potentially_variable(self):
        """Return True if this expression can potentially contain a variable

        The potentially_variable() method returns True iff there are -
        or could be - any variables within this expression (i.e., at any
        point in the future, it is possible that is_fixed() might return
        False).

        Note:  This defaults to False, but gets redefined in sub-classes.

        TODO: Rename _potentially_variable() to potentially_variable()
        """
        return True

    def is_expression(self):
        return True

    def polynomial_degree(self):
        return polynomial_degree(self)

    def _polynomial_degree(self, ans):                          #pragma: no cover
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement _polynomial_degree()" % ( str(self.__class__), ))

    def to_string(self, ostream=None, verbose=None, precedence=None, expr=None, labeler=None):
        if expr is None:
            expr = self
        return expression_to_string(expr, ostream, verbose, precedence)

    def _precedence(self):
        return _ExpressionBase.PRECEDENCE

    def _to_string_skip(self, _idx):
        return False

    def _to_string_term(self, ostream, _idx, _sub, _name_buffer, verbose):
        if _sub.__class__ in native_numeric_types:
            ostream.write(str(_sub))
        elif _sub.__class__ is NumericConstant:
            ostream.write(str(_sub()))
        elif hasattr(_sub, 'to_string'):
             #
             # Generate strings from components that contain expressions, but
             # don't just generate the component name.
             #
            _sub.to_string(ostream=ostream, verbose=verbose)
        elif hasattr(_sub, 'getname'):
            # BUG?  The kernel may return None from getname()
            _s = _sub.getname(True, _name_buffer)
            if _s is None:
                _s = str(_sub)
            ostream.write(_s)
        else:
            ostream.write(str(_sub))

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.getname())

    def _to_string_infix(self, ostream, idx, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            ostream.write(self._inline_operator())

    def _to_string_suffix(self, ostream, verbose):
        pass


class _NegationExpression(_ExpressionBase):
    __slots__ = ()

    PRECEDENCE = 4

    def nargs(self):
        return 1

    def getname(self, *args, **kwds):
        return 'neg'

    def _polynomial_degree(self, result):
        return result[0]

    def _precedence(self):
        return _NegationExpression.PRECEDENCE

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.getname())
        else:
            ostream.write("- ")

    def _apply_operation(self, result):
        return -result[0]


class _NPV_NegationExpression(_NegationExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


class _ExternalFunctionExpression(_ExpressionBase):
    __slots__ = ('_fcn',)

    def __init__(self, args, fcn=None):
        """Construct a call to an external function"""
        self._args = args
        self._fcn = fcn
        self._is_owned = False
        for arg in args:
            if arg.__class__ in pyomo5_expression_types:
                arg._is_owned = True

    def nargs(self):
        return len(self._args)

    def _clone(self, args, memo):
        return self.__class__(args, self._fcn)

    def __getstate__(self):
        result = super(_ExternalFunctionExpression, self).__getstate__()
        for i in _ExternalFunctionExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):           #pragma: no cover
        return self._fcn.getname(*args, **kwds)

    def _polynomial_degree(self, result):
        if result[0] is 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        """Evaluate the expression"""
        return self._fcn.evaluate( result )     #pragma: no cover

    def _to_string_prefix(self, ostream, verbose):
        return self._fcn.getname()

    def _inline_operator(self):                 #pragma: no cover
        return ", "


class _NPV_ExternalFunctionExpression(_ExternalFunctionExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


class _PowExpression(_ExpressionBase):

    __slots__ = ()
    PRECEDENCE = 2

    def _polynomial_degree(self, result):
        # _PowExpression is a tricky thing.  In general, a**b is
        # nonpolynomial, however, if b == 0, it is a constant
        # expression, and if a is polynomial and b is a positive
        # integer, it is also polynomial.  While we would like to just
        # call this a non-polynomial expression, these exceptions occur
        # too frequently (and in particular, a**2)
        l,r = result
        if isclose(r, 0):
            if isclose(l, 0):
                return 0
            try:
                # NOTE: use value before int() so that we don't
                #       run into the disabled __int__ method on
                #       NumericValue
                exp = value(self._args[1])
                if exp == int(exp):
                    if l is not None and exp > 0:
                        return l * exp
                    elif exp == 0:
                        return 0
            except:
                pass
        return None

    def _is_fixed(self, args):
        assert(len(args) == 2)
        if not args[1]:
            return False
        return args[0] or isclose(value(self._args[1]), 0)

    def _precedence(self):
        return _PowExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        return _l ** _r

    def getname(self, *args, **kwds):
        return 'pow'

    def _inline_operator(self):
        return '**'


class _NPV_PowExpression(_PowExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


class _LinearOperatorExpression(_ExpressionBase):
    """An 'abstract' class that defines the polynomial degree for a simple
    linear operator
    """

    __slots__ = ()

    def _polynomial_degree(self, result):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        ans = 0
        for x in result:
            if x is None:
                return None
            elif ans < x:
                ans = x
        return ans


class _InequalityExpression(_LinearOperatorExpression):
    """An object that defines a series of less-than or
    less-than-or-equal expressions"""

    __slots__ = ('_strict', '_cloned_from')
    PRECEDENCE = 9

    # Used to process chained inequalities
    chainedInequality = None
    call_info = None

    def __init__(self, args, strict, cloned_from):
        """Constructor"""
        super(_InequalityExpression,self).__init__(args)
        self._strict = strict
        self._cloned_from = cloned_from

    def nargs(self):
        return len(self._args)

    def _clone(self, args, memo):
        return self.__class__(args, self._strict, self._cloned_from)

    def __getstate__(self):
        result = super(_InequalityExpression, self).__getstate__()
        for i in _InequalityExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def __nonzero__(self):
        if _InequalityExpression.chainedInequality is not None:     #pragma: no cover
            raise TypeError(chainedInequalityErrorMessage())
        if not self.is_constant() and len(self._args) == 2:
            _InequalityExpression.call_info = traceback.extract_stack(limit=2)[-2]
            _InequalityExpression.chainedInequality = self
            #return bool(self())                - This is needed to apply simple evaluation of inequalities
            return True

        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return _InequalityExpression.PRECEDENCE

    def _apply_operation(self, result):
        for i, a in enumerate(result):
            if not i:
                pass
            elif self._strict[i-1]:
                if not _l < a:
                    return False
            else:
                if not _l <= a:
                    return False
            _l = a
        return True

    def _to_string_prefix(self, ostream, verbose):
        pass

    def _to_string_infix(self, ostream, idx, verbose):
        ostream.write( '  <  ' if self._strict[idx-1] else '  <=  ' )

    def is_constant(self):
        return self._args[0].is_constant() and self._args[1].is_constant()

    def _potentially_variable(self):
        return self._args[0]._potentially_variable() or self._args[1]._potentially_variable()


class _EqualityExpression(_LinearOperatorExpression):
    """An object that defines a equal-to expression"""

    __slots__ = ()
    PRECEDENCE = 9

    def nargs(self):
        return len(self._args)

    def __nonzero__(self):
        if _InequalityExpression.chainedInequality is not None:         #pragma: no cover
            raise TypeError(chainedInequalityErrorMessage())
        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return _EqualityExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        return _l == _r

    def _to_string_prefix(self, ostream, verbose):
        pass

    def _to_string_infix(self, ostream, idx, verbose):
        ostream.write('  ==  ' )

    def is_constant(self):
        return self._args[0].is_constant() and self._args[1].is_constant()

    def _potentially_variable(self):
        return self._args[0]._potentially_variable() or self._args[1]._potentially_variable()


class _ProductExpression(_ExpressionBase):
    """An object that defines a product expression"""

    __slots__ = ()
    PRECEDENCE = 4

    def _precedence(self):
        return _ProductExpression.PRECEDENCE

    def _polynomial_degree(self, result):
        # NB: We can't use sum() here because None (non-polynomial)
        # overrides a numeric value (and sum() just ignores it - or
        # errors in py3k)
        a, b = result
        if a is None or b is None:
            return None
        else:
            return a + b

    def getname(self, *args, **kwds):
        return 'prod'

    def _inline_operator(self):
        return '*'

    def _apply_operation(self, result):
        _l, _r = result
        return _l * _r


class _NPV_ProductExpression(_ProductExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


class _ReciprocalExpression(_ExpressionBase):
    """An object that defines a division expression"""

    __slots__ = ()
    PRECEDENCE = 3.5

    def nargs(self):
        return 1

    def _precedence(self):
        return _ReciprocalExpression.PRECEDENCE

    def _polynomial_degree(self, result):
        if result[0] is 0:
            return 0
        return None

    def getname(self, *args, **kwds):
        return 'recip'

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.getname())
        else:
            ostream.write("(1/")

    def _to_string_suffix(self, ostream, verbose):
        ostream.write(")")

    def _apply_operation(self, result):
        return 1 / result[0]


class _NPV_ReciprocalExpression(_ReciprocalExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


class _SumExpression(_LinearOperatorExpression):
    """An object that defines a simple summation of expressions"""

    __slots__ = ()
    PRECEDENCE = 6

    def _precedence(self):
        return _SumExpression.PRECEDENCE

    def _to_string_infix(self, ostream, idx, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            if self._args[idx].__class__ is _NegationExpression:
                ostream.write(' - ')
                return True
            else:
                ostream.write(' + ')

    def _apply_operation(self, result):
        l_, r_ = result
        return l_ + r_


class _NPV_SumExpression(_SumExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


class _ViewSumExpression(_SumExpression):
    """An object that defines a summation with 1 or more terms using a shared list."""

    __slots__ = ('_nargs',)
    PRECEDENCE = 6

    def __init__(self, args):
        self._args = args
        self._is_owned = False
        self._nargs = len(self._args)

    def add(self, new_arg):
        if new_arg.__class__ in native_numeric_types and isclose(new_arg,0):
            return self
        # Clone 'self', because _ViewSumExpression are immutable
        self._is_owned = True
        self = self.__class__(self._args)
        #
        if new_arg.__class__ is _ViewSumExpression or new_arg.__class__ is _MutableViewSumExpression:
            self._args.extend( islice(new_arg._args, new_arg._nargs) )
        elif not new_arg is None:
            self._args.append(new_arg)
        self._nargs = len(self._args)
        if new_arg.__class__ in pyomo5_expression_types:
            new_arg._is_owned = True
        return self

    def nargs(self):
        return self._nargs

    def _precedence(self):
        return _ViewSumExpression.PRECEDENCE

    def _apply_operation(self, result):
        return sum(result)

    def _clone(self, args, memo):
        return self.__class__(list(args))

    def __getstate__(self):
        result = super(_ViewSumExpression, self).__getstate__()
        for i in _ViewSumExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):
        return 'viewsum'

    def is_constant(self):
        return False
        #
        # In most normal contexts, a _ViewSumExpression is non-constant.  When
        # Forming expressions, constant parameters are turned into numbers, which
        # are simply added.  Mutable parameters, variables and expressions are 
        # not constant.
        #
        #for v in islice(self._args, self._nargs):
        #    if not (v.__class__ in native_numeric_types or v.is_constant()):
        #        return False
        #return True

    def _potentially_variable(self):
        for v in islice(self._args, self._nargs):
            if v.__class__ in pyomo5_variable_types:
                return True
            if not v.__class__ in native_numeric_types and v._potentially_variable():
                return True
        return False

    def _to_string_skip(self, _idx):
        return  _idx == 0 and \
                self._args[0].__class__ in native_numeric_types and \
                isclose(self._args[0], 0)


class _MutableViewSumExpression(_ViewSumExpression):

    __slots__ = ()

    def add(self, new_arg):
        if new_arg.__class__ in native_numeric_types and isclose(new_arg,0):
            return self
        # Do not clone 'self', because _MutableViewSumExpression are mutable
        #self._is_owned = True
        #self = self.__class__(list(self.args))
        #
        if new_arg.__class__ is _ViewSumExpression or new_arg.__class__ is _MutableViewSumExpression:
            self._args.extend( islice(new_arg._args, new_arg._nargs) )
        elif not new_arg is None:
            self._args.append(new_arg)
        self._nargs = len(self._args)
        if new_arg.__class__ in pyomo5_expression_types:
            new_arg._is_owned = True
        return self


class _GetItemExpression(_ExpressionBase):
    """Expression to call "__getitem__" on the base"""

    __slots__ = ('_base',)
    PRECEDENCE = 1

    def _precedence(self):
        return _GetItemExpression.PRECEDENCE

    def __init__(self, args, base=None):
        """Construct an expression with an operation and a set of arguments"""
        self._args = args
        self._base = base
        self._is_owned = False
        for arg in args:
            if arg.__class__ in pyomo5_expression_types:
                arg._is_owned = True

    def nargs(self):
        return len(self._args)

    def _clone(self, args, memo):
        return self.__class__(args, self._base)

    def __getstate__(self):
        result = super(_GetItemExpression, self).__getstate__()
        for i in _GetItemExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):
        return self._base.getname(*args, **kwds)

    def _potentially_variable(self):
        if any(arg._potentially_variable() for arg in self._args if not arg.__class__ in native_types):
            for x in itervalues(self._base):
                if not x.__class__ in native_types and x._potentially_variable():
                    return True
        return False
        
    def is_fixed(self):
        if any(self._args):
            for x in itervalues(self._base):
                if not x.__class__ in native_types and not x.is_fixed():
                    return False
        return True
        
    def _polynomial_degree(self, result):
        if any(x != 0 for x in result):
            return None
        ans = 0
        for x in itervalues(self._base):
            if x.__class__ in native_types:
                continue
            tmp = x.polynomial_degree()
            if tmp is None:
                return None
            elif tmp > ans:
                ans = tmp
        return ans

    def _apply_operation(self, result):
        return value(self._base.__getitem__( tuple(result) ))

    def _to_string_prefix(self, ostream, verbose):
        ostream.write(self.name)

    def resolve_template(self):
        return self._base.__getitem__(tuple(value(i) for i in self._args))


class Expr_if(_ExpressionBase):
    """An object that defines a dynamic if-then-else expression"""

    __slots__ = ('_if','_then','_else')

    # **NOTE**: This class evaluates the branching "_if" expression
    #           on a number of occasions. It is important that
    #           one uses __call__ for value() and NOT bool().

    def __init__(self, IF_=None, THEN_=None, ELSE_=None):
        """Constructor"""
        
        if type(IF_) is tuple and THEN_==None and ELSE_==None:
            IF_, THEN_, ELSE_ = IF_
        self._args = (IF_, THEN_, ELSE_)
        self._if = IF_
        self._then = THEN_
        self._else = ELSE_
        if self._if.__class__ in native_types:
            self._if = as_numeric(self._if)
        self._is_owned = False
        if IF_.__class__ in pyomo5_expression_types:
            IF_._is_owned = True
        if THEN_.__class__ in pyomo5_expression_types:
            THEN_._is_owned = True
        if ELSE_.__class__ in pyomo5_expression_types:
            ELSE_._is_owned = True

    def nargs(self):
        return 3

    def __getstate__(self):
        state = super(Expr_if, self).__getstate__()
        for i in Expr_if.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return "Expr_if"

    def _is_fixed(self, args):
        assert(len(args) == 3)
        if args[0]: #self._if.is_constant():
            if self._if():
                return args[1] #self._then.is_constant()
            else:
                return args[2] #self._else.is_constant()
        else:
            return False

    def is_constant(self):
        if self._if.__class__ in native_numeric_types or self._if.is_constant():
            if value(self._if):
                return (self._then.__class__ in native_numeric_types or self._then.is_constant())
            else:
                return (self._else.__class__ in native_numeric_types or self._else.is_constant())
        else:
            return (self._then.__class__ in native_numeric_types or self._then.is_constant()) and (self._else.__class__ in native_numeric_types or self._else.is_constant())

    def _potentially_variable(self):
        return (not self._if.__class__ in native_numeric_types and self._if._potentially_variable()) or (not self._then.__class__ in native_numeric_types and self._then._potentially_variable()) or (not self._if.__class__ in native_numeric_types and self._else._potentially_variable())

    def _polynomial_degree(self, result):
        _if, _then, _else = result
        if _if == 0:
            try:
                return _then if self._if() else _else
            except ValueError:
                pass
        return None

    def _to_string_term(self, ostream, _idx, _sub, _name_buffer, verbose):
        ostream.write("%s=( " % ('if','then','else')[_idx], )
        if type(self._args[_idx]) in native_numeric_types:
            ostream.write(str(self._args[_idx]))
        else:
            self._args[_idx].to_string(ostream=ostream, verbose=verbose)
        ostream.write(" )")

    def _to_string_prefix(self, ostream, verbose):
        ostream.write(self.getname())

    def _to_string_infix(self, ostream, idx, verbose):
        ostream.write(", ")

    def _apply_operation(self, result):
        _if, _then, _else = result
        return _then if _if else _else


class _UnaryFunctionExpression(_ExpressionBase):
    """An object that defines a mathematical expression that can be evaluated"""

    # TODO: Unary functions should define their own subclasses so as to
    # eliminate the need for the fcn and name slots
    __slots__ = ('_fcn', '_name')

    def __init__(self, args, name=None, fcn=None):
        """Construct an expression with an operation and a set of arguments"""
        if not type(args) is tuple:
            args = (args,)
        self._args = args
        self._name = name
        self._fcn = fcn
        self._is_owned = False
        if args[0].__class__ in pyomo5_expression_types:
            args[0]._is_owned = True

    def nargs(self):
        return 1

    def _clone(self, args, memo):
        return self.__class__(args, self._name, self._fcn)

    def __getstate__(self):
        result = super(_UnaryFunctionExpression, self).__getstate__()
        for i in _UnaryFunctionExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):
        return self._name

    def _to_string_prefix(self, ostream, verbose):
        ostream.write(self.getname())

    def _polynomial_degree(self, result):
        if result[0] is 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        return self._fcn(result[0])


class _NPV_UnaryFunctionExpression(_UnaryFunctionExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


# NOTE: This should be a special class, since the expression generation relies
# on the Python __abs__ method.
class _AbsExpression(_UnaryFunctionExpression):

    __slots__ = ()

    def __init__(self, arg):
        super(_AbsExpression, self).__init__(arg, 'abs', abs)

    def _clone(self, args, memo):
        return self.__class__(args)


class _NPV_AbsExpression(_AbsExpression):
    __slots__ = ()

    def _potentially_variable(self):
        return False


class _LinearExpression(_ExpressionBase):
    __slots__ = ('constant',          # The constant term
                 'linear_coefs',      # Linear coefficients
                 'linear_vars')       # Linear variables

    PRECEDENCE = 6

    def __init__(self, args=None):
        self.constant = 0
        self.linear_coefs = []
        self.linear_vars = []
        self._args = tuple()
        self._is_owned = False

    def nargs(self):
        return 0

    def __getstate__(self):
        state = super(_LinearExpression, self).__getstate__()
        for i in _LinearExpression.__slots__:
           state[i] = getattr(self,i)
        return state

    def __deepcopy__(self, memo):
        return self._clone(None, memo)

    def _clone(self, args, memo):
        repn = self.__class__()
        repn.constant = deepcopy(self.constant, memo=memo)
        repn.linear_coefs = deepcopy(self.linear_coefs, memo=memo)
        repn.linear_vars = deepcopy(self.linear_vars, memo=memo)
        return repn

    def getname(self, *args, **kwds):
        return 'sum'

    def _polynomial_degree(self, result):
        return 1 if len(self.linear_vars) > 0 else 0

    def is_constant(self):
        return len(self.linear_vars) == 0

    def is_fixed(self):
        if len(self.linear_vars) == 0:
            return True
        for v in self.linear_vars:
            if v.fixed:
                return True
        return False

    def _to_string_suffix(self, ostream, verbose):
        ostream.write("(%f" % self.constant)
        for c,v in zip(self.linear_coefs, self.linear_vars):
            if c.__class__ in native_numeric_types and isclose(value(c),1):
                ostream.write(" + %s" % str(v))
            else:
                ostream.write(" + %s*%s" % (str(c),  str(v)))

    def _potentially_variable(self):
        return len(self.linear_vars) > 0

    def _apply_operation(self, result):
        return self.constant + sum(c*v.value for c,v in zip(self.linear_coefs, self.linear_vars))

    #@profile
    @staticmethod
    def _decompose_term(expr):
        if expr.__class__ in native_numeric_types:
            return expr,None,None
        elif expr.__class__ in _LinearExpression.vtypes:
            return 0,1,expr
        elif not expr._potentially_variable():
            return expr,None,None
        elif expr.__class__ is _ProductExpression:
            if expr._args[0].__class__ in native_numeric_types or not expr._args[0]._potentially_variable():
                C,c,v = expr._args[0],None,None
            else:
                C,c,v = _LinearExpression._decompose_term(expr._args[0])
            if expr._args[1].__class__ in _LinearExpression.vtypes:
                v_ = expr._args[1]
                if not v is None:
                    raise ValueError("Expected a single linear term (1)")
                return 0,C,v_
            else:
                C_,c_,v_ = _LinearExpression._decompose_term(expr._args[1])
            if not v_ is None:
                if not v is None:
                    raise ValueError("Expected a single linear term (2)")
                return C*C_,C*c_,v_
            return C_*C,C_*c,v
        #
        # A potentially variable _SumExpression class has been supplanted by
        # _ViewSumExpression
        #
        #elif expr.__class__ is _SumExpression:
        #    C,c,v = _LinearExpression._decompose_term(expr._args[0])
        #    C_,c_,v_ = _LinearExpression._decompose_term(expr._args[1])
        #    if not v_ is None:
        #        if not v is None:
        #            if id(v) == id(v_):
        #                return C+C_,c+c_,v
        #            else:
        #                raise ValueError("Expected a single linear term (3)")
        #        return C+C_,c_,v_
        #    return C+C_,c,v
        #
        # The _LinearViewSumExpression class is not used now
        #
        #elif expr.__class__ is _LinearViewSumExpression:
        #    C=0
        #    c=1
        #    v=None
        #    for arg in expr.args:
        #        if arg[1] is None:
        #            C += arg[0]
        #        elif not v is None:
        #            raise ValueError("Expected a single linear term (3a)")
        #        else:
        #            c=arg[0]
        #            v=arg[1]
        #    return C,c,v
        elif expr.__class__ is _NegationExpression:
            C,c,v = _LinearExpression._decompose_term(expr._args[0])
            return -C,-c,v
        elif expr.__class__ is _ReciprocalExpression:
            if expr._potentially_variable():
                raise ValueError("Unexpected nonlinear term (4)")
            return 1/expr,None,None
        elif expr.__class__ is _LinearExpression:
            l = len(expr.linear_vars)
            if l == 0:
                return expr.constant, None, None
            elif l == 1:
                return expr.constant, expr.linear_coefs[0], expr.linear_vars[0]
            else:
                raise ValueError("Expected a single linear term (5)")
        elif expr.__class__ is _ViewSumExpression or expr.__class__ in _MutableViewSumExpression:
            C = 0
            c = None
            v = None
            for e in expr._args:
                C_,c_,v_ = _LinearExpression._decompose_term(e)
                C += C_
                if not v_ is None:
                    if not v is None:
                        raise ValueError("Expected a single linear term (6)")
                    c=c_
                    v=v_
            return C,c,v
        #
        # TODO: ExprIf, POW, Abs?
        #
        raise ValueError("Unexpected nonlinear term (7)")

    #@profile
    def _combine_expr(self, etype, _other):
        if etype == _add or etype == _sub or etype == -_add or etype == -_sub:
            #
            # if etype == _sub,  then _LinearExpression - VAL
            # if etype == -_sub, then VAL - _LinearExpression
            #
            if etype == _sub:
                omult = -1
            else:
                omult = 1
            if etype == -_sub:
                self.constant *= -1
                for i,c in enumerate(self.linear_coefs):
                    self.linear_coefs[i] = -c

            if _other.__class__ in native_numeric_types or not _other._potentially_variable():
                self.constant = self.constant + omult * _other
            elif _other.__class__ is _LinearExpression:
                self.constant = self.constant + omult * _other.constant
                for c,v in zip(_other.linear_coefs, _other.linear_vars):
                    self.linear_coefs.append(omult*c)
                    self.linear_vars.append(v)
            elif _other.__class__ is _ViewSumExpression or _other.__class__ is _MutableViewSumExpression:
                for e in _other._args:
                    C,c,v = _LinearExpression._decompose_term(e)
                    self.constant = self.constant + omult * C
                    if not v is None:
                        self.linear_coefs.append(omult*c)
                        self.linear_vars.append(v)
            else:
                C,c,v = _LinearExpression._decompose_term(_other)
                self.constant = self.constant + omult * C
                if not v is None:
                    self.linear_coefs.append(omult*c)
                    self.linear_vars.append(v)

        elif etype == _mul or etype == -_mul:
            if _other.__class__ in native_numeric_types:
                multiplier = _other
            elif _other._potentially_variable():
                if len(self.linear_vars) > 0:
                    raise ValueError("Cannot multiply a linear expression with a variable expression")
                #
                # The linear expression is a constant, so re-initialize it with
                # a single term that multiplies the expression by the constant value.
                #
                C,c,v = _LinearExpression._decompose_term(_other)
                self.constant = C*self.constant
                self.linear_vars.append(v)
                self.linear_coefs.append(c*self.constant)
                return self
            else:
                multiplier = _other

            if multiplier.__class__ in native_numeric_types and isclose(multiplier, 0.0):
                self.constant = 0
                self.linear_vars = []
                self.linear_coefs = []
            else:
                self.constant *= multiplier
                for i,c in enumerate(self.linear_coefs):
                    self.linear_coefs[i] = c*multiplier

        elif etype == _div:
            if _other.__class__ in native_numeric_types:
                divisor = _other
            elif self._potentially_variable():
                raise ValueError("Unallowed operation on linear expression: division with a variable RHS")
            else:
                divisor = _other
            self.constant /= divisor
            for i,c in enumerate(self.linear_coefs):
                self.linear_coefs[i] = c/divisor

        elif etype == -_div:
            if self._potentially_variable():
                raise ValueError("Unallowed operation on linear expression: division with a variable RHS")
            C,c,v = _LinearExpression._decompose_term(_other)
            self.constant = C/self.constant
            if not v is None:
                self.linear_var = [v]
                self.linear_coef = [c/self.constant]
            
        elif etype == _neg:
            self.constant *= -1
            for i,c in enumerate(self.linear_coefs):
                self.linear_coefs[i] = - c

        else:
            raise ValueError("Unallowed operation on mutable linear expression: %d" % etype)

        return self


class _StaticLinearExpression(_LinearExpression):
    __slots__ = ()


#-------------------------------------------------------
#
# Functions used to generate expressions
#
#-------------------------------------------------------

def decompose_term(term):
    if term.__class__ in native_numeric_types or not term._potentially_variable():
        return True, [(term,None)]
    elif term.__class__ in pyomo5_variable_types:
        return True, [(1,term)]
    else:
        try:
            terms = [t_ for t_ in _decompose_terms(term)]
            return True, terms
        except ValueError:
            return False, None


def _decompose_terms(expr, multiplier=1):
    if expr.__class__ in native_numeric_types or not expr._potentially_variable():
        yield (multiplier*expr,None)
    elif expr.__class__ in pyomo5_variable_types:
        yield (multiplier,expr)
    elif expr.__class__ is _ProductExpression:
        if expr._args[0].__class__ in native_numeric_types or not expr._args[0]._potentially_variable():
            for term in _decompose_terms(expr._args[1], multiplier*expr._args[0]):
                yield term
        else:
            raise ValueError("Quadratic terms exist in a product expression.")
    elif expr.__class__ is _ReciprocalExpression:
        # The argument is potentially variable, so this represents a nonlinear term
        #
        # NOTE: We're ignoring possible simplifications 
        raise ValueError("Unexpected nonlinear term")
    elif expr.__class__ is _ViewSumExpression or expr.__class__ is _MutableViewSumExpression:
        for arg in expr.args:
            for term in _decompose_terms(arg, multiplier):
                yield term
    elif expr.__class__ is _NegationExpression:
        for term in  _decompose_terms(expr._args[0], -multiplier):
            yield term
    elif expr.__class__ is _StaticLinearExpression or expr.__class__ is _LinearExpression:
        if expr.constant.__class__ in native_numeric_types and not isclose(expr.constant,0):
            yield (multiplier*expr.constant,None)
        if len(expr.linear_coefs) > 0:
            for c,v in zip(expr.linear_coefs, expr.linear_vars):
                yield (multiplier*c,v)
    else:
        raise ValueError("Unexpected nonlinear term")   #pragma: no cover


def _process_arg(obj):
    #if False and obj.__class__ is _ViewSumExpression or obj.__class__ is _MutableViewSumExpression:
    #    if ignore_entangled_expressions.detangle[-1] and obj._is_owned:
            #
            # If the viewsum expression is owned, then we need to
            # clone it to avoid creating an entangled expression.
            #
            # But we don't have to worry about entanglement amongst other immutable
            # expression objects.
            #
    #        return clone_expression( obj, clone_leaves=False )
    #    return obj

    #if obj.is_expression():
    #    return obj

    if obj.__class__ is NumericConstant:
        return value(obj)

    if (obj.__class__ is _ParamData or obj.__class__ is SimpleParam) and not obj._component()._mutable:
        if not obj._constructed:
            return obj
        if obj.value is None:
            return obj
        return obj.value

    if obj.is_indexed():
        raise TypeError(
                "Argument for expression is an indexed numeric "
                "value\nspecified without an index:\n\t%s\nIs this "
                "value defined over an index that you did not specify?"
                % (obj.name, ) )

    return obj


#@profile
def generate_sum_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    if _self.__class__ is _LinearExpression:
        if etype >= _unary:
            return _self._combine_expr(etype, None)
        if _other.__class__ is not _LinearExpression:
            if not (_other.__class__ in native_types or _other.is_expression()):
                _other = _process_arg(_other)
        return _self._combine_expr(etype, _other)
    elif _other.__class__ is _LinearExpression:
        if not (_self.__class__ in native_types or _self.is_expression()):
            _self = _process_arg(_self)
        return _other._combine_expr(-etype, _self)

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression()):
        _self = _process_arg(_self)

    if etype == _neg:
        if _self.__class__ in native_numeric_types:
            return - _self
        elif _self._potentially_variable():
            if _self.__class__ is _NegationExpression:
                return _self._args[0]
            return _NegationExpression((_self,))
        else:
            if _self.__class__ is _NPV_NegationExpression:
                return _self._args[0]
            return _NPV_NegationExpression((_self,))

    if not (_other.__class__ in native_types or _other.is_expression()):
        _other = _process_arg(_other)

    if etype < 0:
        #
        # This may seem obvious, but if we are performing an
        # "R"-operation (i.e. reverse operation), then simply reverse
        # self and other.  This is legitimate as we are generating a
        # completely new expression here.
        #
        etype *= -1
        _self, _other = _other, _self

    if etype == _add:
        #
        # x + y
        #
        if (_self.__class__ is _ViewSumExpression and not _self._is_owned) or \
           _self.__class__ is _MutableViewSumExpression:
           #(_self.__class__ is _LinearViewSumExpression and not _self._is_owned) or
            return _self.add(_other)
        elif (_other.__class__ is _ViewSumExpression and not _other._is_owned) or \
            _other.__class__ is _MutableViewSumExpression:
            #_other.__class__ is _LinearViewSumExpression and not _other._is_owned or
            return _other.add(_self)
        elif _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                return _self + _other
            elif _other == 0:   #isclose(_other, 0):
                return _self
            if _self._potentially_variable():
                #return _LinearViewSumExpression((_other, _self))
                #return _ViewSumExpression([_other, _self])
                return _ViewSumExpression([_self, _other])
            return _NPV_SumExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:      #isclose(_self, 0):
                return _other
            if _other._potentially_variable():
                #return _LinearViewSumExpression((_self, _other))
                return _ViewSumExpression([_self, _other])
            return _NPV_SumExpression((_self, _other))
        elif _other._potentially_variable():
            #return _LinearViewSumExpression((_self, _other))
            return _ViewSumExpression([_self, _other])
        elif _self._potentially_variable():
            #return _LinearViewSumExpression((_other, _self))
            #return _ViewSumExpression([_other, _self])
            return _ViewSumExpression([_self, _other])
        else:
            return _NPV_SumExpression((_self, _other))

    elif etype == _sub:
        #
        # x - y
        #
        # TODO: _MultiViewSum logic here
        #
        if (_self.__class__ is _ViewSumExpression and not _self._is_owned) or \
           _self.__class__ is _MutableViewSumExpression:
           #(_self.__class__ is _LinearViewSumExpression and not _self._is_owned) or
            return _self.add(-_other)
        elif _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                return _self - _other
            elif isclose(_other, 0):
                return _self
            if _self._potentially_variable():
                #return _LinearViewSumExpression((_self, -_other))
                return _ViewSumExpression([_self, -_other])
            return _NPV_SumExpression((_self, -_other))
        elif _self.__class__ in native_numeric_types:
            if isclose(_self, 0):
                if _other._potentially_variable():
                    return _NegationExpression((_other,))
                return _NPV_NegationExpression((_other,))
            if _other._potentially_variable():    
                #return _LinearViewSumExpression((_self, _NegationExpression((_other,))))
                return _ViewSumExpression([_self, _NegationExpression((_other,))])
            return _NPV_SumExpression((_self, _NPV_NegationExpression((_other,))))
        elif _other._potentially_variable():    
            #return _LinearViewSumExpression((_self, _NegationExpression((_other,))))
            return _ViewSumExpression([_self, _NegationExpression((_other,))])
        elif _self._potentially_variable():
            #return _LinearViewSumExpression((_self, _NPV_NegationExpression((_other,))))
            return _ViewSumExpression([_self, _NPV_NegationExpression((_other,))])
        else:
            return _NPV_SumExpression((_self, _NPV_NegationExpression((_other,))))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover
        

#@profile
def generate_mul_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    if _self.__class__ is _LinearExpression:
        if _other.__class__ is not _LinearExpression:
            if not (_other.__class__ in native_types or _other.is_expression()):
                _other = _process_arg(_other)
        return _self._combine_expr(etype, _other)
    elif _other.__class__ is _LinearExpression:
        if not (_self.__class__ in native_types or _self.is_expression()):
            _self = _process_arg(_self)
        return _other._combine_expr(-etype, _self)

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression()):
        _self = _process_arg(_self)

    if not (_other.__class__ in native_types or _other.is_expression()):
        _other = _process_arg(_other)

    if etype < 0:
        #
        # This may seem obvious, but if we are performing an
        # "R"-operation (i.e. reverse operation), then simply reverse
        # self and other.  This is legitimate as we are generating a
        # completely new expression here.
        #
        etype *= -1
        _self, _other = _other, _self

    if etype == _mul:
        #
        # x * y
        #
        if _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                return _self * _other
            elif _other == 0:   # isclose(_other, 0)
                return 0
            elif _other == 1:
                return _self
            if _self._potentially_variable():
                return _ProductExpression((_other, _self))
            return _NPV_ProductExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:  # isclose(_self, 0)
                return 0
            elif _self == 1:
                return _other
            if _other._potentially_variable():
                return _ProductExpression((_self, _other))
            return _NPV_ProductExpression((_self, _other))
        elif _other._potentially_variable():
            return _ProductExpression((_self, _other))
        elif _self._potentially_variable():
            return _ProductExpression((_other, _self))
        else:
            return _NPV_ProductExpression((_self, _other))

    elif etype == _div:
        #
        # x / y
        #
        if _other.__class__ in native_numeric_types:
            if _other == 1:
                return _self
            elif not _other:
                raise ZeroDivisionError()
            elif _self.__class__ in native_numeric_types:
                return _self / _other
            if _self._potentially_variable():
                return _ProductExpression((1/_other, _self))
            return _NPV_ProductExpression((1/_other, _self))
        elif _self.__class__ in native_numeric_types:
            if isclose(_self, 0):
                return 0
            elif _self == 1:
                if _other._potentially_variable():
                    return _ReciprocalExpression((_other,))
                return _NPV_ReciprocalExpression((_other,))
            elif _other._potentially_variable():
                return _ProductExpression((_self, _ReciprocalExpression((_other,))))
            return _NPV_ProductExpression((_self, _ReciprocalExpression((_other,))))
        elif _other._potentially_variable():
            return _ProductExpression((_self, _ReciprocalExpression((_other,))))
        elif _self._potentially_variable():
            return _ProductExpression((_NPV_ReciprocalExpression((_other,)), _self))
        else:
            return _NPV_ProductExpression((_self, _NPV_ReciprocalExpression((_other,))))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover


#@profile
def generate_other_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression()):
        _self = _process_arg(_self)

    #
    # abs(x)
    #
    if etype == _abs:
        if _self.__class__ in native_numeric_types:
            return abs(_self)
        elif _self._potentially_variable():
            return _AbsExpression(_self)
        else:
            return _NPV_AbsExpression(_self)

    if not (_other.__class__ in native_types or _other.is_expression()):
        _other = _process_arg(_other)

    if etype < 0:
        #
        # This may seem obvious, but if we are performing an
        # "R"-operation (i.e. reverse operation), then simply reverse
        # self and other.  This is legitimate as we are generating a
        # completely new expression here.
        #
        etype *= -1
        _self, _other = _other, _self

    if etype == _pow:
        if _other.__class__ in native_numeric_types:
            if _other == 1:
                return _self
            elif not _other:
                return 1
            elif _self.__class__ in native_numeric_types:
                return _self ** _other
            elif _self._potentially_variable():
                return _PowExpression((_self, _other))
            return _NPV_PowExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _other._potentially_variable():
                return _PowExpression((_self, _other))
            return _NPV_PowExpression((_self, _other))
        elif _self._potentially_variable() or _other._potentially_variable():
            return _PowExpression((_self, _other))
        else:
            return _NPV_PowExpression((_self, _other))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover


def generate_relational_expression(etype, lhs, rhs):
    # We cannot trust Python not to recycle ID's for temporary POD data
    # (e.g., floats).  So, if it is a "native" type, we will record the
    # value, otherwise we will record the ID.  The tuple for native
    # types is to guarantee that a native value will *never*
    # accidentally match an ID
    cloned_from = (
        id(lhs) if lhs.__class__ not in native_numeric_types else (0,lhs),
        id(rhs) if rhs.__class__ not in native_numeric_types else (0,rhs)
    )
    rhs_is_relational = False
    lhs_is_relational = False

    #
    # TODO: It would be nice to reduce all Constants to literals (and
    # not carry around the overhead of the NumericConstants). For
    # consistency, we will not do that yet, as many things downstream
    # would break; in particular within Constraint.add.  This way, all
    # arguments in the relational Expression's _args will be guaranteed
    # to be NumericValues (just as they are for all other Expressions).
    #
    if not (lhs.__class__ in native_types or lhs.is_expression()):
        lhs = _process_arg(lhs)
    if not (rhs.__class__ in native_types or rhs.is_expression()):
        rhs = _process_arg(rhs)

    if lhs.__class__ in native_numeric_types:
        lhs = as_numeric(lhs)
    elif lhs.is_relational():
        lhs_is_relational = True

    if rhs.__class__ in native_numeric_types:
        rhs = as_numeric(rhs)
    elif rhs.is_relational():
        rhs_is_relational = True

    if _InequalityExpression.chainedInequality is not None:
        prevExpr = _InequalityExpression.chainedInequality
        match = []
        # This is tricky because the expression could have been posed
        # with >= operators, so we must figure out which arguments
        # match.  One edge case is when the upper and lower bounds are
        # the same (implicit equality) - in which case *both* arguments
        # match, and this should be converted into an equality
        # expression.
        for i,arg in enumerate(prevExpr._cloned_from):
            if arg == cloned_from[0]:
                match.append((i,0))
            elif arg == cloned_from[1]:
                match.append((i,1))
        if etype == _eq:
            raise TypeError(chainedInequalityErrorMessage())
        if len(match) == 1:
            if match[0][0] == match[0][1]:
                raise TypeError(chainedInequalityErrorMessage(
                    "Attempting to form a compound inequality with two "
                    "%s bounds" % ('lower' if match[0][0] else 'upper',)))
            if not match[0][1]:
                cloned_from = prevExpr._cloned_from + (cloned_from[1],)
                lhs = prevExpr
                lhs_is_relational = True
            else:
                cloned_from = (cloned_from[0],) + prevExpr._cloned_from
                rhs = prevExpr
                rhs_is_relational = True
        elif len(match) == 2:
            # Special case: implicit equality constraint posed as a <= b <= a
            if prevExpr._strict[0] or etype == _lt:
                _InequalityExpression.chainedInequality = None
                buf = StringIO()
                prevExpr.to_string(buf)
                raise TypeError("Cannot create a compound inequality with "
                      "identical upper and lower\n\tbounds using strict "
                      "inequalities: constraint infeasible:\n\t%s and "
                      "%s < %s" % ( buf.getvalue().strip(), lhs, rhs ))
            if match[0] == (0,0):
                # This is a particularly weird case where someone
                # evaluates the *same* inequality twice in a row.  This
                # should always be an error (you can, for example, get
                # it with "0 <= a >= 0").
                raise TypeError(chainedInequalityErrorMessage())
            etype = _eq
        else:
            raise TypeError(chainedInequalityErrorMessage())
        _InequalityExpression.chainedInequality = None

    if etype == _eq:
        if lhs_is_relational or rhs_is_relational:
            buf = StringIO()
            if lhs_is_relational:
                lhs.to_string(buf)
            else:
                rhs.to_string(buf)
            raise TypeError("Cannot create an EqualityExpression where "\
                  "one of the sub-expressions is a relational expression:\n"\
                  "    " + buf.getvalue().strip())
        ans = _EqualityExpression((lhs,rhs))
        return ans
    else:
        if etype == _le:
            strict = (False,)
        elif etype == _lt:
            strict = (True,)
        else:
            raise ValueError("Unknown relational expression type '%s'" % etype)
        if lhs_is_relational:
            if lhs.__class__ is _InequalityExpression:
                if rhs_is_relational:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where both sub-expressions are also relational "\
                          "expressions (we support no more than 3 terms "\
                          "in an inequality expression).")
                if len(lhs._args) > 2:
                    raise ValueError("Cannot create an InequalityExpression "\
                          "with more than 3 terms.")
                lhs._args = lhs._args + (rhs,)
                lhs._strict = lhs._strict + strict
                lhs._cloned_from = cloned_from
                return lhs
            else:
                buf = StringIO()
                lhs.to_string(buf)
                raise TypeError("Cannot create an InequalityExpression "\
                      "where one of the sub-expressions is an equality "\
                      "expression:\n    " + buf.getvalue().strip())
        elif rhs_is_relational:
            if rhs.__class__ is _InequalityExpression:
                if len(rhs._args) > 2:
                    raise ValueError("Cannot create an InequalityExpression "\
                          "with more than 3 terms.")
                rhs._args = (lhs,) + rhs._args
                rhs._strict = strict + rhs._strict
                rhs._cloned_from = cloned_from
                return rhs
            else:
                buf = StringIO()
                rhs.to_string(buf)
                raise TypeError("Cannot create an InequalityExpression "\
                      "where one of the sub-expressions is an equality "\
                      "expression:\n    " + buf.getvalue().strip())
        else:
            ans = _InequalityExpression((lhs, rhs), strict, cloned_from)
            return ans


def generate_intrinsic_function_expression(arg, name, fcn):
    if not (arg.__class__ in native_types or arg.is_expression()):
        arg = _process_arg(arg)

    if arg.__class__ in native_types:
        return fcn(arg)
    elif arg._potentially_variable():
        return _UnaryFunctionExpression(arg, name, fcn)
    else:
        return _NPV_UnaryFunctionExpression(arg, name, fcn)


pyomo5_expression_types = set([
        _ExpressionBase,
        _NegationExpression,
        _NPV_NegationExpression,
        _ExternalFunctionExpression,
        _NPV_ExternalFunctionExpression,
        _PowExpression,
        _NPV_PowExpression,
        _LinearOperatorExpression,
        _InequalityExpression,
        _EqualityExpression,
        _ProductExpression,
        _NPV_ProductExpression,
        _ReciprocalExpression,
        _NPV_ReciprocalExpression,
        _SumExpression,
        _NPV_SumExpression,
        _ViewSumExpression,
        _GetItemExpression,
        Expr_if,
        _UnaryFunctionExpression,
        _NPV_UnaryFunctionExpression,
        _AbsExpression,
        _NPV_AbsExpression,
        _LinearExpression,
        _StaticLinearExpression
        ])
pyomo5_product_types = set([
        _ProductExpression,
        _NPV_ProductExpression
        ])
pyomo5_reciprocal_types = set([
        _ReciprocalExpression,
        _NPV_ReciprocalExpression
        ])
pyomo5_variable_types = None

