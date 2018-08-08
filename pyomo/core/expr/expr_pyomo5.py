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

_using_chained_inequality = True

#
# These symbols are part of pyomo.core.expr
#
_public = ['linear_expression', 'nonlinear_expression', 'inequality']
#
# These symbols are part of pyomo.core.expr.current
#
__all__ = (
'linear_expression',
'nonlinear_expression',
'inequality',
'decompose_term',
'clone_counter',
'clone_expression',
'FixedExpressionError',
'NonConstantExpressionError',
'evaluate_expression',
'identify_components',
'identify_variables',
'identify_mutable_parameters',
'expression_to_string',
'ExpressionBase',
'EqualityExpression',
'RangedExpression',
'InequalityExpression',
'ProductExpression',
'MonomialTermExpression',
'PowExpression',
'ExternalFunctionExpression',
'GetItemExpression',
'Expr_ifExpression',
'LinearExpression',
'ReciprocalExpression',
'NegationExpression',
'SumExpression',
'UnaryFunctionExpression',
'AbsExpression',
'NPV_NegationExpression',
'NPV_ExternalFunctionExpression',
'NPV_PowExpression',
'NPV_ProductExpression',
'NPV_ReciprocalExpression',
'NPV_SumExpression',
'NPV_UnaryFunctionExpression',
'NPV_AbsExpression',
'SimpleExpressionVisitor',
'ExpressionValueVisitor',
'ExpressionReplacementVisitor',
'LinearDecompositionError',
'SumExpressionBase',
'_MutableSumExpression',    # This should not be referenced, except perhaps while testing code
'_MutableLinearExpression',     # This should not be referenced, except perhaps while testing code
'_decompose_linear_terms',      # This should not be referenced, except perhaps while testing code
'_chainedInequality',           # This should not be referenced, except perhaps while testing code
'_using_chained_inequality',           # This should not be referenced, except perhaps while testing code
'_generate_sum_expression',                 # Only used within pyomo.core.expr
'_generate_mul_expression',                 # Only used within pyomo.core.expr
'_generate_other_expression',               # Only used within pyomo.core.expr
'_generate_intrinsic_function_expression',  # Only used within pyomo.core.expr
'_generate_relational_expression',          # Only used within pyomo.core.expr
)

import math
import logging
import sys
import traceback
from copy import deepcopy
from collections import deque
from itertools import islice
from six import next, string_types, itervalues
from six.moves import xrange, builtins
from weakref import ref

logger = logging.getLogger('pyomo.core')

from pyutilib.misc.visitor import SimpleVisitor, ValueVisitor
from pyutilib.math.util import isclose

from pyomo.common.deprecation import deprecation_warning
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.expr.numvalue import \
    (NumericValue,
     NumericConstant,
     native_types,
     nonpyomo_leaf_types,
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
from pyomo.core.expr.expr_errors import TemplateExpressionError


if _using_chained_inequality:               #pragma: no cover
    class _chainedInequality(object):

        prev = None
        call_info = None
        cloned_from = []

        @staticmethod
        def error_message(msg=None):
            if msg is None:
                msg = "Relational expression used in an unexpected Boolean context."
            val = _chainedInequality.prev.to_string()
            # We are about to raise an exception, so it's OK to reset chainedInequality
            info = _chainedInequality.call_info
            _chainedInequality.call_info = None
            _chainedInequality.prev = None

            args = ( str(msg).strip(), val.strip(), info[0], info[1],
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

else:                               #pragma: no cover
    _chainedInequality = None


class clone_counter(object):
    """ Context manager for counting cloning events.

    This context manager counts the number of times that the
    :func:`clone_expression <pyomo.core.expr.current.clone_expression>`
    function is executed.
    """

    _count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def count(self):
        """A property that returns the clone count value.
        """
        return clone_counter._count


class nonlinear_expression(object):
    """ Context manager for mutable sums.

    This context manager is used to compute a sum while
    treating the summation as a mutable object.
    """

    def __enter__(self):
        self.e = _MutableSumExpression([])
        return self.e

    def __exit__(self, *args):
        if self.e.__class__ == _MutableSumExpression:
            self.e.__class__ = SumExpression


class linear_expression(object):
    """ Context manager for mutable linear sums.

    This context manager is used to compute a linear sum while
    treating the summation as a mutable object.
    """

    def __enter__(self):
        """
        The :class:`_MutableLinearExpression <pyomo.core.expr.current._MutableLinearExpression>`
        class is the context that is used to to
        hold the mutable linear sum.
        """
        self.e = _MutableLinearExpression()
        return self.e

    def __exit__(self, *args):
        """
        The context is changed to the
        :class:`LinearExpression <pyomo.core.expr.current.LinearExpression>`
        class to transform the context into a nonmutable
        form.
        """
        if self.e.__class__ == _MutableLinearExpression:
            self.e.__class__ = LinearExpression


#-------------------------------------------------------
#
# Visitor Logic
#
#-------------------------------------------------------

class SimpleExpressionVisitor(object):
    """
    Note:
        This class is a customization of the PyUtilib :class:`SimpleVisitor
        <pyutilib.misc.visitor.SimpleVisitor>` class that is tailored
        to efficiently walk Pyomo expression trees.  However, this class
        is not a subclass of the PyUtilib :class:`SimpleVisitor
        <pyutilib.misc.visitor.SimpleVisitor>` class because all key methods
        are reimplemented.
    """

    def visit(self, node):  #pragma: no cover
        """
        Visit a node in an expression tree and perform some operation on
        it.

        This method should be over-written by a user
        that is creating a sub-class.

        Args:
            node: a node in an expression tree

        Returns:
            nothing
        """
        pass

    def finalize(self):     #pragma: no cover
        """
        Return the "final value" of the search.

        The default implementation returns :const:`None`, because
        the traditional visitor pattern does not return a value.

        Returns:
            The final value after the search.  Default is :const:`None`.
        """
        pass

    def xbfs(self, node):
        """
        Breadth-first search of an expression tree,
        except that leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs <pyutilib.misc.visitor.SimpleVisitor.xbfs>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
        dq = deque([node])
        while dq:
            current = dq.popleft()
            self.visit(current)
            #for c in self.children(current):
            for c in current.args:
                #if self.is_leaf(c):
                if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                    self.visit(c)
                else:
                    dq.append(c)
        return self.finalize()

    def xbfs_yield_leaves(self, node):
        """
        Breadth-first search of an expression tree, except that
        leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs_yield_leaves <pyutilib.misc.visitor.SimpleVisitor.xbfs_yield_leaves>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
        #
        # If we start with a leaf, then yield it and stop iteration
        #
        if node.__class__ in nonpyomo_leaf_types or not node.is_expression_type() or node.nargs() == 0:
            ans = self.visit(node)
            if not ans is None:
                yield ans
            return
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
                if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                    ans = self.visit(c)
                    if not ans is None:
                        yield ans
                else:
                    dq.append(c)


class ExpressionValueVisitor(object):
    """
    Note:
        This class is a customization of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class that is tailored
        to efficiently walk Pyomo expression trees.  However, this class
        is not a subclass of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class because all key methods
        are reimplemented.
    """

    def visit(self, node, values):  #pragma: no cover
        """
        Visit a node in a tree and compute its value using
        the values of its children.

        This method should be over-written by a user
        that is creating a sub-class.

        Args:
            node: a node in a tree
            values: a list of values of this node's children

        Returns:
            The *value* for this node, which is computed using :attr:`values`
        """
        pass

    def visiting_potential_leaf(self, node):    #pragma: no cover
        """
        Visit a node and return its value if it is a leaf.

        Note:
            This method needs to be over-written for a specific
            visitor application.

        Args:
            node: a node in a tree

        Returns:
            A tuple: ``(flag, value)``.   If ``flag`` is False,
            then the node is not a leaf and ``value`` is :const:`None`.
            Otherwise, ``value`` is the computed value for this node.
        """
        raise RuntimeError("The visiting_potential_leaf method needs to be defined.")

    def finalize(self, ans):    #pragma: no cover
        """
        This method defines the return value for the search methods
        in this class.

        The default implementation returns the value of the
        initial node (aka the root node), because
        this visitor pattern computes and returns value for each
        node to enable the computation of this value.

        Args:
            ans: The final value computed by the search method.

        Returns:
            The final value after the search. Defaults to simply
            returning :attr:`ans`.
        """
        return ans

    def dfs_postorder_stack(self, node):
        """
        Perform a depth-first search in postorder using a stack
        implementation.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`ValueVisitor.dfs_postorder_stack <pyutilib.misc.visitor.ValueVisitor.dfs_postorder_stack>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.
        """
        flag, value = self.visiting_potential_leaf(node)
        if flag:
            return value
        #_stack = [ (node, self.children(node), 0, len(self.children(node)), [])]
        _stack = [ (node, node._args_, 0, node.nargs(), [])]
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
                flag, value = self.visiting_potential_leaf(_sub)
                if flag:
                    _result.append( value )
                else:
                    #
                    # Push an expression onto the stack
                    #
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj                    = _sub
                    #_argList                = self.children(_sub)
                    _argList                = _sub._args_
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


class ExpressionReplacementVisitor(object):
    """
    Note:
        This class is a customization of the PyUtilib :class:`ValueVisitor
        <pyutilib.misc.visitor.ValueVisitor>` class that is tailored
        to support replacement of sub-trees in a Pyomo expression
        tree.  However, this class is not a subclass of the PyUtilib
        :class:`ValueVisitor <pyutilib.misc.visitor.ValueVisitor>`
        class because all key methods are reimplemented.
    """

    def __init__(self,
                 substitute=None,
                 descend_into_named_expressions=True,
                 remove_named_expressions=False):
        """
        Contruct a visitor that is tailored to support the
        replacement of sub-trees in a pyomo expression tree.

        Args:
            memo (dict): A dictionary mapping object ids to
                objects.  This dictionary has the same semantics as
                the memo object used with ``copy.deepcopy``.  Defaults
                to None, which indicates that no user-defined
                dictionary is used.
        """
        self.enter_named_expr = descend_into_named_expressions
        self.rm_named_expr = remove_named_expressions
        if substitute is None:
            self.substitute = {}
        else:
            self.substitute = substitute

    def visit(self, node, values):
        """
        Visit and clone nodes that have been expanded.

        Note:
            This method normally does not need to be re-defined
            by a user.

        Args:
            node: The node that will be cloned.
            values (list): The list of child nodes that have been
                cloned.  These values are used to define the
                cloned node.

        Returns:
            The cloned node.  Default is to simply return the node.
        """
        return node

    def visiting_potential_leaf(self, node):    #pragma: no cover
        """
        Visit a node and return a cloned node if it is a leaf.

        Note:
            This method needs to be over-written for a specific
            visitor application.

        Args:
            node: a node in a tree

        Returns:
            A tuple: ``(flag, value)``.   If ``flag`` is False,
            then the node is not a leaf and ``value`` is :const:`None`.
            Otherwise, ``value`` is a cloned node.
        """
        _id = id(node)
        if _id in self.substitute:
            return True, self.substitute[_id]
        elif type(node) in nonpyomo_leaf_types or not node.is_expression_type():
            return True, node
        elif not self.enter_named_expr and node.is_named_expression_type():
            return True, node
        else:
            return False, None

    def finalize(self, ans):
        """
        This method defines the return value for the search methods
        in this class.

        The default implementation returns the value of the
        initial node (aka the root node), because
        this visitor pattern computes and returns value for each
        node to enable the computation of this value.

        Args:
            ans: The final value computed by the search method.

        Returns:
            The final value after the search. Defaults to simply
            returning :attr:`ans`.
        """
        return ans

    def construct_node(self, node, values):
        """
        Call the expression create_node_with_local_data() method.
        """
        return node.create_node_with_local_data( tuple(values) )

    def dfs_postorder_stack(self, node):
        """
        Perform a depth-first search in postorder using a stack
        implementation.

        This method replaces subtrees.  This method detects if the
        :func:`visit` method returns a different object.  If so, then
        the node has been replaced and search process is adapted
        to replace all subsequent parent nodes in the tree.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`ValueVisitor.dfs_postorder_stack <pyutilib.misc.visitor.ValueVisitor.dfs_postorder_stack>`
            method that is tailored to support the
            replacement of sub-trees in a Pyomo expression tree.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.
        """
        if node.__class__ is LinearExpression:
            _argList = [node.constant] + node.linear_coefs + node.linear_vars
            _len = len(_argList)
            _stack = [ (node, _argList, 0, _len, [False])]
        else:
            flag, value = self.visiting_potential_leaf(node)
            if flag:
                return value
            _stack = [ (node, node._args_, 0, node.nargs(), [False])]
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
            #   _result     The 'dirty' flag followed by return values
            #
            _obj, _argList, _idx, _len, _result = _stack.pop()
            #
            # Iterate through the arguments, entering each one
            #
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                flag, value = self.visiting_potential_leaf(_sub)
                if flag:
                    if id(value) != id(_sub):
                        _result[0] = True
                    _result.append( value )
                else:
                    #
                    # Push an expression onto the stack
                    #
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj = _sub
                    _idx = 0
                    _result = [False]
                    if _sub.__class__ is LinearExpression:
                        _argList = [_sub.constant] + _sub.linear_coefs \
                                   + _sub.linear_vars
                        _len = len(_argList)
                    else:
                        _argList = _sub._args_
                        _len = _sub.nargs()
            #
            # Finalize (exit) the current node
            #
            # If the user has defined a visit() function in a
            # subclass, then call that function.  But if the user
            # hasn't created a new class and we need to, then
            # call the ExpressionReplacementVisitor.visit() function.
            #
            ans = self.visit(_obj, _result[1:])
            if ans.is_named_expression_type():
                if self.rm_named_expr:
                    ans = _result[1]
                    _result[0] = True
                else:
                    _result[0] = False
                    assert(len(_result) == 2)
                    ans.expr = _result[1]
            elif _result[0]:
                if ans.__class__ is LinearExpression:
                    ans = _result[1]
                    nterms = (len(_result)-2)//2
                    for i in range(nterms):
                        ans += _result[2+i]*_result[2+i+nterms]
                if id(ans) == id(_obj):
                    ans = self.construct_node(_obj, _result[1:])
                if ans.__class__ is MonomialTermExpression:
                    if ( ( ans._args_[0].__class__ not in native_numeric_types
                           and ans._args_[0].is_potentially_variable )
                         or
                         ( ans._args_[1].__class__ in native_numeric_types
                           or not ans._args_[1].is_potentially_variable() ) ):
                        ans.__class__ = ProductExpression
                elif ans.__class__ in NPV_expression_types:
                    # For simplicity, not-potentially-variable expressions are
                    # replaced with their potentially variable counterparts.
                    ans = ans.create_potentially_variable_object()
            elif id(ans) != id(_obj):
                _result[0] = True

            if _stack:
                if _result[0]:
                    _stack[-1][-1][0] = True
                #
                # "return" the recursion by putting the return value on
                # the end of the results stack
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

def clone_expression(expr, substitute=None):
    """A function that is used to clone an expression.

    Cloning is equivalent to calling ``copy.deepcopy`` with no Block
    scope.  That is, the expression tree is duplicated, but no Pyomo
    components (leaf nodes *or* named Expressions) are duplicated.

    Args:
        expr: The expression that will be cloned.
        substitute (dict): A dictionary mapping object ids to
            objects. This dictionary has the same semantics as
            the memo object used with ``copy.deepcopy``. Defaults
            to None, which indicates that no user-defined
            dictionary is used.

    Returns:
        The cloned expression.

    """
    clone_counter._count += 1
    memo = {'__block_scope__': {id(None): False}}
    if substitute:
        memo.update(substitute)
    return deepcopy(expr, memo)


# =====================================================
#  _sizeof_expression
# =====================================================

class _SizeVisitor(SimpleExpressionVisitor):

    def __init__(self):
        self.counter = 0

    def visit(self, node):
        self.counter += 1

    def finalize(self):
        return self.counter


def _sizeof_expression(expr):
    """
    Return the number of nodes in the expression tree.

    Args:
        expr: The root node of an expression tree.

    Returns:
        A non-negative integer that is the number of
        interior and leaf nodes in the expression tree.
    """
    visitor = _SizeVisitor()
    return visitor.xbfs(expr)

# =====================================================
#  evaluate_expression
# =====================================================

class _EvaluationVisitor(ExpressionValueVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return True, node

        if node.is_variable_type():
            return True, value(node)

        if not node.is_expression_type():
            return True, value(node)

        return False, None


class FixedExpressionError(Exception):

    def __init__(self, *args, **kwds):
        super(FixedExpressionError, self).__init__(*args, **kwds)


class NonConstantExpressionError(Exception):

    def __init__(self, *args, **kwds):
        super(NonConstantExpressionError, self).__init__(*args, **kwds)


class _EvaluateConstantExpressionVisitor(ExpressionValueVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._apply_operation(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types:
            return True, node

        if node.is_parameter_type():
            if node._component()._mutable:
                raise FixedExpressionError()
            return True, value(node)
                

        if node.is_variable_type():
            if node.fixed:
                raise FixedExpressionError()
            else:
                raise NonConstantExpressionError()

        if not node.is_expression_type():
            return True, value(node)

        return False, None


def evaluate_expression(exp, exception=True, constant=False):
    """Evaluate the value of the expression.

    Args:
        expr: The root node of an expression tree.
        exception (bool): A flag that indicates whether
            exceptions are raised.  If this flag is
            :const:`False`, then an exception that
            occurs while evaluating the expression
            is caught and the return value is :const:`None`.
            Default is :const:`True`.
        constant (bool): If True, constant expressions are
            evaluated and returned but nonconstant expressions
            raise either FixedExpressionError or
            NonconstantExpressionError (default=False).

    Returns:
        A floating point value if the expression evaluates
        normally, or :const:`None` if an exception occurs
        and is caught.

    """
    if constant:
        visitor = _EvaluateConstantExpressionVisitor()
    else:
        visitor = _EvaluationVisitor()
    try:
        return visitor.dfs_postorder_stack(exp)

    except NonConstantExpressionError:  #pragma: no cover
        if exception:
            raise
        return None

    except FixedExpressionError:        #pragma: no cover
        if exception:
            raise
        return None

    except TemplateExpressionError:     #pragma: no cover
        if exception:
            raise
        return None

    except ValueError:
        if exception:
            raise
        return None


# =====================================================
#  identify_components
# =====================================================

class _ComponentVisitor(SimpleExpressionVisitor):

    def __init__(self, types):
        self.seen = set()
        if types.__class__ is set:
            self.types = types
        else:
            self.types = set(types)

    def visit(self, node):
        if node.__class__ in self.types:
            if id(node) in self.seen:
                return
            self.seen.add(id(node))
            return node


def identify_components(expr, component_types):
    """
    A generator that yields a sequence of nodes
    in an expression tree that belong to a specified set.

    Args:
        expr: The root node of an expression tree.
        component_types (set or list): A set of class
            types that will be matched during the search.

    Yields:
        Each node that is found.
    """
    #
    # OPTIONS:
    # component_types - set (or list) if class types to find
    # in the expression.
    #
    visitor = _ComponentVisitor(component_types)
    for v in visitor.xbfs_yield_leaves(expr):
        yield v


# =====================================================
#  identify_variables
# =====================================================

class _VariableVisitor(SimpleExpressionVisitor):

    def __init__(self):
        self.seen = set()

    def visit(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return

        if node.is_variable_type():
            if id(node) in self.seen:
                return
            self.seen.add(id(node))
            return node


def identify_variables(expr, include_fixed=True):
    """
    A generator that yields a sequence of variables
    in an expression tree.

    Args:
        expr: The root node of an expression tree.
        include_fixed (bool): If :const:`True`, then
            this generator will yield variables whose
            value is fixed.  Defaults to :const:`True`.

    Yields:
        Each variable that is found.
    """
    visitor = _VariableVisitor()
    if include_fixed:
        for v in visitor.xbfs_yield_leaves(expr):
            yield v
    else:
        for v in visitor.xbfs_yield_leaves(expr):
            if not v.is_fixed():
                yield v


# =====================================================
#  identify_mutable_parameters
# =====================================================

class _MutableParamVisitor(SimpleExpressionVisitor):

    def __init__(self):
        self.seen = set()

    def visit(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return

        # TODO: Confirm that this has the right semantics
        if not node.is_variable_type() and node.is_fixed():
            if id(node) in self.seen:
                return
            self.seen.add(id(node))
            return node


def identify_mutable_parameters(expr):
    """
    A generator that yields a sequence of mutable
    parameters in an expression tree.

    Args:
        expr: The root node of an expression tree.

    Yields:
        Each mutable parameter that is found.
    """
    visitor = _MutableParamVisitor()
    for v in visitor.xbfs_yield_leaves(expr):
        yield v


# =====================================================
#  _polynomial_degree
# =====================================================

class _PolynomialDegreeVisitor(ExpressionValueVisitor):

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._compute_polynomial_degree(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types or not node.is_potentially_variable():
            return True, 0

        if not node.is_expression_type():
            return True, 0 if node.is_fixed() else 1

        return False, None


def _polynomial_degree(node):
    """
    Return the polynomial degree of the expression.

    Args:
        node: The root node of an expression tree.

    Returns:
        A non-negative integer that is the polynomial
        degree if the expression is polynomial, or :const:`None` otherwise.
    """
    visitor = _PolynomialDegreeVisitor()
    return visitor.dfs_postorder_stack(node)


# =====================================================
#  _expression_is_fixed
# =====================================================

class _IsFixedVisitor(ExpressionValueVisitor):
    """
    NOTE: This doesn't check if combiner logic is
    all or any and short-circuit the test.  It's
    not clear that that is an important optimization.
    """

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        return node._is_fixed(values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in nonpyomo_leaf_types or not node.is_potentially_variable():
            return True, True

        elif not node.is_expression_type():
            return True, node.is_fixed()

        return False, None


def _expression_is_fixed(node):
    """
    Return the polynomial degree of the expression.

    Args:
        node: The root node of an expression tree.

    Returns:
        A non-negative integer that is the polynomial
        degree if the expression is polynomial, or :const:`None` otherwise.
    """
    visitor = _IsFixedVisitor()
    return visitor.dfs_postorder_stack(node)


# =====================================================
#  expression_to_string
# =====================================================

class _ToStringVisitor(ExpressionValueVisitor):

    def __init__(self, verbose, smap, compute_values):
        super(_ToStringVisitor, self).__init__()
        self.verbose = verbose
        self.smap = smap
        self.compute_values = compute_values

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        tmp = []
        for i,val in enumerate(values):
            arg = node._args_[i]

            if arg is None:
                tmp.append('Undefined')                 # TODO: coverage
            elif arg.__class__ in native_numeric_types:
                tmp.append(val)
            elif arg.__class__ in nonpyomo_leaf_types:
                tmp.append("'{0}'".format(val))
            elif arg.is_variable_type():
                tmp.append(val)
            elif not self.verbose and arg.is_expression_type() and node._precedence() < arg._precedence():
                tmp.append("({0})".format(val))
            else:
                tmp.append(val)

        return node._to_string(tmp, self.verbose, self.smap, self.compute_values)

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node is None:
            return True, None                           # TODO: coverage

        if node.__class__ in nonpyomo_leaf_types:
            return True, str(node)

        if node.is_variable_type():
            if not node.fixed:
                return True, node.to_string(verbose=self.verbose, smap=self.smap, compute_values=False)
            return True, node.to_string(verbose=self.verbose, smap=self.smap, compute_values=self.compute_values)

        if not node.is_expression_type():
            return True, node.to_string(verbose=self.verbose, smap=self.smap, compute_values=self.compute_values)

        return False, None


def expression_to_string(expr, verbose=None, labeler=None, smap=None, compute_values=False):
    """
    Return a string representation of an expression.

    Args:
        expr: The root node of an expression tree.
        verbose (bool): If :const:`True`, then the output is
            a nested functional form.  Otherwise, the output
            is an algebraic expression.  Default is :const:`False`.
        labeler:  If specified, this labeler is used to label
            variables in the expression.
        smap:  If specified, this :class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>` is
            used to cache labels.
        compute_values (bool): If :const:`True`, then
            parameters and fixed variables are evaluated before the
            expression string is generated.  Default is :const:`False`.

    Returns:
        A string representation for the expression.
    """
    verbose = common.TO_STRING_VERBOSE if verbose is None else verbose
    #
    # Setup the symbol map
    #
    if labeler is not None:
        if smap is None:
            smap = SymbolMap()
        smap.default_labeler = labeler
    #
    # Create and execute the visitor pattern
    #
    visitor = _ToStringVisitor(verbose, smap, compute_values)
    return visitor.dfs_postorder_stack(expr)


#-------------------------------------------------------
#
# Expression classes
#
#-------------------------------------------------------


class ExpressionBase(NumericValue):
    """
    The base class for Pyomo expressions.

    This class is used to define nodes in an expression
    tree.

    Args:
        args (list or tuple): Children of this node.
    """

    # Previously, we used _args to define expression class arguments.
    # Here, we use _args_ to force errors for code that was referencing this
    # data.  There are now accessor methods, so in most cases users
    # and developers should not directly access the _args_ data values.
    __slots__ =  ('_args_',)
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        """
        Returns the number of child nodes.

        By default, Pyomo expressions represent binary operations
        with two arguments.

        Note:
            This function does not simply compute the length of
            :attr:`_args_` because some expression classes use
            a subset of the :attr:`_args_` array.  Thus, it
            is imperative that developers use this method!

        Returns:
            A nonnegative integer that is the number of child nodes.
        """
        return 2

    def arg(self, i):
        """
        Return the i-th child node.

        Args:
            i (int): Nonnegative index of the child that is returned.

        Returns:
            The i-th child node.
        """
        if i >= self.nargs():
            raise KeyError("Invalid index for expression argument: %d" % i)
        if i < 0:
            return self._args_[self.nargs()+i]
        return self._args_[i]

    @property
    def args(self):
        """
        A generator that yields the child nodes.

        Yields:
            Each child node in order.
        """
        return islice(self._args_, self.nargs())

    def __getstate__(self):
        """
        Pickle the expression object

        Returns:
            The pickled state.
        """
        state = super(ExpressionBase, self).__getstate__()
        for i in ExpressionBase.__slots__:
           state[i] = getattr(self,i)
        return state

    def __nonzero__(self):      #pragma: no cover
        """
        Compute the value of the expression and convert it to
        a boolean.

        Returns:
            A boolean value.
        """
        return bool(self())

    __bool__ = __nonzero__

    def __call__(self, exception=True):
        """
        Evaluate the value of the expression tree.

        Args:
            exception (bool): If :const:`False`, then
                an exception raised while evaluating
                is captured, and the value returned is
                :const:`None`.  Default is :const:`True`.

        Returns:
            The value of the expression or :const:`None`.
        """
        return evaluate_expression(self, exception)

    def __str__(self):
        """
        Returns a string description of the expression.

        Note:
            The value of ``pyomo.core.expr.expr_common.TO_STRING_VERBOSE``
            is used to configure the execution of this method.
            If this value is :const:`True`, then the string
            representation is a nested function description of the expression.
            The default is :const:`False`, which is an algebraic
            description of the expression.

        Returns:
            A string.
        """
        return expression_to_string(self)

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """
        Return a string representation of the expression tree.

        Args:
            verbose (bool): If :const:`True`, then the the string
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
                Defaults to :const:`False`.
            labeler: An object that generates string labels for
                variables in the expression tree.  Defaults to :const:`None`.
            smap:  If specified, this :class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>` is
                used to cache labels for variables.
            compute_values (bool): If :const:`True`, then
                parameters and fixed variables are evaluated before the
                expression string is generated.  Default is :const:`False`.

        Returns:
            A string representation for the expression tree.
        """
        return expression_to_string(self, verbose=verbose, labeler=labeler, smap=smap, compute_values=compute_values)

    def _precedence(self):
        return ExpressionBase.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):            #pragma: no cover
        """
        Construct a string representation for this node, using the string
        representations of its children.

        This method is called by the :class:`_ToStringVisitor
        <pyomo.core.expr.current._ToStringVisitor>` class.  It must
        must be defined in subclasses.

        Args:
            values (list): The string representations of the children of this
                node.
            verbose (bool): If :const:`True`, then the the string
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
            smap:  If specified, this :class:`SymbolMap
                <pyomo.core.expr.symbol_map.SymbolMap>` is
                used to cache labels for variables.
            compute_values (bool): If :const:`True`, then
                parameters and fixed variables are evaluated before the
                expression string is generated.

        Returns:
            A string representation for this node.
        """
        pass

    def getname(self, *args, **kwds):                       #pragma: no cover
        """
        Return the text name of a function associated with this expression object.

        In general, no arguments are passed to this function.

        Args:
            *arg: a variable length list of arguments
            **kwds: keyword arguments

        Returns:
            A string name for the function.
        """
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement getname()" % ( str(self.__class__), ))

    def clone(self, substitute=None):
        """
        Return a clone of the expression tree.

        Note:
            This method does not clone the leaves of the
            tree, which are numeric constants and variables.
            It only clones the interior nodes, and
            expression leaf nodes like
            :class:`_MutableLinearExpression<pyomo.core.expr.current._MutableLinearExpression>`.
            However, named expressions are treated like
            leaves, and they are not cloned.

        Args:
            substitute (dict): a dictionary that maps object ids to clone
                objects generated earlier during the cloning process.

        Returns:
            A new expression tree.
        """
        return clone_expression(self, substitute=substitute)

    def create_node_with_local_data(self, args):
        """
        Construct a node using given arguments.

        This method provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.  In the simplest
        case, this simply returns::

            self.__class__(args)

        But in general this creates an expression object using local
        data as well as arguments that represent the child nodes.

        Args:
            args (list): A list of child nodes for the new expression
                object
            memo (dict): A dictionary that maps object ids to clone
                objects generated earlier during a cloning process.
                This argument is needed to clone objects that are
                owned by a model, and it can be safely ignored for
                most expression classes.

        Returns:
            A new expression object with the same type as the current
            class.
        """
        return self.__class__(args)

    def create_potentially_variable_object(self):
        """
        Create a potentially variable version of this object.

        This method returns an object that is a potentially variable
        version of the current object.  In the simplest
        case, this simply sets the value of `__class__`:

            self.__class__ = self.__class__.__mro__[1]

        Note that this method is allowed to modify the current object
        and return it.  But in some cases it may create a new 
        potentially variable object.

        Returns:
            An object that is potentially variable.
        """
        self.__class__ = self.__class__.__mro__[1]
        return self

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
        """
        Return :const:`True` if this expression contains no free variables.

        Returns:
            A boolean.
        """
        return _expression_is_fixed(self)

    def _is_fixed(self, values):
        """
        Compute whether this expression is fixed given
        the fixed values of its children.

        This method is called by the :class:`_IsFixedVisitor
        <pyomo.core.expr.current._IsFixedVisitor>` class.  It can
        be over-written by expression classes to customize this
        logic.

        Args:
            values (list): A list of boolean values that indicate whether
                the children of this expression are fixed

        Returns:
            A boolean that is :const:`True` if the fixed values of the
            children are all :const:`True`.
        """
        return all(values)

    def is_potentially_variable(self):
        """
        Return :const:`True` if this expression might represent
        a variable expression.

        This method returns :const:`True` when (a) the expression
        tree contains one or more variables, or (b) the expression 
        tree contains a named expression. In both cases, the
        expression cannot be treated as constant since (a) the variables
        may not be fixed, or (b) the named expressions may be changed
        at a later time to include non-fixed variables.

        Returns:
            A boolean.  Defaults to :const:`True` for expressions.
        """
        return True

    def is_named_expression_type(self):
        """
        Return :const:`True` if this object is a named expression.

        This method returns :const:`False` for this class, and it
        is included in other classes within Pyomo that are not named
        expressions, which allows for a check for named expressions
        without evaluating the class type.

        Returns:
            A boolean.
        """
        return False

    def is_expression_type(self):
        """
        Return :const:`True` if this object is an expression.

        This method obviously returns :const:`True` for this class, but it
        is included in other classes within Pyomo that are not expressions,
        which allows for a check for expressions without
        evaluating the class type.

        Returns:
            A boolean.
        """
        return True

    def size(self):
        """
        Return the number of nodes in the expression tree.

        Returns:
            A nonnegative integer that is the number of interior and leaf
            nodes in the expression tree.
        """
        return _sizeof_expression(self)

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return _PolynomialDegreeVisitor().dfs_postorder_stack(self)

    def _compute_polynomial_degree(self, values):                          #pragma: no cover
        """
        Compute the polynomial degree of this expression given
        the degree values of its children.

        This method is called by the :class:`_PolynomialDegreeVisitor
        <pyomo.core.expr.current._PolynomialDegreeVisitor>` class.  It can
        be over-written by expression classes to customize this
        logic.

        Args:
            values (list): A list of values that indicate the degree
                of the children expression.

        Returns:
            A nonnegative integer that is the polynomial degree of the
            expression, or :const:`None`.  Default is :const:`None`.
        """
        return None

    def _apply_operation(self, result):     #pragma: no cover
        """
        Compute the values of this node given the values of its children.

        This method is called by the :class:`_EvaluationVisitor
        <pyomo.core.expr.current._EvaluationVisitor>` class.  It must
        be over-written by expression classes to customize this logic.

        Note:
            This method applies the logical operation of the
            operator to the arguments.  It does *not* evaluate
            the arguments in the process, but assumes that they
            have been previously evaluated.  But noted that if
            this class contains auxilliary data (e.g. like the
            numeric coefficients in the :class:`LinearExpression
            <pyomo.core.expr.current.LinearExpression>` class, then
            those values *must* be evaluated as part of this
            function call.  An uninitialized parameter value
            encountered during the execution of this method is
            considered an error.

        Args:
            values (list): A list of values that indicate the value
                of the children expressions.

        Returns:
            A floating point value for this expression.
        """
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement _apply_operation()" % ( str(self.__class__), ))


class NegationExpression(ExpressionBase):
    """
    Negation expressions::

        - x
    """

    __slots__ = ()

    PRECEDENCE = 4

    def nargs(self):
        return 1

    def getname(self, *args, **kwds):
        return 'neg'

    def _compute_polynomial_degree(self, result):
        return result[0]

    def _precedence(self):
        return NegationExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1})".format(self.getname(), values[0])
        tmp = values[0]
        if tmp[0] == '-':
            i = 1
            while tmp[i] == ' ':
                i += 1
            return tmp[i:]
        return "- "+tmp

    def _apply_operation(self, result):
        return -result[0]


class NPV_NegationExpression(NegationExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class ExternalFunctionExpression(ExpressionBase):
    """
    External function expressions

    Example::

        model = ConcreteModel()
        model.a = Var()
        model.f = ExternalFunction(library='foo.so', function='bar')
        expr = model.f(model.a)

    Args:
        args (tuple): children of this node
        fcn: a class that defines this external function
    """
    __slots__ = ('_fcn',)

    def __init__(self, args, fcn=None):
        self._args_ = args
        self._fcn = fcn

    def nargs(self):
        return len(self._args_)

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._fcn)

    def __getstate__(self):
        state = super(ExternalFunctionExpression, self).__getstate__()
        for i in ExternalFunctionExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):           #pragma: no cover
        return self._fcn.getname(*args, **kwds)

    def _compute_polynomial_degree(self, result):
        # If the expression is constant, then
        # this is detected earlier.  Hence, we can safely
        # return None.
        return None

    def _apply_operation(self, result):
        return self._fcn.evaluate( result )     #pragma: no cover

    def _to_string(self, values, verbose, smap, compute_values):
        return "{0}({1})".format(self.getname(), ", ".join(values))


class NPV_ExternalFunctionExpression(ExternalFunctionExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class PowExpression(ExpressionBase):
    """
    Power expressions::

        x**y
    """

    __slots__ = ()
    PRECEDENCE = 2

    def _compute_polynomial_degree(self, result):
        # PowExpression is a tricky thing.  In general, a**b is
        # nonpolynomial, however, if b == 0, it is a constant
        # expression, and if a is polynomial and b is a positive
        # integer, it is also polynomial.  While we would like to just
        # call this a non-polynomial expression, these exceptions occur
        # too frequently (and in particular, a**2)
        l,r = result
        if r == 0:
            if l == 0:
                return 0
            # NOTE: use value before int() so that we don't
            #       run into the disabled __int__ method on
            #       NumericValue
            exp = value(self._args_[1], exception=False)
            if exp is None:
                return None
            if exp == int(exp):
                if l is not None and exp > 0:
                    return l * exp
                elif exp == 0:
                    return 0
        return None

    def _is_fixed(self, args):
        assert(len(args) == 2)
        if not args[1]:
            return False
        return args[0] or value(self._args_[1]) == 0

    def _precedence(self):
        return PowExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        return _l ** _r

    def getname(self, *args, **kwds):
        return 'pow'

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1}, {2})".format(self.getname(), values[0], values[1])
        return "{0}**{1}".format(values[0], values[1])


class NPV_PowExpression(PowExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class ProductExpression(ExpressionBase):
    """
    Product expressions::

        x*y
    """

    __slots__ = ()
    PRECEDENCE = 4

    def _precedence(self):
        return ProductExpression.PRECEDENCE

    def _compute_polynomial_degree(self, result):
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

    def _is_fixed(self, args):
        # Anything times 0 equals 0, so one of the children is
        # fixed and has a value of 0, then this expression is fixed
        assert(len(args) == 2)
        if all(args):
            return True
        for i in (0, 1):
            if args[i] and value(self._args_[i]) == 0:
                return True
        return False

    def _apply_operation(self, result):
        _l, _r = result
        return _l * _r

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1}, {2})".format(self.getname(), values[0], values[1])
        if values[0] == "1" or values[0] == "1.0":
            return values[1]
        if values[0] == "-1" or values[0] == "-1.0":
            return "- {0}".format(values[1])
        return "{0}*{1}".format(values[0],values[1])


class NPV_ProductExpression(ProductExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class MonomialTermExpression(ProductExpression):
    __slots__ = ()


class ReciprocalExpression(ExpressionBase):
    """
    Reciprocal expressions::

        1/x
    """
    __slots__ = ()
    PRECEDENCE = 3.5

    def nargs(self):
        return 1

    def _precedence(self):
        return ReciprocalExpression.PRECEDENCE

    def _compute_polynomial_degree(self, result):
        if result[0] == 0:
            return 0
        return None

    def getname(self, *args, **kwds):
        return 'recip'

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1})".format(self.getname(), values[0])
        return "(1/{0})".format(values[0])

    def _apply_operation(self, result):
        return 1 / result[0]


class NPV_ReciprocalExpression(ReciprocalExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class _LinearOperatorExpression(ExpressionBase):
    """
    An 'abstract' class that defines the polynomial degree for a simple
    linear operator
    """

    __slots__ = ()

    def _compute_polynomial_degree(self, result):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        ans = 0
        for x in result:
            if x is None:
                return None
            elif ans < x:
                ans = x
        return ans


class RangedExpression(_LinearOperatorExpression):
    """
    Ranged expressions, which define relations with a lower and upper bound::

        x < y < z
        x <= y <= z

    Args:
        args (tuple): child nodes
        strict (tuple): flags that indicates whether the inequalities are strict
    """

    __slots__ = ('_strict',)
    PRECEDENCE = 9

    def __init__(self, args, strict):
        super(RangedExpression,self).__init__(args)
        self._strict = strict

    def nargs(self):
        return 3

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def __getstate__(self):
        state = super(RangedExpression, self).__getstate__()
        for i in RangedExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def __nonzero__(self):
        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return RangedExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _b, _r = result
        if not self._strict[0]:
            if not self._strict[1]:
                return _l <= _b and _b <= _r
            else:
                return _l <= _b and _b < _r
        elif not self._strict[1]:
            return _l < _b and _b <= _r
        else:
            return _l < _b and _b < _r

    def _to_string(self, values, verbose, smap, compute_values):
        return "{0}  {1}  {2}  {3}  {4}".format(values[0], '<' if self._strict[0] else '<=', values[1], '<' if self._strict[1] else '<=', values[2])

    def is_constant(self):
        return (self._args_[0].__class__ in native_numeric_types or self._args_[0].is_constant()) and \
               (self._args_[1].__class__ in native_numeric_types or self._args_[1].is_constant()) and \
               (self._args_[2].__class__ in native_numeric_types or self._args_[2].is_constant())

    def is_potentially_variable(self):
        return (self._args_[1].__class__ not in native_numeric_types and \
                self._args_[1].is_potentially_variable()) or \
               (self._args_[0].__class__ not in native_numeric_types and \
                self._args_[0].is_potentially_variable()) or \
               (self._args_[2].__class__ not in native_numeric_types and \
                self._args_[2].is_potentially_variable())


class InequalityExpression(_LinearOperatorExpression):
    """
    Inequality expressions, which define less-than or
    less-than-or-equal relations::

        x < y
        x <= y

    Args:
        args (tuple): child nodes
        strict (bool): a flag that indicates whether the inequality is strict
    """

    __slots__ = ('_strict',)
    PRECEDENCE = 9

    def __init__(self, args, strict):
        super(InequalityExpression,self).__init__(args)
        self._strict = strict

    def nargs(self):
        return 2

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def __getstate__(self):
        state = super(InequalityExpression, self).__getstate__()
        for i in InequalityExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def __nonzero__(self):
        if _using_chained_inequality and not self.is_constant():    #pragma: no cover
            deprecation_warning("Chained inequalities are deprecated. "
                                "Use the inequality() function to "
                                "express ranged inequality expressions.")     # Remove in Pyomo 6.0
            _chainedInequality.call_info = traceback.extract_stack(limit=2)[-2]
            _chainedInequality.prev = self
            return True
            #return bool(self())                # This is needed to apply simple evaluation of inequalities

        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return InequalityExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        if self._strict:
            return _l < _r
        return _l <= _r

    def _to_string(self, values, verbose, smap, compute_values):
        if len(values) == 2:
            return "{0}  {1}  {2}".format(values[0], '<' if self._strict else '<=', values[1])

    def is_constant(self):
        return (self._args_[0].__class__ in native_numeric_types or self._args_[0].is_constant()) and \
               (self._args_[1].__class__ in native_numeric_types or self._args_[1].is_constant())

    def is_potentially_variable(self):
        return (self._args_[0].__class__ not in native_numeric_types and \
                self._args_[0].is_potentially_variable()) or \
               (self._args_[1].__class__ not in native_numeric_types and \
                self._args_[1].is_potentially_variable())


def inequality(lower=None, body=None, upper=None, strict=False):
    """
    A utility function that can be used to declare inequality and
    ranged inequality expressions.  The expression::

        inequality(2, model.x)

    is equivalent to the expression::

        2 <= model.x

    The expression::

        inequality(2, model.x, 3)

    is equivalent to the expression::

        2 <= model.x <= 3

    .. note:: This ranged inequality syntax is deprecated in Pyomo.
        This function provides a mechanism for expressing
        ranged inequalities without chained inequalities.

    Args:
        lower: an expression defines a lower bound
        body: an expression defines the body of a ranged constraint
        upper: an expression defines an upper bound
        strict (bool): A boolean value that indicates whether the inequality
            is strict.  Default is :const:`False`.

    Returns:
        A relational expression.  The expression is an inequality
        if any of the values :attr:`lower`, :attr:`body` or
        :attr:`upper` is :const:`None`.  Otherwise, the expression
        is a ranged inequality.
    """
    if lower is None:
        if body is None or upper is None:
            raise ValueError("Invalid inequality expression.")
        return InequalityExpression((body, upper), strict)
    if body is None:
        if lower is None or upper is None:
            raise ValueError("Invalid inequality expression.")
        return InequalityExpression((lower, upper), strict)
    if upper is None:
        return InequalityExpression((lower, body), strict)
    return RangedExpression((lower, body, upper), (strict, strict))

class EqualityExpression(_LinearOperatorExpression):
    """
    Equality expression::

        x == y
    """

    __slots__ = ()
    PRECEDENCE = 9

    def nargs(self):
        return 2

    def __nonzero__(self):
        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return EqualityExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        return _l == _r

    def _to_string(self, values, verbose, smap, compute_values):
        return "{0}  ==  {1}".format(values[0], values[1])

    def is_constant(self):
        return self._args_[0].is_constant() and self._args_[1].is_constant()

    def is_potentially_variable(self):
        return self._args_[0].is_potentially_variable() or self._args_[1].is_potentially_variable()


class SumExpressionBase(_LinearOperatorExpression):
    """
    A base class for simple summation of expressions

    The class hierarchy for summation is different than for other 
    expression types.  For example, ProductExpression defines 
    the class for representing binary products, and sub-classes are
    specializations of that class.

    By contrast, the SumExpressionBase is not directly used to 
    represent expressions.  Rather, this base class provides 
    commonly used methods and data.  The reason is that some
    subclasses of SumExpressionBase are binary while others
    are n-ary.

    Thus, developers will need to treat checks for summation
    classes differently, depending on whether the binary/n-ary 
    operations are different.
    """

    __slots__ = ()
    PRECEDENCE = 6

    def _precedence(self):
        return SumExpressionBase.PRECEDENCE

    def getname(self, *args, **kwds):
        return 'sum'


class NPV_SumExpression(SumExpressionBase):
    __slots__ = ()

    def create_potentially_variable_object(self):
        return SumExpression( self._args_ )

    def _apply_operation(self, result):
        l_, r_ = result
        return l_ + r_

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1}, {2})".format(self.getname(), values[0], values[1])
        if values[1][0] == '-':
            return "{0} {1}".format(values[0],values[1])
        return "{0} + {1}".format(values[0],values[1])

    def is_potentially_variable(self):
        return False


class SumExpression(SumExpressionBase):
    """
    Sum expression::

        x + y

    Args:
        args (list): Children nodes
    """
    __slots__ = ('_nargs','_shared_args')
    PRECEDENCE = 6

    def __init__(self, args):
        self._args_ = args
        self._shared_args = False
        self._nargs = len(self._args_)

    def add(self, new_arg):
        if new_arg.__class__ in native_numeric_types and new_arg == 0:
            return self
        # Clone 'self', because SumExpression are immutable
        self._shared_args = True
        self = self.__class__(self._args_)
        #
        if new_arg.__class__ is SumExpression or new_arg.__class__ is _MutableSumExpression:
            self._args_.extend( islice(new_arg._args_, new_arg._nargs) )
        elif not new_arg is None:
            self._args_.append(new_arg)
        self._nargs = len(self._args_)
        return self

    def nargs(self):
        return self._nargs

    def _precedence(self):
        return SumExpression.PRECEDENCE

    def _apply_operation(self, result):
        return sum(result)

    def create_node_with_local_data(self, args):
        return self.__class__(list(args))

    def __getstate__(self):
        state = super(SumExpression, self).__getstate__()
        for i in SumExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def is_constant(self):
        #
        # In most normal contexts, a SumExpression is
        # non-constant.  When Forming expressions, constant
        # parameters are turned into numbers, which are
        # simply added.  Mutable parameters, variables and
        # expressions are not constant.
        #
        return False

    def is_potentially_variable(self):
        for v in islice(self._args_, self._nargs):
            if v.__class__ in nonpyomo_leaf_types:
                continue
            if v.is_variable_type() or v.is_potentially_variable():
                return True
        return False

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            tmp = [values[0]]
            for i in range(1,len(values)):
                tmp.append(", ")
                tmp.append(values[i])
            return "{0}({1})".format(self.getname(), "".join(tmp))

        tmp = [values[0]]
        for i in range(1,len(values)):
            if values[i][0] == '-':
                tmp.append(' - ')
                j = 1
                while values[i][j] == ' ':
                    j += 1
                tmp.append(values[i][j:])
            else:
                tmp.append(' + ')
                tmp.append(values[i])
        return ''.join(tmp)


class _MutableSumExpression(SumExpression):
    """
    A mutable SumExpression

    The :func:`add` method is slightly different in that it
    does not create a new sum expression, but modifies the
    :attr:`_args_` data in place.
    """

    __slots__ = ()

    def add(self, new_arg):
        if new_arg.__class__ in native_numeric_types and new_arg == 0:
            return self
        # Do not clone 'self', because _MutableSumExpression are mutable
        #self._shared_args = True
        #self = self.__class__(list(self.args))
        #
        if new_arg.__class__ is SumExpression or new_arg.__class__ is _MutableSumExpression:
            self._args_.extend( islice(new_arg._args_, new_arg._nargs) )
        elif not new_arg is None:
            self._args_.append(new_arg)
        self._nargs = len(self._args_)
        return self


class GetItemExpression(ExpressionBase):
    """
    Expression to call :func:`__getitem__` on the base object.
    """
    __slots__ = ('_base',)
    PRECEDENCE = 1

    def _precedence(self):  #pragma: no cover
        return GetItemExpression.PRECEDENCE

    def __init__(self, args, base=None):
        """Construct an expression with an operation and a set of arguments"""
        self._args_ = args
        self._base = base

    def nargs(self):
        return len(self._args_)

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._base)

    def __getstate__(self):
        state = super(GetItemExpression, self).__getstate__()
        for i in GetItemExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return self._base.getname(*args, **kwds)

    def is_potentially_variable(self):
        if any(arg.is_potentially_variable() for arg in self._args_
               if arg.__class__ not in nonpyomo_leaf_types):
            return True
        for x in itervalues(self._base):
            if x.__class__ not in nonpyomo_leaf_types \
               and x.is_potentially_variable():
                return True
        return False

    def is_fixed(self):
        if any(self._args_):
            for x in itervalues(self._base):
                if not x.__class__ in nonpyomo_leaf_types and not x.is_fixed():
                    return False
        return True

    def _is_fixed(self, values):
        for x in itervalues(self._base):
            if not x.__class__ in nonpyomo_leaf_types and not x.is_fixed():
                return False
        return True

    def _compute_polynomial_degree(self, result):       # TODO: coverage
        if any(x != 0 for x in result):
            return None
        ans = 0
        for x in itervalues(self._base):
            if x.__class__ in nonpyomo_leaf_types:
                continue
            tmp = x.polynomial_degree()
            if tmp is None:
                return None
            elif tmp > ans:
                ans = tmp
        return ans

    def _apply_operation(self, result):                 # TODO: coverage
        return value(self._base.__getitem__( tuple(result) ))

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1})".format(self.getname(), values[0])
        return "%s%s" % (self.getname(), values[0])

    def resolve_template(self):                         # TODO: coverage
        return self._base.__getitem__(tuple(value(i) for i in self._args_))


class Expr_ifExpression(ExpressionBase):
    """
    A logical if-then-else expression::

        Expr_if(IF_=x, THEN_=y, ELSE_=z)

    Args:
        IF_ (expression): A relational expression
        THEN_ (expression): An expression that is used if :attr:`IF_` is true.
        ELSE_ (expression): An expression that is used if :attr:`IF_` is false.
    """
    __slots__ = ('_if','_then','_else')

    # **NOTE**: This class evaluates the branching "_if" expression
    #           on a number of occasions. It is important that
    #           one uses __call__ for value() and NOT bool().

    def __init__(self, IF_=None, THEN_=None, ELSE_=None):
        if type(IF_) is tuple and THEN_==None and ELSE_==None:
            IF_, THEN_, ELSE_ = IF_
        self._args_ = (IF_, THEN_, ELSE_)
        self._if = IF_
        self._then = THEN_
        self._else = ELSE_
        if self._if.__class__ in native_numeric_types:
            self._if = as_numeric(self._if)

    def nargs(self):
        return 3

    def __getstate__(self):
        state = super(Expr_ifExpression, self).__getstate__()
        for i in Expr_ifExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return "Expr_if"

    def _is_fixed(self, args):
        assert(len(args) == 3)
        if args[0]: #self._if.is_constant():
            if value(self._if):
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
            return (self._then.__class__ in native_numeric_types or self._then.is_constant()) and \
                (self._else.__class__ in native_numeric_types or self._else.is_constant())

    def is_potentially_variable(self):
        return ((not self._if.__class__ in native_numeric_types) and self._if.is_potentially_variable()) or \
            ((not self._then.__class__ in native_numeric_types) and self._then.is_potentially_variable()) or \
            ((not self._else.__class__ in native_numeric_types) and self._else.is_potentially_variable())

    def _compute_polynomial_degree(self, result):
        _if, _then, _else = result
        if _if == 0:
            try:
                return _then if value(self._if) else _else
            except ValueError:
                pass
        return None

    def _to_string(self, values, verbose, smap, compute_values):
        return '{0}( ( {1} ), then=( {2} ), else=( {3} ) )'.\
            format(self.getname(), self._if, self._then, self._else)

    def _apply_operation(self, result):
        _if, _then, _else = result
        return _then if _if else _else


class UnaryFunctionExpression(ExpressionBase):
    """
    An expression object used to define intrinsic functions (e.g. sin, cos, tan).

    Args:
        args (tuple): Children nodes
        name (string): The function name
        fcn: The function that is used to evaluate this expression
    """
    __slots__ = ('_fcn', '_name')

    def __init__(self, args, name=None, fcn=None):
        if not type(args) is tuple:
            args = (args,)
        self._args_ = args
        self._name = name
        self._fcn = fcn

    def nargs(self):
        return 1

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._name, self._fcn)

    def __getstate__(self):
        state = super(UnaryFunctionExpression, self).__getstate__()
        for i in UnaryFunctionExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return self._name

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1})".format(self.getname(), values[0])
        if values[0][0] == '(':
            return '{0}{1}'.format(self._name, values[0])
        else:
            return '{0}({1})'.format(self._name, values[0])

    def _compute_polynomial_degree(self, result):
        if result[0] == 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        return self._fcn(result[0])


class NPV_UnaryFunctionExpression(UnaryFunctionExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


# NOTE: This should be a special class, since the expression generation relies
# on the Python __abs__ method.
class AbsExpression(UnaryFunctionExpression):
    """
    An expression object for the :func:`abs` function.

    Args:
        args (tuple): Children nodes
    """
    __slots__ = ()

    def __init__(self, arg):
        super(AbsExpression, self).__init__(arg, 'abs', abs)

    def create_node_with_local_data(self, args):
        return self.__class__(args)


class NPV_AbsExpression(AbsExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class LinearExpression(ExpressionBase):
    """
    An expression object linear polynomials.

    Args:
        args (tuple): Children nodes
    """
    __slots__ = ('constant',          # The constant term
                 'linear_coefs',      # Linear coefficients
                 'linear_vars')       # Linear variables

    PRECEDENCE = 6

    def __init__(self, args=None):
        # I am not sure why LinearExpression allows omitting args, but
        # it does.  If they are provided, they should be the constant
        # followed by the coefficients followed by the variables.
        if args:
            self.constant = args[0]
            n = (len(args)-1) // 2
            self.linear_coefs = args[1:n+1]
            self.linear_vars = args[n+1:]
        else:
            self.constant = 0
            self.linear_coefs = []
            self.linear_vars = []
        self._args_ = tuple()

    def nargs(self):
        return 0

    def _precedence(self):
        return LinearExpression.PRECEDENCE

    def __getstate__(self):
        state = super(LinearExpression, self).__getstate__()
        for i in LinearExpression.__slots__:
           state[i] = getattr(self,i)
        return state

    def create_node_with_local_data(self, args):
        return self.__class__(args)

    def getname(self, *args, **kwds):
        return 'sum'

    def _compute_polynomial_degree(self, result):
        return 1 if len(self.linear_vars) > 0 else 0

    def is_constant(self):
        return len(self.linear_vars) == 0

    def is_fixed(self):
        if len(self.linear_vars) == 0:
            return True
        for v in self.linear_vars:
            if not v.fixed:
                return False
        return True

    def _to_string(self, values, verbose, smap, compute_values):
        tmp = []
        if compute_values:
            const_ = value(self.constant)
            if not isclose(const_,0):
                tmp = [str(const_)]
        elif self.constant.__class__ in native_numeric_types:
            if not isclose(self.constant, 0):
                tmp = [str(self.constant)]
        else:
            tmp = [self.constant.to_string(compute_values=False)]
        if verbose:
            for c,v in zip(self.linear_coefs, self.linear_vars):
                if smap:                        # TODO: coverage
                    v_ = smap.getSymbol(v)
                else:
                    v_ = str(v)
                if c.__class__ in native_numeric_types or compute_values:
                    c_ = value(c)
                    if isclose(c_,1):
                        tmp.append(str(v_))
                    elif isclose(c_,0):
                        continue
                    else:
                        tmp.append("prod(%s, %s)" % (str(c_),str(v_)))
                else:
                    tmp.append("prod(%s, %s)" % (str(c), v_))
            return "{0}({1})".format(self.getname(), ', '.join(tmp))
        for c,v in zip(self.linear_coefs, self.linear_vars):
            if smap:
                v_ = smap.getSymbol(v)
            else:
                v_ = str(v)
            if c.__class__ in native_numeric_types or compute_values:
                c_ = value(c)
                if isclose(c_,1):
                   tmp.append(" + %s" % v_)
                elif isclose(c_,0):
                    continue
                elif isclose(c_,-1):
                   tmp.append(" - %s" % v_)
                elif c_ < 0:
                   tmp.append(" - %s*%s" % (str(math.fabs(c_)), v_))
                else:
                   tmp.append(" + %s*%s" % (str(c_), v_))
            else:
                tmp.append(" + %s*%s" % (str(c), v_))
        s = "".join(tmp)
        if len(s) == 0:                 #pragma: no cover
            return s
        if s[0] == " ":
            if s[1] == "+":
                return s[3:]
            return s[1:]
        return s

    def is_potentially_variable(self):
        return len(self.linear_vars) > 0

    def _apply_operation(self, result):
        return value(self.constant) + sum(value(c)*v.value for c,v in zip(self.linear_coefs, self.linear_vars))

    #@profile
    def _combine_expr(self, etype, _other):
        if etype == _add or etype == _sub or etype == -_add or etype == -_sub:
            #
            # if etype == _sub,  then _MutableLinearExpression - VAL
            # if etype == -_sub, then VAL - _MutableLinearExpression
            #
            if etype == _sub:
                omult = -1
            else:
                omult = 1
            if etype == -_sub:
                self.constant *= -1
                for i,c in enumerate(self.linear_coefs):
                    self.linear_coefs[i] = -c

            if _other.__class__ in native_numeric_types or not _other.is_potentially_variable():
                self.constant = self.constant + omult * _other
            #
            # WEH - These seem like uncommon cases, so I think we should defer processing them
            #       until _decompose_linear_terms
            #
            #elif _other.__class__ is _MutableLinearExpression:
            #    self.constant = self.constant + omult * _other.constant
            #    for c,v in zip(_other.linear_coefs, _other.linear_vars):
            #        self.linear_coefs.append(omult*c)
            #        self.linear_vars.append(v)
            #elif _other.__class__ is SumExpression or _other.__class__ is _MutableSumExpression:
            #    for e in _other._args_:
            #        for c,v in _decompose_linear_terms(e, multiplier=omult):
            #            if v is None:
            #                self.constant += c
            #            else:
            #                self.linear_coefs.append(c)
            #                self.linear_vars.append(v)
            else:
                for c,v in _decompose_linear_terms(_other, multiplier=omult):
                    if v is None:
                        self.constant += c
                    else:
                        self.linear_coefs.append(c)
                        self.linear_vars.append(v)

        elif etype == _mul or etype == -_mul:
            if _other.__class__ in native_numeric_types:
                multiplier = _other
            elif _other.is_potentially_variable():
                if len(self.linear_vars) > 0:
                    raise ValueError("Cannot multiply a linear expression with a variable expression")
                #
                # The linear expression is a constant, so re-initialize it with
                # a single term that multiplies the expression by the constant value.
                #
                c_ = self.constant
                self.constant = 0
                for c,v in _decompose_linear_terms(_other):
                    if v is None:
                        self.constant = c*c_
                    else:
                        self.linear_vars.append(v)
                        self.linear_coefs.append(c*c_)
                return self
            else:
                multiplier = _other

            if multiplier.__class__ in native_numeric_types and multiplier == 0:
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
            elif _other.is_potentially_variable():
                raise ValueError("Unallowed operation on linear expression: division with a variable RHS")
            else:
                divisor = _other
            self.constant /= divisor
            for i,c in enumerate(self.linear_coefs):
                self.linear_coefs[i] = c/divisor

        elif etype == -_div:
            if self.is_potentially_variable():
                raise ValueError("Unallowed operation on linear expression: division with a variable RHS")
            return _other / self.constant

        elif etype == _neg:
            self.constant *= -1
            for i,c in enumerate(self.linear_coefs):
                self.linear_coefs[i] = - c

        else:
            raise ValueError("Unallowed operation on mutable linear expression: %d" % etype)    #pragma: no cover

        return self


class _MutableLinearExpression(LinearExpression):
    __slots__ = ()


#-------------------------------------------------------
#
# Functions used to generate expressions
#
#-------------------------------------------------------

def decompose_term(expr):
    """
    A function that returns a tuple consisting of (1) a flag indicated
    whether the expression is linear, and (2) a list of tuples that
    represents the terms in the linear expression.

    Args:
        expr (expression): The root node of an expression tree

    Returns:
        A tuple with the form ``(flag, list)``.  If :attr:`flag` is :const:`False`, then
        a nonlinear term has been found, and :const:`list` is :const:`None`.
        Otherwise, :const:`list` is a list of tuples: ``(coef, value)``.
        If :attr:`value` is :const:`None`, then this
        represents a constant term with value :attr:`coef`.  Otherwise,
        :attr:`value` is a variable object, and :attr:`coef` is the
        numeric coefficient.
    """
    if expr.__class__ in nonpyomo_leaf_types or not expr.is_potentially_variable():
        return True, [(expr,None)]
    elif expr.is_variable_type():
        return True, [(1,expr)]
    else:
        try:
            terms = [t_ for t_ in _decompose_linear_terms(expr)]
            return True, terms
        except LinearDecompositionError:
            return False, None

class LinearDecompositionError(Exception):

    def __init__(self, message):
        super(LinearDecompositionError, self).__init__(message)


def _decompose_linear_terms(expr, multiplier=1):
    """
    A generator function that yields tuples for the linear terms
    in an expression.  If nonlinear terms are encountered, this function
    raises the :class:`LinearDecompositionError` exception.

    Args:
        expr (expression): The root node of an expression tree

    Yields:
        Tuples: ``(coef, value)``.  If :attr:`value` is :const:`None`,
        then this represents a constant term with value :attr:`coef`.
        Otherwise, :attr:`value` is a variable object, and :attr:`coef`
        is the numeric coefficient.

    Raises:
        :class:`LinearDecompositionError` if a nonlinear term is encountered.
    """
    if expr.__class__ in native_numeric_types or not expr.is_potentially_variable():
        yield (multiplier*expr,None)
    elif expr.is_variable_type():
        yield (multiplier,expr)
    elif expr.__class__ is MonomialTermExpression:
        yield (multiplier*expr._args_[0], expr._args_[1])
    elif expr.__class__ is ProductExpression:
        if expr._args_[0].__class__ in native_numeric_types or not expr._args_[0].is_potentially_variable():
            for term in _decompose_linear_terms(expr._args_[1], multiplier*expr._args_[0]):
                yield term
        elif expr._args_[1].__class__ in native_numeric_types or not expr._args_[1].is_potentially_variable():
            for term in _decompose_linear_terms(expr._args_[0], multiplier*expr._args_[1]):
                yield term
        else:
            raise LinearDecompositionError("Quadratic terms exist in a product expression.")
    elif expr.__class__ is ReciprocalExpression:
        # The argument is potentially variable, so this represents a nonlinear term
        #
        # NOTE: We're ignoring possible simplifications
        raise LinearDecompositionError("Unexpected nonlinear term")
    elif expr.__class__ is SumExpression or expr.__class__ is _MutableSumExpression:
        for arg in expr.args:
            for term in _decompose_linear_terms(arg, multiplier):
                yield term
    elif expr.__class__ is NegationExpression:
        for term in  _decompose_linear_terms(expr._args_[0], -multiplier):
            yield term
    elif expr.__class__ is LinearExpression or expr.__class__ is _MutableLinearExpression:
        if not (expr.constant.__class__ in native_numeric_types and expr.constant == 0):
            yield (multiplier*expr.constant,None)
        if len(expr.linear_coefs) > 0:
            for c,v in zip(expr.linear_coefs, expr.linear_vars):
                yield (multiplier*c,v)
    else:
        raise LinearDecompositionError("Unexpected nonlinear term")   #pragma: no cover


def _process_arg(obj):
    try:
        if obj.is_parameter_type() and not obj._component()._mutable and obj._constructed:
            # Return the value of an immutable SimpleParam or ParamData object
            return obj()

        elif obj.__class__ is NumericConstant:
            return obj.value

        return obj
    except AttributeError:
        if obj.is_indexed():
            raise TypeError(
                    "Argument for expression is an indexed numeric "
                    "value\nspecified without an index:\n\t%s\nIs this "
                    "value defined over an index that you did not specify?"
                    % (obj.name, ) )
        raise


#@profile
def _generate_sum_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    if _self.__class__ is _MutableLinearExpression:
        try:
            if etype >= _unary:
                return _self._combine_expr(etype, None)
            if _other.__class__ is not _MutableLinearExpression:
                if not (_other.__class__ in native_types or _other.is_expression_type()):
                    _other = _process_arg(_other)
            return _self._combine_expr(etype, _other)
        except LinearDecompositionError:
            pass
    elif _other.__class__ is _MutableLinearExpression:
        try:
            if not (_self.__class__ in native_types or _self.is_expression_type()):
                _self = _process_arg(_self)
            return _other._combine_expr(-etype, _self)
        except LinearDecompositionError:
            pass

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression_type()):
        _self = _process_arg(_self)

    if etype == _neg:
        if _self.__class__ in native_numeric_types:
            return - _self
        elif _self.__class__ is MonomialTermExpression:
            tmp = _self._args_[0]
            if tmp.__class__ in native_numeric_types:
                return MonomialTermExpression((-tmp, _self._args_[1]))
            else:
                return MonomialTermExpression((NPV_NegationExpression((tmp,)), _self._args_[1]))
        elif _self.is_variable_type():
            return MonomialTermExpression((-1, _self))
        elif _self.is_potentially_variable():
            return NegationExpression((_self,))
        else:
            if _self.__class__ is NPV_NegationExpression:
                return _self._args_[0]
            return NPV_NegationExpression((_self,))

    if not (_other.__class__ in native_types or _other.is_expression_type()):
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
        if (_self.__class__ is SumExpression and not _self._shared_args) or \
           _self.__class__ is _MutableSumExpression:
            return _self.add(_other)
        elif (_other.__class__ is SumExpression and not _other._shared_args) or \
            _other.__class__ is _MutableSumExpression:
            return _other.add(_self)
        elif _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                return _self + _other
            elif _other == 0:
                return _self
            if _self.is_potentially_variable():
                return SumExpression([_self, _other])
            return NPV_SumExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                return _other
            if _other.is_potentially_variable():
                #return _LinearSumExpression((_self, _other))
                return SumExpression([_self, _other])
            return NPV_SumExpression((_self, _other))
        elif _other.is_potentially_variable():
            #return _LinearSumExpression((_self, _other))
            return SumExpression([_self, _other])
        elif _self.is_potentially_variable():
            #return _LinearSumExpression((_other, _self))
            #return SumExpression([_other, _self])
            return SumExpression([_self, _other])
        else:
            return NPV_SumExpression((_self, _other))

    elif etype == _sub:
        #
        # x - y
        #
        if (_self.__class__ is SumExpression and not _self._shared_args) or \
           _self.__class__ is _MutableSumExpression:
            return _self.add(-_other)
        elif _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                return _self - _other
            elif _other == 0:
                return _self
            if _self.is_potentially_variable():
                return SumExpression([_self, -_other])
            return NPV_SumExpression((_self, -_other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                if _other.__class__ is MonomialTermExpression:
                    tmp = _other._args_[0]
                    if tmp.__class__ in native_numeric_types:
                        return MonomialTermExpression((-tmp, _other._args_[1]))
                    return MonomialTermExpression((NPV_NegationExpression((_other._args_[0],)), _other._args_[1]))
                elif _other.is_variable_type():
                    return MonomialTermExpression((-1, _other))
                elif _other.is_potentially_variable():
                    return NegationExpression((_other,))
                return NPV_NegationExpression((_other,))
            elif _other.__class__ is MonomialTermExpression:
                return SumExpression([_self, MonomialTermExpression((-_other._args_[0], _other._args_[1]))])
            elif _other.is_variable_type():
                return SumExpression([_self, MonomialTermExpression((-1,_other))])
            elif _other.is_potentially_variable():
                return SumExpression([_self, NegationExpression((_other,))])
            return NPV_SumExpression((_self, NPV_NegationExpression((_other,))))
        elif _other.__class__ is MonomialTermExpression:
            return SumExpression([_self, MonomialTermExpression((-_other._args_[0], _other._args_[1]))])
        elif _other.is_variable_type():
            return SumExpression([_self, MonomialTermExpression((-1,_other))])
        elif _other.is_potentially_variable():
            return SumExpression([_self, NegationExpression((_other,))])
        elif _self.is_potentially_variable():
            return SumExpression([_self, NPV_NegationExpression((_other,))])
        else:
            return NPV_SumExpression((_self, NPV_NegationExpression((_other,))))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover

#@profile
def _generate_mul_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    if _self.__class__ is _MutableLinearExpression:
        try:
            if _other.__class__ is not _MutableLinearExpression:
                if not (_other.__class__ in native_types or _other.is_expression_type()):
                    _other = _process_arg(_other)
            return _self._combine_expr(etype, _other)
        except LinearDecompositionError:
            pass
    elif _other.__class__ is _MutableLinearExpression:
        try:
            if not (_self.__class__ in native_types or _self.is_expression_type()):
                _self = _process_arg(_self)
            return _other._combine_expr(-etype, _self)
        except LinearDecompositionError:
            pass

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression_type()):
        _self = _process_arg(_self)

    if not (_other.__class__ in native_types or _other.is_expression_type()):
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
            elif _other == 0:
                return 0
            elif _other == 1:
                return _self
            if _self.is_variable_type():
                return MonomialTermExpression((_other, _self))
            elif _self.__class__ is MonomialTermExpression:
                tmp = _self._args_[0]
                if tmp.__class__ in native_numeric_types:
                    return MonomialTermExpression((_other*tmp, _self._args_[1]))
                else:
                    return MonomialTermExpression((NPV_ProductExpression((_other,tmp)), _self._args_[1]))
            elif _self.is_potentially_variable():
                return ProductExpression((_self, _other))
            return NPV_ProductExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                return 0
            elif _self == 1:
                return _other
            if _other.is_variable_type():
                return MonomialTermExpression((_self, _other))
            elif _other.__class__ is MonomialTermExpression:
                tmp = _other._args_[0]
                if tmp.__class__ in native_numeric_types:
                    return MonomialTermExpression((_self*tmp, _other._args_[1]))
                else:
                    return MonomialTermExpression((NPV_ProductExpression((_self,tmp)), _other._args_[1]))
            elif _other.is_potentially_variable():
                return ProductExpression((_self, _other))
            return NPV_ProductExpression((_self, _other))
        elif _other.is_variable_type():
            if _self.is_potentially_variable():
                return ProductExpression((_self, _other))
            return MonomialTermExpression((_self, _other))
        elif _other.is_potentially_variable():
            return ProductExpression((_self, _other))
        elif _self.is_variable_type():
            return MonomialTermExpression((_other, _self))
        elif _self.is_potentially_variable():
            return ProductExpression((_self, _other))
        else:
            return NPV_ProductExpression((_self, _other))

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
            if _self.is_variable_type():
                return MonomialTermExpression((1/_other, _self))
            elif _self.__class__ is MonomialTermExpression:
                return MonomialTermExpression((_self._args_[0]/_other, _self._args_[1]))
            elif _self.is_potentially_variable():
                return ProductExpression((_self, 1/_other))
            return NPV_ProductExpression((_self, 1/_other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                return 0
            elif _self == 1:
                if _other.is_potentially_variable():
                    return ReciprocalExpression((_other,))
                return NPV_ReciprocalExpression((_other,))
            elif _other.is_potentially_variable():
                return ProductExpression((_self, ReciprocalExpression((_other,))))
            return NPV_ProductExpression((_self, ReciprocalExpression((_other,))))
        elif _other.is_potentially_variable():
            return ProductExpression((_self, ReciprocalExpression((_other,))))
        elif _self.is_potentially_variable():
            if _self.is_variable_type():
                return MonomialTermExpression((NPV_ReciprocalExpression((_other,)), _self))
            return ProductExpression((_self, NPV_ReciprocalExpression((_other,))))
        else:
            return NPV_ProductExpression((_self, NPV_ReciprocalExpression((_other,))))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover


#@profile
def _generate_other_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression_type()):
        _self = _process_arg(_self)

    #
    # abs(x)
    #
    if etype == _abs:
        if _self.__class__ in native_numeric_types:
            return abs(_self)
        elif _self.is_potentially_variable():
            return AbsExpression(_self)
        else:
            return NPV_AbsExpression(_self)

    if not (_other.__class__ in native_types or _other.is_expression_type()):
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
            elif _self.is_potentially_variable():
                return PowExpression((_self, _other))
            return NPV_PowExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _other.is_potentially_variable():
                return PowExpression((_self, _other))
            return NPV_PowExpression((_self, _other))
        elif _self.is_potentially_variable() or _other.is_potentially_variable():
            return PowExpression((_self, _other))
        else:
            return NPV_PowExpression((_self, _other))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover


if _using_chained_inequality:
    def _generate_relational_expression(etype, lhs, rhs):               #pragma: no cover
        # We cannot trust Python not to recycle ID's for temporary POD data
        # (e.g., floats).  So, if it is a "native" type, we will record the
        # value, otherwise we will record the ID.  The tuple for native
        # types is to guarantee that a native value will *never*
        # accidentally match an ID
        cloned_from = (\
            id(lhs) if lhs.__class__ not in native_numeric_types else (0,lhs),
            id(rhs) if rhs.__class__ not in native_numeric_types else (0,rhs)
            )
        rhs_is_relational = False
        lhs_is_relational = False

        if not (lhs.__class__ in native_types or lhs.is_expression_type()):
            lhs = _process_arg(lhs)
        if not (rhs.__class__ in native_types or rhs.is_expression_type()):
            rhs = _process_arg(rhs)

        if lhs.__class__ in native_numeric_types:
            lhs = as_numeric(lhs)
        elif lhs.is_relational():
            lhs_is_relational = True

        if rhs.__class__ in native_numeric_types:
            rhs = as_numeric(rhs)
        elif rhs.is_relational():
            rhs_is_relational = True

        if _chainedInequality.prev is not None:
            prevExpr = _chainedInequality.prev
            match = []
            # This is tricky because the expression could have been posed
            # with >= operators, so we must figure out which arguments
            # match.  One edge case is when the upper and lower bounds are
            # the same (implicit equality) - in which case *both* arguments
            # match, and this should be converted into an equality
            # expression.
            for i,arg in enumerate(_chainedInequality.cloned_from):
                if arg == cloned_from[0]:
                    match.append((i,0))
                elif arg == cloned_from[1]:
                    match.append((i,1))
            if etype == _eq:
                raise TypeError(_chainedInequality.error_message())
            if len(match) == 1:
                if match[0][0] == match[0][1]:
                    raise TypeError(_chainedInequality.error_message(
                        "Attempting to form a compound inequality with two "
                        "%s bounds" % ('lower' if match[0][0] else 'upper',)))
                if not match[0][1]:
                    cloned_from = _chainedInequality.cloned_from + (cloned_from[1],)
                    lhs = prevExpr
                    lhs_is_relational = True
                else:
                    cloned_from = (cloned_from[0],) + _chainedInequality.cloned_from
                    rhs = prevExpr
                    rhs_is_relational = True
            elif len(match) == 2:
                # Special case: implicit equality constraint posed as a <= b <= a
                if prevExpr._strict or etype == _lt:
                    _chainedInequality.prev = None
                    raise TypeError("Cannot create a compound inequality with "
                          "identical upper and lower\n\tbounds using strict "
                          "inequalities: constraint infeasible:\n\t%s and "
                          "%s < %s" % ( prevExpr.to_string(), lhs, rhs ))
                if match[0] == (0,0):
                    # This is a particularly weird case where someone
                    # evaluates the *same* inequality twice in a row.  This
                    # should always be an error (you can, for example, get
                    # it with "0 <= a >= 0").
                    raise TypeError(_chainedInequality.error_message())
                etype = _eq
            else:
                raise TypeError(_chainedInequality.error_message())
            _chainedInequality.prev = None

        if etype == _eq:
            if lhs_is_relational or rhs_is_relational:
                if lhs_is_relational:
                    val = lhs.to_string()
                else:
                    val = rhs.to_string()
                raise TypeError("Cannot create an EqualityExpression where "\
                      "one of the sub-expressions is a relational expression:\n"\
                      "    " + val)
            _chainedInequality.prev = None
            return EqualityExpression((lhs,rhs))
        else:
            if etype == _le:
                strict = False
            elif etype == _lt:
                strict = True
            else:
                raise ValueError("Unknown relational expression type '%s'" % etype) #pragma: no cover
            if lhs_is_relational:
                if lhs.__class__ is InequalityExpression:
                    if rhs_is_relational:
                        raise TypeError("Cannot create an InequalityExpression "\
                              "where both sub-expressions are relational "\
                              "expressions.")
                    _chainedInequality.prev = None
                    return RangedExpression(lhs._args_ + (rhs,), (lhs._strict,strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + lhs.to_string())
            elif rhs_is_relational:
                if rhs.__class__ is InequalityExpression:
                    _chainedInequality.prev = None
                    return RangedExpression((lhs,) + rhs._args_, (strict, rhs._strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + rhs.to_string())
            else:
                obj = InequalityExpression((lhs, rhs), strict)
                #_chainedInequality.prev = obj
                _chainedInequality.cloned_from = cloned_from
                return obj

else:

    def _generate_relational_expression(etype, lhs, rhs):               #pragma: no cover
        rhs_is_relational = False
        lhs_is_relational = False

        if not (lhs.__class__ in native_types or lhs.is_expression_type()):
            lhs = _process_arg(lhs)
        if not (rhs.__class__ in native_types or rhs.is_expression_type()):
            rhs = _process_arg(rhs)

        if lhs.__class__ in native_numeric_types:
            # TODO: Why do we need this?
            lhs = as_numeric(lhs)
        elif lhs.is_relational():
            lhs_is_relational = True

        if rhs.__class__ in native_numeric_types:
            # TODO: Why do we need this?
            rhs = as_numeric(rhs)
        elif rhs.is_relational():
            rhs_is_relational = True

        if etype == _eq:
            if lhs_is_relational or rhs_is_relational:
                if lhs_is_relational:
                    val = lhs.to_string()
                else:
                    val = rhs.to_string()
                raise TypeError("Cannot create an EqualityExpression where "\
                      "one of the sub-expressions is a relational expression:\n"\
                      "    " + val)
            return EqualityExpression((lhs,rhs))
        else:
            if etype == _le:
                strict = False
            elif etype == _lt:
                strict = True
            else:
                raise ValueError("Unknown relational expression type '%s'" % etype) #pragma: no cover
            if lhs_is_relational:
                if lhs.__class__ is InequalityExpression:
                    if rhs_is_relational:
                        raise TypeError("Cannot create an InequalityExpression "\
                              "where both sub-expressions are relational "\
                              "expressions.")
                    return RangedExpression(lhs._args_ + (rhs,), (lhs._strict,strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + lhs.to_string())
            elif rhs_is_relational:
                if rhs.__class__ is InequalityExpression:
                    return RangedExpression((lhs,) + rhs._args_, (strict, rhs._strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + rhs.to_string())
            else:
                return InequalityExpression((lhs, rhs), strict)


def _generate_intrinsic_function_expression(arg, name, fcn):
    if not (arg.__class__ in native_types or arg.is_expression_type()):
        arg = _process_arg(arg)

    if arg.__class__ in native_types:
        return fcn(arg)
    elif arg.is_potentially_variable():
        return UnaryFunctionExpression(arg, name, fcn)
    else:
        return NPV_UnaryFunctionExpression(arg, name, fcn)


NPV_expression_types = set(
   [NPV_NegationExpression,
    NPV_ExternalFunctionExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_ReciprocalExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
    NPV_AbsExpression])

