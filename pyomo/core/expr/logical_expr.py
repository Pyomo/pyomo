# -*- coding: utf-8 -*-
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

import types
from itertools import islice

_using_chained_inequality = True
import logging
import traceback

logger = logging.getLogger('pyomo.core')
from pyomo.common.deprecation import deprecation_warning
from .numvalue import (
    native_types,
    native_numeric_types,
    as_numeric,
    native_logical_types,
)

from .boolean_value import (
    BooleanValue,
    BooleanConstant,
)

from .expr_common import (
    _lt, _le,
    _eq,
    _and, _or, _equiv, _inv, _xor, _impl)

from .visitor import (
    evaluate_expression, expression_to_string, polynomial_degree,
    clone_expression, sizeof_expression, _expression_is_fixed
)

from .numeric_expr import _LinearOperatorExpression, _process_arg
import operator


if _using_chained_inequality:
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

else:
    _chainedInequality = None


#-------------------------------------------------------
#
# Expression classes
#
#-------------------------------------------------------


class RangedExpression(_LinearOperatorExpression):
    """
    Ranged expressions, which define relations with a lower and upper bound::

        x < y < z
        x <= y <= z

    args:
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

    args:
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
        if _using_chained_inequality and not self.is_constant():
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

    args:
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



if _using_chained_inequality:
    def _generate_relational_expression(etype, lhs, rhs):
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
                raise ValueError("Unknown relational expression type '%s'" % etype)
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

    def _generate_relational_expression(etype, lhs, rhs):
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
                raise ValueError("Unknown relational expression type '%s'" % etype)
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


def _generate_logical_proposition(etype, lhs, rhs):
    if lhs.__class__ in native_types and lhs.__class__ not in native_logical_types:
        raise TypeError("Cannot create Logical expression with lhs of type '%s'" % lhs.__class__)
    if rhs.__class__ in native_types and rhs.__class__ not in native_logical_types and rhs is not None:
        raise TypeError("Cannot create Logical expression with rhs of type '%s'" % rhs.__class__)

    if etype == _equiv:
        return EquivalenceExpression((lhs, rhs))
    elif etype == _inv:
        assert rhs is None
        return NotExpression((lhs,))
    elif etype == _xor:
        return XorExpression((lhs, rhs))
    elif etype == _impl:
        return ImplicationExpression((lhs, rhs))
    elif etype == _and:
        return land(lhs, rhs)
    elif etype == _or:
        return lor(lhs, rhs)
    else:
        raise ValueError("Unknown logical proposition type '%s'" % etype)  # pragma: no cover


class BooleanExpressionBase(BooleanValue):
    """
    Logical expressions base expression.

    This class is used to define nodes in an expression
    tree.
    
    Abstract

    args:
        args (list or tuple): Children of this node.
    """

    __slots__ = ('_args_',)
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        """
        Returns the number of child nodes.
        """
        raise NotImplementedError(
            "Derived expression (%s) failed to "
            "implement nargs()" % (str(self.__class__), ))

    def args(self, i):
        """
        Return the i-th child node.

        args:
            i (int): Nonnegative index of the child that is returned.

        Returns:
            The i-th child node.
        """
        if i >= self.nargs():
            raise KeyError("Invalid index for expression argsument: %d" % i)
        if i < 0:
            return self._args_[self.nargs()+i]
        return self._args_[i]

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Either a list or tuple (depending on the node storage
            model) containing only the child nodes of this node
        """
        return self._args_[:self.nargs()]

    def __getstate__(self):
        """
        Pickle the expression object

        Returns:
            The pickled state.
        """
        state = super(BooleanExpressionBase, self).__getstate__()
        for i in BooleanExpressionBase.__slots__:
           state[i] = getattr(self,i)
        return state

    def __call__(self, exception=True):
        """
        Evaluate the value of the expression tree.
        args:
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
        args:
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
        return BooleanExpressionBase.PRECEDENCE

    def _associativity(self):
        """Return the associativity of this operator.

        Returns 1 if this operator is left-to-right associative or -1 if
        it is right-to-left associative.  Any other return value will be
        interpreted as "not associative" (implying any arguments that
        are at this operator's _precedence() will be enclosed in parens).
        """
        return 1

    def _to_string(self, values, verbose, smap, compute_values):            #pragma: no cover
        """
        Construct a string representation for this node, using the string
        representations of its children.

        This method is called by the :class:`_ToStringVisitor
        <pyomo.core.expr.current._ToStringVisitor>` class.  It must
        must be defined in subclasses.

        args:
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
        raise NotImplementedError(
            "Derived expression (%s) failed to "
            "implement _to_string()" % (str(self.__class__), ))

    def getname(self, *args, **kwds):                       #pragma: no cover
        """
        Return the text name of a function associated with this expression object.

        In general, no arguments are passed to this function.

        args:
            *arg: a variable length list of arguments
            **kwds: keyword arguments

        Returns:
            A string name for the function.
        """
        raise NotImplementedError(
            "Derived expression (%s) failed to "
            "implement getname()" % (str(self.__class__), ))

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

        args:
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

        args:
            args (list): A list of child nodes for the new expression
                object
            memo (dict): A dictionary that maps object ids to clone
                objects generated earlier during a cloning process.
                This argsument is needed to clone objects that are
                owned by a model, and it can be safely ignored for
                most expression classes.

        Returns:
            A new expression object with the same type as the current
            class.
        """
        return self.__class__(args)

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

        args:
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

        This method returns :const:`True` when the expression
        tree contains one or more variables

        Returns:
            A boolean.  Defaults to :const:`True` for expressions.
        """
        return True

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
        return sizeof_expression(self)

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
            this class contains auxiliary data (e.g. like the
            numeric coefficients in the :class:`LinearExpression
            <pyomo.core.expr.current.LinearExpression>` class, then
            those values *must* be evaluated as part of this
            function call.  An uninitialized parameter value
            encountered during the execution of this method is
            considered an error.

        args:
            values (list): A list of values that indicate the value
                of the children expressions.

        Returns:
            A floating point value for this expression.
        """
        raise NotImplementedError(
            "Derived expression (%s) failed to "
            "implement _apply_operation()" % (str(self.__class__), ))

"""
---------------------------******************--------------------
The following methods are static methods for nodes creator. Those should
do the exact same thing as the class methods as well as overloaded operators.
"""


def lnot(Y):
    """
    Construct a NotExpression for the passed BooleanValue.
    """
    return NotExpression((Y,))


def equivalent(Y1, Y2):
    """
    Construct an EquivalenceExpression Y1 == Y2
    """
    return EquivalenceExpression((Y1, Y2))


def xor(Y1, Y2):
    """
    Construct an XorExpression Y1 xor Y2
    """
    return XorExpression((Y1, Y2))


def implies(Y1, Y2):
    """
    Construct an Implication using function, where Y1 implies Y2
    """
    return ImplicationExpression((Y1, Y2))


def _flattened(args):
    """Flatten any potentially indexed arguments."""
    for arg in args:
        if arg.__class__ in native_types:
            yield arg
        else:
            if isinstance(arg, (types.GeneratorType, list)):
                for _argdata in arg:
                    yield _argdata
            elif arg.is_indexed():
                for _argdata in arg.values():
                    yield _argdata
            else:
                yield arg


def land(*args):
    """
    Construct an AndExpression between passed arguments.
    """
    result = AndExpression([])
    for argdata in _flattened(args):
        result = result.add(argdata)
    return result


def lor(*args):
    """
    Construct an OrExpression between passed arguments.
    """
    result = OrExpression([])
    for argdata in _flattened(args):
        result = result.add(argdata)
    return result


def exactly(n, *args):
    """Creates a new ExactlyExpression

    Require exactly n arguments to be True, to make the expression True

    Usage: exactly(2, m.Y1, m.Y2, m.Y3, ...)

    """
    result = ExactlyExpression([n, ] + list(_flattened(args)))
    return result


def atmost(n, *args):
    """Creates a new AtMostExpression

    Require at most n arguments to be True, to make the expression True

    Usage: atmost(2, m.Y1, m.Y2, m.Y3, ...)

    """
    result = AtMostExpression([n, ] + list(_flattened(args)))
    return result


def atleast(n, *args):
    """Creates a new AtLeastExpression

    Require at least n arguments to be True, to make the expression True

    Usage: atleast(2, m.Y1, m.Y2, m.Y3, ...)

    """
    result = AtLeastExpression([n, ] + list(_flattened(args)))
    return result


class UnaryBooleanExpression(BooleanExpressionBase):
    """
    Abstract class for single-argument logical expressions.
    """
    def nargs(self):
        """
        Returns number of arguments in expression
        """
        return 1


class NotExpression(UnaryBooleanExpression):
    """
    This is the node for a NotExpression, this node should have exactly one child
    """
    PRECEDENCE = 2

    def getname(self, *arg, **kwd):
        return 'Logical Negation'

    def _precedence(self):
        return NotExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return "~%s" % values[0]

    def _apply_operation(self, result):
        return not result[0]


class BinaryBooleanExpression(BooleanExpressionBase):
    """
    Abstract class for binary logical expressions.
    """
    def nargs(self):
        """
        Return the number of argument the expression has
        """
        return 2


class EquivalenceExpression(BinaryBooleanExpression):
    """
    Logical equivalence statement: Y_1 iff Y_2.

    """
    __slots__ = ()

    PRECEDENCE = 6

    def getname(self, *arg, **kwd):
        return 'iff'

    def _precedence(self):
        return EquivalenceExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return " iff ".join(values)

    def _apply_operation(self, result):
        return result[0] == result[1]


class XorExpression(BinaryBooleanExpression):
    """
    Logical Exclusive OR statement: Y_1 ⊻ Y_2
    """
    __slots__ = ()

    PRECEDENCE = 5

    def getname(self, *arg, **kwd):
        return 'xor'

    def _precedence(self):
        return XorExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return " ⊻ ".join(values)

    def _apply_operation(self, result):
        return operator.xor(result[0], result[1])


class ImplicationExpression(BinaryBooleanExpression):
    """
    Logical Implication statement: Y_1 --> Y_2.
    """
    __slots__ = ()

    PRECEDENCE = 6

    def getname(self, *arg, **kwd):
        return 'implies'

    def _precedence(self):
        return ImplicationExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return " --> ".join(values)

    def _apply_operation(self, result):
        return (not result[0]) or result[1]


class NaryBooleanExpression(BooleanExpressionBase):
    """
    The abstract class for NaryBooleanExpression. This class should never be initialized.
    """
    def __init__(self, args):
        self._args_ = args
        self._nargs = len(self._args_)

    def nargs(self):
        """
        Return the number of expression arguments
        """
        return self._nargs

    def getname(self, *arg, **kwd):
        return 'NaryBooleanExpression'


def _add_to_and_or_expression(orig_expr, new_arg):
    """
    Since AND and OR are Nary expressions, we extend the existing expression
    instead of creating a nested expression object if the types are compatible.
    """
    # Clone 'self', because AndExpression/OrExpression are immutable
    if new_arg.__class__ is orig_expr.__class__:
        # adding new AndExpression/OrExpression on the right
        new_expr = orig_expr.__class__(orig_expr._args_)
        new_expr._args_.extend(islice(new_arg._args_, new_arg._nargs))
    else:
        # adding new singleton on the right
        new_expr = orig_expr.__class__(orig_expr._args_)
        new_expr._args_.append(new_arg)

    # TODO set up id()-based scheme for avoiding duplicate entries

    new_expr._nargs = len(new_expr._args_)
    return new_expr


class AndExpression(NaryBooleanExpression):
    """
    This is the node for AndExpression.
    """
    __slots__ = ()

    PRECEDENCE = 4

    def getname(self, *arg, **kwd):
        return 'and'

    def _precedence(self):
        return AndExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return " ∧ ".join(values)

    def _apply_operation(self, result):
        return all(result)

    def add(self, new_arg):
        if new_arg.__class__ in native_logical_types:
            if new_arg is False:
                return BooleanConstant(False)
            elif new_arg is True:
                return self
        return _add_to_and_or_expression(self, new_arg)


class OrExpression(NaryBooleanExpression):
    """
    This is the node for OrExpression.
    """
    __slots__ = ()

    PRECEDENCE = 4

    def getname(self, *arg, **kwd):
        return 'or'

    def _precedence(self):
        return OrExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return " ∨ ".join(values)

    def _apply_operation(self, result):
        return any(result)

    def add(self, new_arg):
        if new_arg.__class__ in native_logical_types:
            if new_arg is False:
                return self
            elif new_arg is True:
                return BooleanConstant(True)
        return _add_to_and_or_expression(self, new_arg)


class ExactlyExpression(NaryBooleanExpression):
    """
    Logical constraint that exactly N child statements are True.

    The first argument N is expected to be a numeric non-negative integer.
    Subsequent arguments are expected to be Boolean.

    Usage: exactly(1, True, False, False) --> True

    """
    __slots__ = ()

    PRECEDENCE = 9

    def getname(self, *arg, **kwd):
        return 'exactly'

    def _precedence(self):
        return ExactlyExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return "exactly(%s: [%s])" % (values[0], ", ".join(values[1:]))

    def _apply_operation(self, result):
        return sum(result[1:]) == result[0]


class AtMostExpression(NaryBooleanExpression):
    """
    Logical constraint that at most N child statements are True.

    The first argument N is expected to be a numeric non-negative integer.
    Subsequent arguments are expected to be Boolean.

    Usage: atmost(1, True, False, False) --> True

    """
    __slots__ = ()

    PRECEDENCE = 9

    def getname(self, *arg, **kwd):
        return 'atmost'

    def _precedence(self):
        return AtMostExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return "atmost(%s: [%s])" % (values[0], ", ".join(values[1:]))

    def _apply_operation(self, result):
        return sum(result[1:]) <= result[0]


class AtLeastExpression(NaryBooleanExpression):
    """
    Logical constraint that at least N child statements are True.

    The first argument N is expected to be a numeric non-negative integer.
    Subsequent arguments are expected to be Boolean.

    Usage: atleast(1, True, False, False) --> True

    """
    __slots__ = ()

    PRECEDENCE = 9

    def getname(self, *arg, **kwd):
        return 'atleast'

    def _precedence(self):
        return AtLeastExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        return "atleast(%s: [%s])" % (values[0], ", ".join(values[1:]))

    def _apply_operation(self, result):
        return sum(result[1:]) >= result[0]


special_boolean_atom_types = {ExactlyExpression, AtMostExpression, AtLeastExpression}
