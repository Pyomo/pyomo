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

import logging
import traceback

logger = logging.getLogger('pyomo.core')

from pyomo.common.deprecation import deprecation_warning
from .numvalue import (
     native_types,
     native_numeric_types,
     as_numeric,
)
from .expr_common import (
    _add, _sub, _mul, _div,
    _pow, _neg, _abs, _inplace,
    _unary, _radd, _rsub, _rmul,
    _rdiv, _rpow, _iadd, _isub,
    _imul, _idiv, _ipow, _lt, _le,
    _eq,
)
from .numeric_expr import _LinearOperatorExpression, _process_arg

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

