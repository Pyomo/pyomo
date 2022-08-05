# -*- coding: utf-8 -*-
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.errors import PyomoException, DeveloperError

from .base import ExpressionBaseMixin
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq
from .numvalue import native_numeric_types
from .numeric_expr import _process_arg
from .visitor import polynomial_degree

#-------------------------------------------------------
#
# Expression classes
#
#-------------------------------------------------------

class RelationalExpressionBase(ExpressionBaseMixin, BooleanValue):
    __slots__ = ('_args_',)

    def __init__(self, args):
        self._args_ = args

    def __getstate__(self):
        """
        Pickle the expression object

        Returns:
            The pickled state.
        """
        state = super().__getstate__()
        for i in RelationalExpressionBase.__slots__:
           state[i] = getattr(self,i)
        return state

    def __bool__(self):
        if self.is_constant():
            return bool(self())
        raise PyomoException("""
Cannot convert non-constant Pyomo expression (%s) to bool.
This error is usually caused by using a Var, unit, or mutable Param in a
Boolean context such as an "if" statement, or when checking container
membership or equality. For example,
    >>> m.x = Var()
    >>> if m.x >= 1:
    ...     pass
and
    >>> m.y = Var()
    >>> if m.y in [m.x, m.y]:
    ...     pass
would both cause this exception.""".strip() % (self,))

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Either a list or tuple (depending on the node storage
            model) containing only the child nodes of this node
        """
        return self._args_[:self.nargs()]

    def is_relational(self):
        return True

    def is_constant(self):
        return all(arg is None
                   or arg.__class__ in native_numeric_types
                   or arg.is_constant()
                   for arg in self._args_)

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return polynomial_degree(self)

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


class RangedExpression(RelationalExpressionBase):
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

    # Shared tuples for the most common RangedExpression objects encountered
    # in math programming.  Creating a single (shared) tuple saves memory
    STRICT = {
        False: (False, False),
        True: (True, True),
        (True, True): (True, True),
        (False, False): (False, False),
        (True, False): (True, False),
        (False, True): (False, True),
    }

    def __init__(self, args, strict):
        super(RangedExpression, self).__init__(args)
        self._strict = RangedExpression.STRICT[strict]

    def nargs(self):
        return 3

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def __getstate__(self):
        state = super(RangedExpression, self).__getstate__()
        for i in RangedExpression.__slots__:
            state[i] = getattr(self, i)
        return state

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

    @property
    def strict(self):
        return self._strict


class InequalityExpression(RelationalExpressionBase):
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
        super().__init__(args)
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

    def _apply_operation(self, result):
        _l, _r = result
        if self._strict:
            return _l < _r
        return _l <= _r

    def _to_string(self, values, verbose, smap, compute_values):
        if len(values) == 2:
            return "{0}  {1}  {2}".format(values[0], '<' if self._strict else '<=', values[1])

    @property
    def strict(self):
        return self._strict

    def __lt__(self, other):
        """
        Less than operator

        This method is called when Python processes statements of the form::

            self < other
            other > self
        """
        return inequality(*self._args_, other, (self.strict, True))

    def __gt__(self, other):
        """
        Greater than operator

        This method is called when Python processes statements of the form::

            self > other
            other < self
        """
        return inequality(other, *self._args_, (True, self.strict))

    def __le__(self, other):
        """
        Less than or equal operator

        This method is called when Python processes statements of the form::

            self <= other
            other >= self
        """
        return inequality(*self._args_, other, (self.strict, False))

    def __ge__(self, other):
        """
        Greater than or equal operator

        This method is called when Python processes statements of the form::

            self >= other
            other <= self
        """
        return inequality(other, *self._args_, (False, self.strict))


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
    return RangedExpression((lower, body, upper), strict)


class EqualityExpression(RelationalExpressionBase):
    """
    Equality expression::

        x == y
    """

    __slots__ = ()
    PRECEDENCE = 9

    def nargs(self):
        return 2

    def __bool__(self):
        lhs, rhs = self.args
        if lhs is rhs:
            return True
        return super().__bool__()

    def _apply_operation(self, result):
        _l, _r = result
        return _l == _r

    def _to_string(self, values, verbose, smap, compute_values):
        return "{0}  ==  {1}".format(values[0], values[1])


def _generate_relational_expression(etype, lhs, rhs):
    rhs_is_relational = False
    lhs_is_relational = False

    constant_lhs = True
    constant_rhs = True

    if lhs is not None and lhs.__class__ not in native_numeric_types:
        lhs = _process_arg(lhs)
        # Note: _process_arg can return a native type
        if lhs is not None and lhs.__class__ not in native_numeric_types:
            lhs_is_relational = lhs.is_relational()
            constant_lhs = False
    if rhs is not None and rhs.__class__ not in native_numeric_types:
        rhs = _process_arg(rhs)
        # Note: _process_arg can return a native type
        if rhs is not None and rhs.__class__ not in native_numeric_types:
            rhs_is_relational = rhs.is_relational()
            constant_rhs = False

    if constant_lhs and constant_rhs:
        if etype == _eq:
            return lhs == rhs
        elif etype == _le:
            return lhs <= rhs
        elif etype == _lt:
            return lhs < rhs
        else:
            raise ValueError("Unknown relational expression type '%s'" % etype)

    if etype == _eq:
        if lhs_is_relational or rhs_is_relational:
            raise TypeError(
                "Cannot create an EqualityExpression where one of the "
                "sub-expressions is a relational expression:\n"
                "    %s\n    {==}\n    %s" % (lhs, rhs,)
            )
        return EqualityExpression((lhs, rhs))
    else:
        if etype == _le:
            strict = False
        elif etype == _lt:
            strict = True
        else:
            raise DeveloperError(
                "Unknown relational expression type '%s'" % (etype,))
        if lhs_is_relational:
            if lhs.__class__ is InequalityExpression:
                if rhs_is_relational:
                    raise TypeError(
                        "Cannot create an InequalityExpression where both "
                        "sub-expressions are relational expressions:\n"
                        "    %s\n    {%s}\n    %s"
                        % (lhs, "<" if strict else "<=", rhs,))
                return RangedExpression(
                    lhs._args_ + (rhs,), (lhs._strict, strict))
            else:
                raise TypeError(
                    "Cannot create an InequalityExpression where one of the "
                    "sub-expressions is an equality or ranged expression:\n"
                    "    %s\n    {%s}\n    %s"
                    % (lhs, "<" if strict else "<=", rhs,))
        elif rhs_is_relational:
            if rhs.__class__ is InequalityExpression:
                return RangedExpression(
                    (lhs,) + rhs._args_, (strict, rhs._strict))
            else:
                raise TypeError(
                    "Cannot create an InequalityExpression where one of the "
                    "sub-expressions is an equality or ranged expression:\n"
                    "    %s\n    {%s}\n    %s"
                    % (lhs, "<" if strict else "<=", rhs,))
        else:
            return InequalityExpression((lhs, rhs), strict)

