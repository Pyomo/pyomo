# -*- coding: utf-8 -*-
# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import collections
import operator

from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
    native_numeric_types,
    check_if_numeric_type,
    value,
)

# Note: There is a circular dependence between numeric_expr and this
# module: this module would like to reuse/build on
# numeric_expr._categorize_arg_type(), and numeric_expr.NumericValue
# needs to call the relational dispatchers here.  Instead of ensuring
# that one of the modules is fully declared before importing into the
# other, we will have BOTH modules assume that the other module has NOT
# been declared.
import pyomo.core.expr.numeric_expr as numeric_expr

from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr.expr_common import (
    ExpressionType,
    RELATIONAL_ARG_TYPE as ARG_TYPE,
    _binary_op_dispatcher_type_mapping,
)
from pyomo.core.expr.visitor import polynomial_degree
from pyomo.core.pyomoobject import PyomoObject

# -------------------------------------------------------
#
# Expression classes
#
# -------------------------------------------------------


def _categorize_relational_arg_type(arg):
    """Attempt to categorize an unknown object type into a RELATIONAL_ARG_TYPE

    Note that this can return the following types:
    - MUTABLE
    - ASNUMERIC
    - INVALID
    - NATIVE
    - PARAM
    - OTHER
    - INEQUALITY
    - INVALID_RELATIONAL
    """
    arg_type = numeric_expr._categorize_arg_type(arg)
    if arg_type is ARG_TYPE.INVALID:
        if isinstance(arg, PyomoObject):
            if isinstance(arg, InequalityExpression):
                arg_type = ARG_TYPE.INEQUALITY
            else:
                arg_type = ARG_TYPE.INVALID_RELATIONAL
    elif arg_type > ARG_TYPE.NATIVE and arg_type != ARG_TYPE.PARAM:
        arg_type = ARG_TYPE.OTHER
    return arg_type


def _categorize_relational_arg_types(*args):
    return tuple(_categorize_relational_arg_type(arg) for arg in args)


class RelationalExpression(ExpressionBase, BooleanValue):
    __slots__ = ('_args_',)

    EXPRESSION_SYSTEM = ExpressionType.RELATIONAL

    def __init__(self, args):
        self._args_ = args

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
        return self._args_[: self.nargs()]

    @deprecated(
        "is_relational() is deprecated in favor of "
        "is_expression_type(ExpressionType.RELATIONAL)",
        version='6.4.3',
    )
    def is_relational(self):
        return self.is_expression_type(ExpressionType.RELATIONAL)

    def is_potentially_variable(self):
        return any(
            arg.__class__ not in native_numeric_types and arg.is_potentially_variable()
            for arg in self._args_
        )

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

    def __eq__(self, other):
        """
        Equal to operator

        This method is called when Python processes statements of the form::

            self == other
            other == self
        """
        return _eq_dispatcher[self.__class__, other.__class__](self, other)

    def __lt__(self, other):
        """
        Less than operator

        This method is called when Python processes statements of the form::

            self < other
            other > self
        """
        return _lt_dispatcher[self.__class__, other.__class__](self, other)

    def __gt__(self, other):
        """
        Greater than operator

        This method is called when Python processes statements of the form::

            self > other
            other < self
        """
        return _lt_dispatcher[other.__class__, self.__class__](other, self)

    def __le__(self, other):
        """
        Less than or equal operator

        This method is called when Python processes statements of the form::

            self <= other
            other >= self
        """
        return _le_dispatcher[self.__class__, other.__class__](self, other)

    def __ge__(self, other):
        """
        Greater than or equal operator

        This method is called when Python processes statements of the form::

            self >= other
            other <= self
        """
        return _le_dispatcher[other.__class__, self.__class__](other, self)


class RangedExpression(RelationalExpression):
    """
    Ranged expressions, which define relations with a lower and upper bound::

        x < y < z
        x <= y <= z

    args:
        args (tuple): child nodes
        strict (tuple): flags that indicate whether the inequalities are strict
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

    def _to_string(self, values, verbose, smap):
        return "%s  %s  %s  %s  %s" % (
            values[0],
            "<="[: 2 - self._strict[0]],
            values[1],
            "<="[: 2 - self._strict[1]],
            values[2],
        )

    @property
    def strict(self):
        return self._strict


class InequalityExpression(RelationalExpression):
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

    def _apply_operation(self, result):
        _l, _r = result
        if self._strict:
            return _l < _r
        return _l <= _r

    def _to_string(self, values, verbose, smap):
        return "%s  %s  %s" % (values[0], "<="[: 2 - self._strict], values[1])

    @property
    def strict(self):
        return self._strict


class TrivialRelationalExpression(InequalityExpression):
    """A trivial relational expression

    Note that an inequality is sufficient to induce infeasibility and is
    simpler to relax (in the Big-M sense) than an equality.

    Note that we do not want to provide a trivial equality constraint as
    that can confuse solvers like Ipopt into believing that the model
    has fewer degrees of freedom than it actually has.

    """

    __slots__ = ('_name',)
    singleton = {}

    def __new__(cls, name, args):
        if name not in cls.singleton:
            cls.singleton[name] = super().__new__(cls)
            super().__init__(cls.singleton[name], args, False)
            cls.singleton[name]._name = name
        return cls.singleton[name]

    def __init__(self, name, args):
        # note that the meat of __init__ is called as part of __new__ above.
        assert args == self.args

    def __deepcopy__(self, memo):
        # Prevent deepcopy from duplicating this object
        return self

    def __reduce__(self):
        return self.__class__, (self._name, self._args_)

    def _to_string(self, values, verbose, smap):
        return self._name


def inequality(lower=None, body=None, upper=None, strict=False):
    r"""A utility function that can be used to declare inequality and
    ranged inequality expressions.

    The expression::

        inequality(2, model.x)

    is equivalent to:

    .. math::

        2 \leq x

    and the expression::

        inequality(2, model.x, 3)

    is equivalent to:

    .. math::

        2 \leq x \leq 3

    .. note:: Pyomo does not support constructing
       :class:`RangedExpression` objects using Python's chained
       comparison syntax (``lb <= body <= ub``).

       Python relies on the pairwise comparisons returning intermediates
       that are convertible to :class:`bool`, which is incompatible with
       Pyomo's operator overloading.  Instead, :class:`RangedExpression`
       objects should be created using this function.

    :func:`inequality` allows for any (or all) of `lower`, `body`, and
    `upper` to be ``None``.  What gets returned from the function
    depends on the number and type of values provided:

    No arguments are ``None``
       If `lower`, `body`, and `upper` are all non-``None``, then the
       function will return a :class:`RangedExpression`, unless two
       adjacent arguments (either `lower` and `body`, or `body` and
       `upper`) are both constants (e.g., native numeric types or
       immutable :class:`Param` objects).  If there are two adjacent
       constant values, then that pair is evaluated.  If the result is
       ``False``, then the constraint is known to be trivially
       infeasible and :func:`inequality` will return ``False``.  If the
       result is ``True``, then the return value is as if the "outer"
       (`lower` or `upper`) argument had been ``None`` (see the following).

    One argument is ``None``
       If one of `lower`, `body`, or `upper` is ``None``, then
       :func:`inequality` will return an :class:`InequalityExpression`
       object, unless both arguments are constants (e.g., native numeric
       types or immutable :class:`Param` objects), in which case the
       expression will be evaluated and the resulting :class:`bool`
       returned.

    Two arguments are ``None``
       If two of `lower`, `body`, or `upper` are ``None``, then the
       remaining argument is returned from :func:`inequality`.

    All None
       If all arguments are ``None``, then ``None`` is returned


    Parameters
    ----------

    lower : int | float | NumericValue | None
        The expression for the lower bound

    body : int | float | NumericValue | None
        The expression for the body of a ranged constraint

    upper : int | float | NumericValue | None
        The expression for the upper bound

    strict : bool
        If ``True``, construct strict inequalities (``<``); otherwise
        use ``<=``.

    Returns
    -------
    RangedExpression | InequalityExpression | NumericValue | int | float | None

    """
    _genexpr = _lt_dispatcher if strict else _le_dispatcher
    if body is None:
        lower, body = body, lower
    expr = body
    if lower is not None:
        expr = _genexpr[lower.__class__, body.__class__](lower, body)
        # If the LHS of the ranged inequality evaluated to a bool, we
        # need to do some special processing:
        #  - False: the LHS is trivially infeasible.  Return False.
        #  - True: the LHS is trivially feasible. Leave body unchanged
        #    so that it can be compared to the RHS
        #
        # ...otherwise, "replace" body with the resulting inequality and
        # then process the RHS
        if expr.__class__ is bool:
            if not expr:
                return False
        else:
            body = expr
    if upper is not None:
        if body is not None:
            expr = _genexpr[body.__class__, upper.__class__](body, upper)
        else:
            expr = upper
    return expr


class EqualityExpression(RelationalExpression):
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

    def _to_string(self, values, verbose, smap):
        return "%s  ==  %s" % (values[0], values[1])


class NotEqualExpression(RelationalExpression):
    """
    Not-equal expression::

        x != y
    """

    __slots__ = ()

    def nargs(self):
        return 2

    def __bool__(self):
        lhs, rhs = self.args
        if lhs is not rhs:
            return True
        return super().__bool__()

    def _apply_operation(self, result):
        _l, _r = result
        return _l != _r

    def _to_string(self, values, verbose, smap):
        return "%s  !=  %s" % (values[0], values[1])


def tuple_to_relational_expr(args):
    if len(args) == 3:
        return inequality(*args, strict=False)
    elif len(args) == 2:
        lhs, rhs = args
        ans = _eq_dispatcher[lhs.__class__, rhs.__class__](lhs, rhs)
        if ans is NotImplemented:
            raise ValueError(
                "Cannot create EqualityExpression from argument types "
                f"'{type(lhs).__name__}' and '{type(rhs).__name__}'"
            )
        return ans
    raise ValueError(
        "Cannot convert tuple to relational expression. "
        f"Found a tuple of length {len(args)}. Expecting a tuple of "
        "length 2 or 3:\n"
        "    Equality:   (left, right)\n"
        "    Inequality: (lower, expression, upper)"
    )


def _invalid_relational(op_type, op_str, a, b):
    def no(*args):
        return False

    if getattr(a, 'is_expression_type', no)(ExpressionType.RELATIONAL):
        if getattr(b, 'is_expression_type', no)(ExpressionType.RELATIONAL):
            msg = (
                f"Cannot create an {op_type} where both "
                "sub-expressions are relational expressions:"
            )
        else:
            msg = (
                f"Cannot create an {op_type} where one of the "
                "sub-expressions is a relational expression:"
            )
    elif getattr(b, 'is_expression_type', no)(ExpressionType.RELATIONAL):
        msg = (
            f"Cannot create an {op_type} where one of the "
            "sub-expressions is a relational expression:"
        )
    elif getattr(a, 'is_component_type', no)() and a.is_indexed():
        msg = (
            f"Argument for {op_type} is an indexed numeric "
            f"value specified without an index:\n\t{a.name}\nIs this "
            "value defined over an index that you did not specify?"
        )
    elif getattr(b, 'is_component_type', no)() and b.is_indexed():
        msg = (
            f"Argument for {op_type} is an indexed numeric "
            f"value specified without an index:\n\t{b.name}\nIs this "
            "value defined over an index that you did not specify?"
        )
    else:
        msg = "Attempting to use a non-numeric type in a numeric expression context:"
    raise TypeError(msg + f"\n    {a}\n    {{{op_str}}}\n    {b}")


def _eq_invalid(a, b):
    _invalid_relational('EqualityExpression', '==', a, b)


def _eq_native(a, b):
    return a == b


def _eq_expr(a, b):
    return EqualityExpression((a, b))


def _eq_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a == b.value
    elif b.is_constant():
        b = b.value
    return EqualityExpression((a, b))


def _eq_param_any(a, b):
    if a.is_constant():
        return a.value == b
    return EqualityExpression((a, b))


def _eq_any_param(a, b):
    if b.is_constant():
        return a == b.value
    return EqualityExpression((a, b))


def _register_new_eq_handler(a, b):
    types = _categorize_relational_arg_types(a, b)
    # Retrieve the appropriate handler, record it in the main
    # _eq_dispatcher dict (so this method is not called a second time for
    # these types)
    _eq_dispatcher[a.__class__, b.__class__] = handler = _eq_type_handler_mapping[types]
    # Call the appropriate handler
    return handler(a, b)


_eq_dispatcher = collections.defaultdict(lambda: _register_new_eq_handler)
_eq_type_handler_mapping = _binary_op_dispatcher_type_mapping(
    _eq_dispatcher,
    {
        (ARG_TYPE.NATIVE, ARG_TYPE.NATIVE): _eq_native,
        (ARG_TYPE.NATIVE, ARG_TYPE.PARAM): _eq_any_param,
        (ARG_TYPE.NATIVE, ARG_TYPE.OTHER): _eq_expr,
        (ARG_TYPE.NATIVE, ARG_TYPE.INEQUALITY): _eq_invalid,
        (ARG_TYPE.NATIVE, ARG_TYPE.INVALID_RELATIONAL): _eq_invalid,
        (ARG_TYPE.PARAM, ARG_TYPE.NATIVE): _eq_param_any,
        (ARG_TYPE.PARAM, ARG_TYPE.PARAM): _eq_param_param,
        (ARG_TYPE.PARAM, ARG_TYPE.OTHER): _eq_param_any,
        (ARG_TYPE.PARAM, ARG_TYPE.INEQUALITY): _eq_invalid,
        (ARG_TYPE.PARAM, ARG_TYPE.INVALID_RELATIONAL): _eq_invalid,
        (ARG_TYPE.OTHER, ARG_TYPE.NATIVE): _eq_expr,
        (ARG_TYPE.OTHER, ARG_TYPE.PARAM): _eq_any_param,
        (ARG_TYPE.OTHER, ARG_TYPE.OTHER): _eq_expr,
        (ARG_TYPE.OTHER, ARG_TYPE.INEQUALITY): _eq_invalid,
        (ARG_TYPE.OTHER, ARG_TYPE.INVALID_RELATIONAL): _eq_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.NATIVE): _eq_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.PARAM): _eq_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.OTHER): _eq_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.INEQUALITY): _eq_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.INVALID_RELATIONAL): _eq_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.NATIVE): _eq_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.PARAM): _eq_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.OTHER): _eq_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.INEQUALITY): _eq_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.INVALID_RELATIONAL): _eq_invalid,
    },
    ARG_TYPE,
)


def _le_invalid(a, b):
    _invalid_relational('InequalityExpression', '<=', a, b)


def _le_native(a, b):
    return a <= b


def _le_expr(a, b):
    return InequalityExpression((a, b), False)


def _le_expr_ineq(a, b):
    return RangedExpression((a,) + b.args, (False, b._strict))


def _le_native_ineq(a, b):
    b0, b1 = b.args
    lhs = _le_dispatcher[a.__class__, b0.__class__](a, b0)
    if lhs.__class__ is InequalityExpression:
        bounds = (_lt_dispatcher if b.strict else _le_dispatcher)[
            a.__class__, b1.__class__
        ](a, b1)
        if bounds.__class__ is bool and not bounds:
            # Trivial infeasible bounds
            return False
        return RangedExpression((a, b0, b1), (False, b._strict))
    elif lhs:
        # Trivial (feasible) LHS
        return b
    else:
        # Trivial infeasible LHS
        return False


def _le_param_ineq(a, b):
    if a.is_constant():
        return _le_native_ineq(a.value, b)
    return RangedExpression((a,) + b.args, (False, b._strict))


def _le_ineq_expr(a, b):
    return RangedExpression(a.args + (b,), (a._strict, False))


def _le_ineq_native(a, b):
    a0, a1 = a.args
    rhs = _le_dispatcher[a1.__class__, b.__class__](a1, b)
    if rhs.__class__ is InequalityExpression:
        bounds = (_lt_dispatcher if a.strict else _le_dispatcher)[
            a0.__class__, b.__class__
        ](a0, b)
        if bounds.__class__ is bool and not bounds:
            # Trivial infeasible bounds
            return False
        return RangedExpression((a0, a1, b), (a._strict, False))
    elif rhs:
        # Trivial (feasible) RHS
        return a
    else:
        # Trivial infeasible RHS
        return False


def _le_ineq_param(a, b):
    if b.is_constant():
        return _le_ineq_native(a, b.value)
    return RangedExpression(a.args + (b,), (a._strict, False))


def _le_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a <= b.value
    elif b.is_constant():
        b = b.value
    return InequalityExpression((a, b), False)


def _le_param_any(a, b):
    if a.is_constant():
        return a.value <= b
    return InequalityExpression((a, b), False)


def _le_any_param(a, b):
    if b.is_constant():
        return a <= b.value
    return InequalityExpression((a, b), False)


def _register_new_le_handler(a, b):
    types = _categorize_relational_arg_types(a, b)
    # Retrieve the appropriate handler, record it in the main
    # _le_dispatcher dict (so this method is not called a second time for
    # these types)
    _le_dispatcher[a.__class__, b.__class__] = handler = _le_type_handler_mapping[types]
    # Call the appropriate handler
    return handler(a, b)


_le_dispatcher = collections.defaultdict(lambda: _register_new_le_handler)
_le_type_handler_mapping = _binary_op_dispatcher_type_mapping(
    _le_dispatcher,
    {
        (ARG_TYPE.NATIVE, ARG_TYPE.NATIVE): _le_native,
        (ARG_TYPE.NATIVE, ARG_TYPE.PARAM): _le_any_param,
        (ARG_TYPE.NATIVE, ARG_TYPE.OTHER): _le_expr,
        (ARG_TYPE.NATIVE, ARG_TYPE.INEQUALITY): _le_native_ineq,
        (ARG_TYPE.NATIVE, ARG_TYPE.INVALID_RELATIONAL): _le_invalid,
        (ARG_TYPE.PARAM, ARG_TYPE.NATIVE): _le_param_any,
        (ARG_TYPE.PARAM, ARG_TYPE.PARAM): _le_param_param,
        (ARG_TYPE.PARAM, ARG_TYPE.OTHER): _le_param_any,
        (ARG_TYPE.PARAM, ARG_TYPE.INEQUALITY): _le_param_ineq,
        (ARG_TYPE.PARAM, ARG_TYPE.INVALID_RELATIONAL): _le_invalid,
        (ARG_TYPE.OTHER, ARG_TYPE.NATIVE): _le_expr,
        (ARG_TYPE.OTHER, ARG_TYPE.PARAM): _le_any_param,
        (ARG_TYPE.OTHER, ARG_TYPE.OTHER): _le_expr,
        (ARG_TYPE.OTHER, ARG_TYPE.INEQUALITY): _le_expr_ineq,
        (ARG_TYPE.OTHER, ARG_TYPE.INVALID_RELATIONAL): _le_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.NATIVE): _le_ineq_native,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.PARAM): _le_ineq_param,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.OTHER): _le_ineq_expr,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.INEQUALITY): _le_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.INVALID_RELATIONAL): _le_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.NATIVE): _le_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.PARAM): _le_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.OTHER): _le_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.INEQUALITY): _le_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.INVALID_RELATIONAL): _le_invalid,
    },
    ARG_TYPE,
)


def _lt_invalid(a, b):
    _invalid_relational('InequalityExpression', '<', a, b)


def _lt_native(a, b):
    return a < b


def _lt_expr(a, b):
    return InequalityExpression((a, b), True)


def _lt_expr_ineq(a, b):
    return RangedExpression((a,) + b.args, (True, b._strict))


def _lt_native_ineq(a, b):
    b0, b1 = b.args
    lhs = _lt_dispatcher[a.__class__, b0.__class__](a, b0)
    if lhs.__class__ is InequalityExpression:
        bounds = _lt_dispatcher[a.__class__, b1.__class__](a, b1)
        if bounds.__class__ is bool and not bounds:
            # Trivial infeasible bounds
            return False
        return RangedExpression((a, b0, b1), (True, b._strict))
    elif lhs:
        # Trivial (feasible) LHS
        return b
    else:
        # Trivial infeasible LHS
        return False


def _lt_param_ineq(a, b):
    if a.is_constant():
        return _lt_native_ineq(a.value, b)
    return RangedExpression((a,) + b.args, (True, b._strict))


def _lt_ineq_expr(a, b):
    return RangedExpression(a.args + (b,), (a._strict, True))


def _lt_ineq_native(a, b):
    a0, a1 = a.args
    rhs = _lt_dispatcher[a1.__class__, b.__class__](a1, b)
    if rhs.__class__ is InequalityExpression:
        bounds = _lt_dispatcher[a0.__class__, b.__class__](a0, b)
        if bounds.__class__ is bool and not bounds:
            # Trivial infeasible bounds
            return False
        return RangedExpression((a0, a1, b), (a._strict, True))
    elif rhs:
        # Trivial (feasible) RHS
        return a
    else:
        # Trivial infeasible RHS
        return False


def _lt_ineq_param(a, b):
    if b.is_constant():
        return _lt_ineq_native(a, b.value)
    return RangedExpression(a.args + (b,), (a._strict, True))


def _lt_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a < b.value
    elif b.is_constant():
        b = b.value
    return InequalityExpression((a, b), True)


def _lt_param_any(a, b):
    if a.is_constant():
        return a.value < b
    return InequalityExpression((a, b), True)


def _lt_any_param(a, b):
    if b.is_constant():
        return a < b.value
    return InequalityExpression((a, b), True)


def _register_new_lt_handler(a, b):
    types = _categorize_relational_arg_types(a, b)
    # Retrieve the appropriate handler, record it in the main
    # _lt_dispatcher dict (so this method is not called a second time for
    # these types)
    _lt_dispatcher[a.__class__, b.__class__] = handler = _lt_type_handler_mapping[types]
    # Call the appropriate handler
    return handler(a, b)


_lt_dispatcher = collections.defaultdict(lambda: _register_new_lt_handler)
_lt_type_handler_mapping = _binary_op_dispatcher_type_mapping(
    _lt_dispatcher,
    {
        (ARG_TYPE.NATIVE, ARG_TYPE.NATIVE): _lt_native,
        (ARG_TYPE.NATIVE, ARG_TYPE.PARAM): _lt_any_param,
        (ARG_TYPE.NATIVE, ARG_TYPE.OTHER): _lt_expr,
        (ARG_TYPE.NATIVE, ARG_TYPE.INEQUALITY): _lt_native_ineq,
        (ARG_TYPE.NATIVE, ARG_TYPE.INVALID_RELATIONAL): _lt_invalid,
        (ARG_TYPE.PARAM, ARG_TYPE.NATIVE): _lt_param_any,
        (ARG_TYPE.PARAM, ARG_TYPE.PARAM): _lt_param_param,
        (ARG_TYPE.PARAM, ARG_TYPE.OTHER): _lt_param_any,
        (ARG_TYPE.PARAM, ARG_TYPE.INEQUALITY): _lt_param_ineq,
        (ARG_TYPE.PARAM, ARG_TYPE.INVALID_RELATIONAL): _lt_invalid,
        (ARG_TYPE.OTHER, ARG_TYPE.NATIVE): _lt_expr,
        (ARG_TYPE.OTHER, ARG_TYPE.PARAM): _lt_any_param,
        (ARG_TYPE.OTHER, ARG_TYPE.OTHER): _lt_expr,
        (ARG_TYPE.OTHER, ARG_TYPE.INEQUALITY): _lt_expr_ineq,
        (ARG_TYPE.OTHER, ARG_TYPE.INVALID_RELATIONAL): _lt_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.NATIVE): _lt_ineq_native,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.PARAM): _lt_ineq_param,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.OTHER): _lt_ineq_expr,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.INEQUALITY): _lt_invalid,
        (ARG_TYPE.INEQUALITY, ARG_TYPE.INVALID_RELATIONAL): _lt_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.NATIVE): _lt_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.PARAM): _lt_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.OTHER): _lt_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.INEQUALITY): _lt_invalid,
        (ARG_TYPE.INVALID_RELATIONAL, ARG_TYPE.INVALID_RELATIONAL): _lt_invalid,
    },
    ARG_TYPE,
)
