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

import collections
import enum
import logging
import math
import operator

logger = logging.getLogger('pyomo.core')

from math import isclose

from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.deprecation import (
    deprecated,
    deprecation_warning,
    relocated_module_attribute,
)
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
    native_types,
    nonpyomo_leaf_types,
    native_numeric_types,
    check_if_numeric_type,
    value,
)

from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
    OperatorAssociativity,
    ExpressionType,
    _lt,
    _le,
    _eq,
)

# Note: pyggyback on expr.base's use of attempt_import(visitor)
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor

relocated_module_attribute(
    'is_potentially_variable',
    'pyomo.core.expr.numvalue.is_potentially_variable',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'as_numeric',
    'pyomo.core.expr.numvalue.as_numeric',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'clone_counter',
    'pyomo.core.expr.expr_common.clone_counter',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'evaluate_expression',
    'pyomo.core.expr.visitor.evaluate_expression',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'expression_to_string',
    'pyomo.core.expr.visitor.expression_to_string',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'polynomial_degree',
    'pyomo.core.expr.visitor.polynomial_degree',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'clone_expression',
    'pyomo.core.expr.visitor.clone_expression',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'sizeof_expression',
    'pyomo.core.expr.visitor.sizeof_expression',
    version='6.6.2',
    f_globals=globals(),
)

_zero_one_optimizations = {1}


# Stub in the dispatchers
def _generate_relational_expression(etype, lhs, rhs):
    raise RuntimeError("incomplete import of Pyomo expression system")


def enable_expression_optimizations(zero=None, one=None):
    """Enable(disable) expression generation optimizations

    There are currently two optimizations available during expression generation:

    - zero: aggressively resolve `0*f(.)` expressions to `0`, `0/f(.)`
      expressions to `0`, and `f(.)**0` expressions to `1`

    - one: aggressively resolve identities: `1*f(.)` expressions to
      `f(.)`, `f(.)/1` expressions to `f(.)`, and `f(.)**1` expressions
      to `f(.)`.

    The default optimizations are `zero=False` and `one=True`.

    Notes
    -----

    Enabling the `zero` optimization can mask certain modeling errors.
    In particular, the optimization will suppress `ZeroDivisionError`s
    that should be raised if `f(.)` resolves to `0` (in the case of
    `0/f(.)`), as well as any errors that would have otherwise been
    raised during the evaluation of `f(.)`.  In addition, optimizing
    `f(.)**0 == 1` is only valid when `f(.)!=0`.  **Users who enable
    this optimization bear responsibility for ensuring that these
    optimizations will be valid for the model.**

    The `one` optimizations should generally be safe.

    Parameters
    ----------
    zero: bool, optional

        If `True` (`False`), enable (disable) the "zero" optimizations.
        If None, leave the optimization state unchanged.

    one: bool, optional

        If `True` (`False`), enable (disable) the "one" optimizations.
        If None, leave the optimization state unchanged.

    """
    for arg, key in ((zero, 0), (one, 1)):
        if arg is None:
            continue
        if arg:
            _zero_one_optimizations.add(key)
        else:
            _zero_one_optimizations.discard(key)


class mutable_expression(object):
    """Context manager for mutable sums.

    This context manager is used to compute a sum while treating the
    summation as a mutable object.

    """

    def __enter__(self):
        self.e = _MutableNPVSumExpression([])
        return self.e

    def __exit__(self, *args):
        if isinstance(self.e, _MutableSumExpression):
            self.e.make_immutable()


class nonlinear_expression(mutable_expression):
    """Context manager for mutable nonlinear sums.

    This context manager is used to compute a general nonlinear sum
    while treating the summation as a mutable object.

    Note
    ----

    The preferred context manager is :py:class:`mutable_expression`, as
    the return type will be the most specific of
    :py:class:`SumExpression`, :py:class:`LinearExpression`, or
    :py:class:`NPV_SumExpression`.  This context manager will *always*
    return a :py:class:`SumExpression`.

    """

    def __enter__(self):
        self.e = _MutableSumExpression([])
        return self.e


class linear_expression(mutable_expression):
    """Context manager for mutable linear sums.

    This context manager is used to compute a linear sum while
    treating the summation as a mutable object.

    Note
    ----

    The preferred context manager is :py:class:`mutable_expression`.
    :py:class:`linear_expression` is an alias to
    :py:class:`mutable_expression` provided for backwards compatibility.

    """


class NumericValue(PyomoObject):
    """
    This is the base class for numeric values used in Pyomo.
    """

    __slots__ = ()

    # This is required because we define __eq__
    __hash__ = None

    def getname(self, fully_qualified=False, name_buffer=None):
        """
        If this is a component, return the component's name on the owning
        block; otherwise return the value converted to a string
        """
        _base = super(NumericValue, self)
        if hasattr(_base, 'getname'):
            return _base.getname(fully_qualified, name_buffer)
        else:
            return str(type(self))

    @property
    def name(self):
        return self.getname(fully_qualified=True)

    @property
    def local_name(self):
        return self.getname(fully_qualified=False)

    def is_numeric_type(self):
        """Return True if this class is a Pyomo numeric object"""
        return True

    def is_constant(self):
        """Return True if this numeric value is a constant value"""
        return False

    def is_fixed(self):
        """Return True if this is a non-constant value that has been fixed"""
        return False

    def is_potentially_variable(self):
        """Return True if variables can appear in this expression"""
        return False

    @deprecated(
        "is_relational() is deprecated in favor of "
        "is_expression_type(ExpressionType.RELATIONAL)",
        version='6.4.3',
    )
    def is_relational(self):
        """
        Return True if this numeric value represents a relational expression.
        """
        return False

    def is_indexed(self):
        """Return True if this numeric value is an indexed object"""
        return False

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            :const:`None`
        """
        return self._compute_polynomial_degree(None)

    def _compute_polynomial_degree(self, values):
        """
        Compute the polynomial degree of this expression given
        the degree values of its children.

        Args:
            values (list): A list of values that indicate the degree
                of the children expression.

        Returns:
            :const:`None`
        """
        return None

    def __bool__(self):
        """Coerce the value to a bool

        Numeric values can be coerced to bool only if the value /
        expression is constant.  Fixed (but non-constant) or variable
        values will raise an exception.

        Raises:
            PyomoException

        """
        # Note that we want to implement __bool__, as scalar numeric
        # components (e.g., Param, Var) implement __len__ (since they
        # are implicit containers), and Python falls back on __len__ if
        # __bool__ is not defined.
        if self.is_constant():
            return bool(self())
        raise PyomoException(
            """
Cannot convert non-constant Pyomo numeric value (%s) to bool.
This error is usually caused by using a Var, unit, or mutable Param in a
Boolean context such as an "if" statement. For example,
    >>> m.x = Var()
    >>> if not m.x:
    ...     pass
would cause this exception.""".strip()
            % (self,)
        )

    def __float__(self):
        """Coerce the value to a floating point

        Numeric values can be coerced to float only if the value /
        expression is constant.  Fixed (but non-constant) or variable
        values will raise an exception.

        Raises:
            TypeError

        """
        if self.is_constant():
            return float(self())
        raise TypeError(
            """
Implicit conversion of Pyomo numeric value (%s) to float is disabled.
This error is often the result of using Pyomo components as arguments to
one of the Python built-in math module functions when defining
expressions. Avoid this error by using Pyomo-provided math functions or
explicitly resolving the numeric value using the Pyomo value() function.
""".strip()
            % (self,)
        )

    def __int__(self):
        """Coerce the value to an integer

        Numeric values can be coerced to int only if the value /
        expression is constant.  Fixed (but non-constant) or variable
        values will raise an exception.

        Raises:
            TypeError

        """
        if self.is_constant():
            return int(self())
        raise TypeError(
            """
Implicit conversion of Pyomo numeric value (%s) to int is disabled.
This error is often the result of using Pyomo components as arguments to
one of the Python built-in math module functions when defining
expressions. Avoid this error by using Pyomo-provided math functions or
explicitly resolving the numeric value using the Pyomo value() function.
""".strip()
            % (self,)
        )

    def __lt__(self, other):
        """
        Less than operator

        This method is called when Python processes statements of the form::

            self < other
            other > self
        """
        return _generate_relational_expression(_lt, self, other)

    def __gt__(self, other):
        """
        Greater than operator

        This method is called when Python processes statements of the form::

            self > other
            other < self
        """
        return _generate_relational_expression(_lt, other, self)

    def __le__(self, other):
        """
        Less than or equal operator

        This method is called when Python processes statements of the form::

            self <= other
            other >= self
        """
        return _generate_relational_expression(_le, self, other)

    def __ge__(self, other):
        """
        Greater than or equal operator

        This method is called when Python processes statements of the form::

            self >= other
            other <= self
        """
        return _generate_relational_expression(_le, other, self)

    def __eq__(self, other):
        """
        Equal to operator

        This method is called when Python processes the statement::

            self == other
        """
        return _generate_relational_expression(_eq, self, other)

    def __add__(self, other):
        """
        Binary addition

        This method is called when Python processes the statement::

            self + other
        """
        return _add_dispatcher[self.__class__, other.__class__](self, other)

    def __sub__(self, other):
        """
        Binary subtraction

        This method is called when Python processes the statement::

            self - other
        """
        return self.__add__(-other)

    def __mul__(self, other):
        """
        Binary multiplication

        This method is called when Python processes the statement::

            self * other
        """
        return _mul_dispatcher[self.__class__, other.__class__](self, other)

    def __div__(self, other):
        """
        Binary division

        This method is called when Python processes the statement::

            self / other
        """
        return _div_dispatcher[self.__class__, other.__class__](self, other)

    def __truediv__(self, other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::

            self / other
        """
        return _div_dispatcher[self.__class__, other.__class__](self, other)

    def __pow__(self, other):
        """
        Binary power

        This method is called when Python processes the statement::

            self ** other
        """
        return _pow_dispatcher[self.__class__, other.__class__](self, other)

    def __radd__(self, other):
        """
        Binary addition

        This method is called when Python processes the statement::

            other + self
        """
        return _add_dispatcher[other.__class__, self.__class__](other, self)

    def __rsub__(self, other):
        """
        Binary subtraction

        This method is called when Python processes the statement::

            other - self
        """
        return other + (-self)

    def __rmul__(self, other):
        """
        Binary multiplication

        This method is called when Python processes the statement::

            other * self

        when other is not a :class:`NumericValue <pyomo.core.expr.numvalue.NumericValue>` object.
        """
        return _mul_dispatcher[other.__class__, self.__class__](other, self)

    def __rdiv__(self, other):
        """Binary division

        This method is called when Python processes the statement::

            other / self
        """
        return _div_dispatcher[other.__class__, self.__class__](other, self)

    def __rtruediv__(self, other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::

            other / self
        """
        return _div_dispatcher[other.__class__, self.__class__](other, self)

    def __rpow__(self, other):
        """
        Binary power

        This method is called when Python processes the statement::

            other ** self
        """
        return _pow_dispatcher[other.__class__, self.__class__](other, self)

    def __iadd__(self, other):
        """
        Binary addition

        This method is called when Python processes the statement::

            self += other
        """
        return _add_dispatcher[self.__class__, other.__class__](self, other)

    def __isub__(self, other):
        """
        Binary subtraction

        This method is called when Python processes the statement::

            self -= other
        """
        return self.__iadd__(-other)

    def __imul__(self, other):
        """
        Binary multiplication

        This method is called when Python processes the statement::

            self *= other
        """
        return _mul_dispatcher[self.__class__, other.__class__](self, other)

    def __idiv__(self, other):
        """
        Binary division

        This method is called when Python processes the statement::

            self /= other
        """
        return _div_dispatcher[self.__class__, other.__class__](self, other)

    def __itruediv__(self, other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::

            self /= other
        """
        return _div_dispatcher[self.__class__, other.__class__](self, other)

    def __ipow__(self, other):
        """
        Binary power

        This method is called when Python processes the statement::

            self **= other
        """
        return _pow_dispatcher[self.__class__, other.__class__](self, other)

    def __neg__(self):
        """
        Negation

        This method is called when Python processes the statement::

            - self
        """
        return _neg_dispatcher[self.__class__](self)

    def __pos__(self):
        """
        Positive expression

        This method is called when Python processes the statement::

            + self
        """
        return self

    def __abs__(self):
        """Absolute value

        This method is called when Python processes the statement::

            abs(self)
        """
        return _abs_dispatcher[self.__class__](self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NumericNDArray.__array_ufunc__(None, ufunc, method, *inputs, **kwargs)

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """Return a string representation of the expression tree.

        Args:
            verbose (bool): If :const:`True`, then the string
                representation consists of nested functions.  Otherwise,
                the string representation is an infix algebraic equation.
                Defaults to :const:`False`.
            labeler: An object that generates string labels for
                non-constant in the expression tree.  Defaults to
                :const:`None`.
            smap: A SymbolMap instance that stores string labels for
                non-constant nodes in the expression tree.  Defaults to
                :const:`None`.
            compute_values (bool): If :const:`True`, then fixed
                expressions are evaluated and the string representation
                of the resulting value is returned.

        Returns:
            A string representation for the expression tree.

        """
        if compute_values and self.is_fixed():
            try:
                return str(self())
            except:
                pass
        if not self.is_constant():
            if smap is not None:
                return smap.getSymbol(self, labeler)
            elif labeler is not None:
                return labeler(self)
        return str(self)


#
# Note: the "if numpy_available" in the class definition also ensures
# that the numpy types are registered if numpy is in fact available
#
# TODO: Move this to a separate module to support avoiding the numpy
# import if numpy is not actually used.
class NumericNDArray(np.ndarray if numpy_available else object):
    """An ndarray subclass that stores Pyomo numeric expressions"""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            # Convert all incoming types to ndarray (to prevent recursion)
            args = [np.asarray(i) for i in inputs]
            # Set the return type to be an 'object'.  This prevents the
            # logical operators from casting the result to a bool.  This
            # requires numpy >= 1.6
            kwargs['dtype'] = object

        # Delegate to the base ufunc, but return an instance of this
        # class so that additional operators hit this method.
        ans = getattr(ufunc, method)(*args, **kwargs)
        if isinstance(ans, np.ndarray):
            if ans.size == 1:
                return ans[0]
            return ans.view(NumericNDArray)
        else:
            return ans


# -------------------------------------------------------
#
# Expression classes
#
# -------------------------------------------------------


class NumericExpression(ExpressionBase, NumericValue):
    """
    The base class for Pyomo expressions.

    This class is used to define nodes in a numeric expression
    tree.

    Args:
        args (list or tuple): Children of this node.
    """

    # Previously, we used _args to define expression class arguments.
    # Here, we use _args_ to force errors for code that was referencing this
    # data.  There are now accessor methods, so in most cases users
    # and developers should not directly access the _args_ data values.
    __slots__ = ('_args_',)
    EXPRESSION_SYSTEM = ExpressionType.NUMERIC
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        # by default, Pyomo numeric operators are binary operators
        return 2

    @property
    def args(self):
        """
        Return the child nodes

        Returns
        -------
        list or tuple:
            Sequence containing only the child nodes of this node.  The
            return type depends on the node storage model.  Users are
            not permitted to change the returned data (even for the case
            of data returned as a list), as that breaks the promise of
            tree immutability.
        """
        return self._args_

    @deprecated(
        'The implicit recasting of a "not potentially variable" '
        'expression node to a potentially variable one is no '
        'longer supported (this violates that immutability '
        'promise for Pyomo5 expression trees).',
        version='6.4.3',
    )
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
        if not self.is_potentially_variable():
            logger.error(
                'recasting a non-potentially variable expression to a '
                'potentially variable one violates the immutability '
                'promise for Pyomo expression trees.'
            )
            self.__class__ = self.potentially_variable_base_class()
        return self

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return visitor.polynomial_degree(self)

    def _compute_polynomial_degree(self, values):
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
        if all(val == 0 for val in values):
            return 0
        else:
            return None


class Numeric_NPV_Mixin(NPV_Mixin):
    __slots__ = ()

    def potentially_variable_base_class(self):
        cls = list(self.__class__.__bases__)
        cls.remove(Numeric_NPV_Mixin)
        assert len(cls) == 1
        return cls[0]

    #
    # Special cases: unary operators on NPV expressions are NPV
    #
    def __neg__(self):
        return NPV_NegationExpression((self,))

    def __abs__(self):
        return NPV_AbsExpression((self,))


class NegationExpression(NumericExpression):
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

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({values[0]})"
        tmp = values[0]
        if tmp[0] == '-':
            return tmp[1:].strip()
        # TODO: remove space after negation
        return "- " + tmp

    def _apply_operation(self, result):
        return -result[0]

    def __neg__(self):
        return self._args_[0]


class NPV_NegationExpression(Numeric_NPV_Mixin, NegationExpression):
    __slots__ = ()

    # Because NPV also defines __neg__ we need to override it here, too
    def __neg__(self):
        return self._args_[0]


class ExternalFunctionExpression(NumericExpression):
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

    # This operator does not have an infix representation
    PRECEDENCE = None

    def __init__(self, args, fcn=None):
        self._args_ = args
        self._fcn = fcn

    def nargs(self):
        return len(self._args_)

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        return classtype(args, self._fcn)

    def getname(self, *args, **kwds):
        return self._fcn.getname(*args, **kwds)

    def _apply_operation(self, result):
        return self._fcn.evaluate(result)

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"

    def get_arg_units(self):
        """Return the units for this external functions arguments"""
        return self._fcn.get_arg_units()

    def get_units(self):
        """Get the units of the return value for this external function"""
        return self._fcn.get_units()


class NPV_ExternalFunctionExpression(Numeric_NPV_Mixin, ExternalFunctionExpression):
    __slots__ = ()


class PowExpression(NumericExpression):
    """
    Power expressions::

        x**y
    """

    __slots__ = ()
    PRECEDENCE = 2

    # "**" is right-to-left associative in Python (so this should
    # return -1), however, as this rule is not widely known and can
    # confuse novice users, we will make our "**" operator
    # non-associative (forcing parens)
    ASSOCIATIVITY = OperatorAssociativity.NON_ASSOCIATIVE

    def _compute_polynomial_degree(self, result):
        # PowExpression is a tricky thing.  In general, a**b is
        # nonpolynomial, however, if b == 0, it is a constant
        # expression, and if a is polynomial and b is a positive
        # integer, it is also polynomial.  While we would like to just
        # call this a non-polynomial expression, these exceptions occur
        # too frequently (and in particular, a**2)
        l, r = result
        if r == 0:
            # NOTE: use value before int() so that we don't
            #       run into the disabled __int__ method on
            #       NumericValue
            exp = value(self._args_[1], exception=False)
            if exp is None:
                return None
            if exp == int(exp):
                if not exp:
                    return 0
                if l is not None and exp > 0:
                    return l * int(exp)
        return None

    def _is_fixed(self, args):
        if not args[1]:
            return False
        return args[0] or value(self._args_[1], exception=False) == 0

    def _apply_operation(self, result):
        _l, _r = result
        return _l**_r

    def getname(self, *args, **kwds):
        return 'pow'

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        return f"{values[0]}**{values[1]}"


class NPV_PowExpression(Numeric_NPV_Mixin, PowExpression):
    __slots__ = ()


class MaxExpression(NumericExpression):
    """
    Maximum expressions::

        max(x, y, ...)
    """

    __slots__ = ()

    # This operator does not have an infix representation
    PRECEDENCE = None

    def nargs(self):
        return len(self._args_)

    def _apply_operation(self, result):
        return max(result)

    def getname(self, *args, **kwds):
        return 'max'

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"


class NPV_MaxExpression(Numeric_NPV_Mixin, MaxExpression):
    __slots__ = ()


class MinExpression(NumericExpression):
    """
    Minimum expressions::

        min(x, y, ...)
    """

    __slots__ = ()

    # This operator does not have an infix representation
    PRECEDENCE = None

    def nargs(self):
        return len(self._args_)

    def _apply_operation(self, result):
        return min(result)

    def getname(self, *args, **kwds):
        return 'min'

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"


class NPV_MinExpression(Numeric_NPV_Mixin, MinExpression):
    __slots__ = ()


class ProductExpression(NumericExpression):
    """
    Product expressions::

        x*y
    """

    __slots__ = ()
    PRECEDENCE = 4

    def _compute_polynomial_degree(self, result):
        # NB: We can't use sum() here because None (non-polynomial)
        # overrides a numeric value (and sum() just ignores it - or
        # errors in py3k)
        a, b = result
        if a == 0 and value(self._args_[0], exception=False) == 0:
            return 0
        if b == 0 and value(self._args_[1], exception=False) == 0:
            return 0
        if a is None or b is None:
            return None
        else:
            return a + b

    def getname(self, *args, **kwds):
        return 'prod'

    def _is_fixed(self, args):
        # Anything times 0 equals 0, so one of the children is
        # fixed and has a value of 0, then this expression is fixed
        if all(args):
            return True
        for i in (0, 1):
            if args[i] and value(self._args_[i], exception=False) == 0:
                return True
        return False

    def _apply_operation(self, result):
        _l, _r = result
        return _l * _r

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        if values[0] in self._to_string.one:
            return values[1]
        # TODO: remove space after negation
        if values[0] in self._to_string.minus_one:
            return f"- {values[1]}"
        return f"{values[0]}*{values[1]}"

    # Store these reference sets on the function for quick lookup
    _to_string.one = {"1", "1.0", "(1)", "(1.0)"}
    _to_string.minus_one = {"-1", "-1.0", "(-1)", "(-1.0)"}


class NPV_ProductExpression(Numeric_NPV_Mixin, ProductExpression):
    __slots__ = ()


class MonomialTermExpression(ProductExpression):
    __slots__ = ()

    def getname(self, *args, **kwds):
        return 'mon'

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            # Because monomial terms place requirements on the argument
            # types, the simplest / fastest thing to do is just defer to
            # the operator dispatcher.
            return operator.mul(*args)
        return self.__class__(args)


class DivisionExpression(NumericExpression):
    """
    Division expressions::

        x/y
    """

    __slots__ = ()
    PRECEDENCE = 4

    def _compute_polynomial_degree(self, result):
        if result[1] == 0:
            return result[0]
        return None

    def getname(self, *args, **kwds):
        return 'div'

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        return f"{values[0]}/{values[1]}"

    def _apply_operation(self, result):
        return result[0] / result[1]


class NPV_DivisionExpression(Numeric_NPV_Mixin, DivisionExpression):
    __slots__ = ()


class SumExpression(NumericExpression):
    """
    Sum expression::

        x + y + ...

    This node represents an "n-ary" sum expression over at least 2 arguments.

    Args:
        args (list): Children nodes

    """

    __slots__ = ('_nargs',)
    PRECEDENCE = 6

    def __init__(self, args):
        # unlike other expressions, we expect (require) args to be a list
        if args.__class__ is not list:
            args = list(args)
        self._args_ = args
        self._nargs = len(args)

    def nargs(self):
        return self._nargs

    @property
    def args(self):
        if len(self._args_) != self._nargs:
            self._args_ = self._args_[: self._nargs]
        return self._args_

    def getname(self, *args, **kwds):
        return 'sum'

    def _apply_operation(self, result):
        return sum(result)

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

    def _to_string(self, values, verbose, smap):
        if not values:
            values = ['0']
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        term = values[0]
        # TODO: remove space after negation for first term
        # if term[0] in '-+':
        #     values[0] = term[0] + term[1:].strip()
        for i in range(1, len(values)):
            term = values[i]
            # TODO: remove unnecessary parenthetical grouping
            # (addition is the lowest priority algebraic operator, so if
            # a term is enclosed in parens, then it is a nested sum.)
            # if term[0] == '(' and term[-1] == ')' and _balanced_parens(term[1:-1]):
            #     term = term[1:-1]
            if term[0] in '-+':
                values[i] = term[0] + ' ' + term[1:].strip()
            else:
                values[i] = '+ ' + term.strip()
        return ' '.join(values)

    @deprecated(
        "SumExpression.add() is deprecated.  Please use regular Python operators "
        "(infix '+' or inplace '+='.)",
        version='6.6.0',
    )
    def add(self, new_arg):
        self += new_arg
        return self


# TODO: deprecate this class name
SumExpressionBase = SumExpression


class LinearExpression(SumExpression):
    """An expression object for linear polynomials.

    This is a derived :py:class`SumExpression` that guarantees all
    arguments are either not potentially variable (e.g., native types,
    Params, or NPV expressions) OR :py:class:`MonomialTermExpression`
    objects.

    Args:
        args (tuple): Children nodes

    """

    __slots__ = ()

    _allowable_linear_expr_arg_types = set([MonomialTermExpression])
    _cache = (None, None, None, None)

    def __init__(self, args=None, constant=None, linear_coefs=None, linear_vars=None):
        """A linear expression of the form `const + sum_i(c_i*x_i)`.

        You can specify `args` OR (`constant`, `linear_coefs`, and
        `linear_vars`).  If `args` is provided, it should be a list that
        contains only constants, NPV objects/expressions, or
        :py:class:`MonomialTermExpression` objects.  Alternatively, you
        can specify the constant, the list of linear_coefs and the list
        of linear_vars separately.  Note that these lists are NOT
        preserved.

        """
        # I am not sure why LinearExpression allows omitting args, but
        # it does.  If they are provided, they should be the (non-zero)
        # constant followed by MonomialTermExpressions.
        if args is not None:
            if not (constant is None and linear_coefs is None and linear_vars is None):
                raise ValueError(
                    "Cannot specify both args and any of "
                    "{constant, linear_coefs, or linear_vars}"
                )
            # unlike other expressions, we expect (require) args to be a list
            if args.__class__ is not list:
                args = list(args)
            self._args_ = args
        else:
            self._args_ = []
            if constant is not None:
                # Filter 0, but only if it is a native type
                if constant.__class__ not in native_types or constant:
                    self._args_.append(constant)
            if linear_vars is not None:
                if linear_coefs is None or len(linear_vars) != len(linear_coefs):
                    raise ValueError(
                        f"linear_vars ({tostr(linear_vars)}) is not compatible "
                        f"with linear_coefs ({tostr(linear_coefs)})"
                    )
                self._args_.extend(
                    map(MonomialTermExpression, zip(linear_coefs, linear_vars))
                )
        self._nargs = len(self._args_)

    def _build_cache(self):
        const = 0
        coef = []
        var = []
        for arg in self.args:
            if arg.__class__ is MonomialTermExpression:
                coef.append(arg._args_[0])
                var.append(arg._args_[1])
            else:
                const += arg
        LinearExpression._cache = (self, const, coef, var)

    @property
    def constant(self):
        if LinearExpression._cache[0] is not self:
            self._build_cache()
        return LinearExpression._cache[1]

    @property
    def linear_coefs(self):
        if LinearExpression._cache[0] is not self:
            self._build_cache()
        return LinearExpression._cache[2]

    @property
    def linear_vars(self):
        if LinearExpression._cache[0] is not self:
            self._build_cache()
        return LinearExpression._cache[3]

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        if type(args) is not list:
            args = list(args)
        for i, arg in enumerate(args):
            if arg.__class__ in self._allowable_linear_expr_arg_types:
                # 99% of the time, the arg type hasn't changed
                continue
            elif arg.__class__ in native_numeric_types:
                # native numbers are OK (that's part of the constant)
                pass
            elif not arg.is_potentially_variable():
                # NPV expressions are OK
                pass
            elif arg.is_variable_type():
                # vars are OK, but need to be mapped to monomial terms
                args[i] = MonomialTermExpression((1, arg))
                continue
            else:
                # For anything else, convert this to a general sum
                classtype = SumExpression
                break
            # We get here for new types (likely NPV types) --
            # remember them for when they show up again
            self._allowable_linear_expr_arg_types.add(arg.__class__)
        return super().create_node_with_local_data(args, classtype)


class NPV_SumExpression(Numeric_NPV_Mixin, LinearExpression):
    __slots__ = ()


class _MutableSumExpression(SumExpression):
    """
    A mutable SumExpression

    The :func:`add` method is slightly different in that it
    does not create a new sum expression, but modifies the
    :attr:`_args_` data in place.
    """

    __slots__ = ()

    def make_immutable(self):
        self.__class__ = SumExpression

    def __iadd__(self, other):
        return _iadd_mutablesum_dispatcher[other.__class__](self, other)


class _MutableLinearExpression(_MutableSumExpression):
    __slots__ = ()

    def make_immutable(self):
        self.__class__ = LinearExpression

    def __iadd__(self, other):
        return _iadd_mutablelinear_dispatcher[other.__class__](self, other)


class _MutableNPVSumExpression(_MutableLinearExpression):
    __slots__ = ()

    def make_immutable(self):
        self.__class__ = NPV_SumExpression

    def __iadd__(self, other):
        return _iadd_mutablenpvsum_dispatcher[other.__class__](self, other)


class Expr_ifExpression(NumericExpression):
    """A numeric ternary (if-then-else) expression::

        Expr_if(IF=x, THEN=y, ELSE=z)

    Note that this is a mixed expression: `IF` can be numeric or logical;
    `THEN` and `ELSE` are numeric, and the result is a numeric expression.

    """

    __slots__ = ()

    # This operator does not have an infix representation
    PRECEDENCE = None

    # **NOTE**: This class evaluates the branching "_if" expression
    #           on a number of occasions. It is important that
    #           one uses __call__ for value() and NOT bool().

    def nargs(self):
        return 3

    def getname(self, *args, **kwds):
        return "Expr_if"

    def _is_fixed(self, args):
        if args[0]:  # if.is_fixed():
            if value(self._args_[0]):
                return args[1]  # then.is_fixed()
            else:
                return args[2]  # else.is_fixed()
        else:
            return False

    def _compute_polynomial_degree(self, result):
        _if, _then, _else = result
        if _if == 0:
            if _then == _else:
                # It doesn't matter which branch is active
                return _then
            val = value(self.arg(0), exception=False)
            if val is not None:
                return _then if val else _else
        return None

    def _to_string(self, values, verbose, smap):
        return (
            f'{self.getname()}( ( {values[0]} ), then=( {values[1]} ), '
            f'else=( {values[2]} ) )'
        )

    def _apply_operation(self, result):
        _if, _then, _else = result
        return _then if _if else _else


class NPV_Expr_ifExpression(Numeric_NPV_Mixin, Expr_ifExpression):
    __slots__ = ()


class UnaryFunctionExpression(NumericExpression):
    """
    An expression object for intrinsic (math) functions (e.g. sin, cos, tan).

    Args:
        args (tuple): Children nodes
        name (string): The function name
        fcn: The function that is used to evaluate this expression
    """

    __slots__ = ('_fcn', '_name')

    # This operator does not have an infix representation
    PRECEDENCE = None

    def __init__(self, args, name=None, fcn=None):
        self._args_ = args
        self._name = name
        self._fcn = fcn

    def nargs(self):
        return 1

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        return classtype(args, self._name, self._fcn)

    def getname(self, *args, **kwds):
        return self._name

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"

    def _apply_operation(self, result):
        return self._fcn(result[0])


class NPV_UnaryFunctionExpression(Numeric_NPV_Mixin, UnaryFunctionExpression):
    __slots__ = ()


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

    def create_node_with_local_data(self, args, classtype=None):
        # Because this class removes arguments from the __init__, we
        # also need to reimplement create_node_with_local_data to not
        # add those arguments here.
        if classtype is None:
            classtype = self.__class__
        return classtype(args)


class NPV_AbsExpression(Numeric_NPV_Mixin, AbsExpression):
    __slots__ = ()


# -----------------------------------------------------------------
#
# Functions for decomposing a linear expression into linear terms
#
# -----------------------------------------------------------------


def decompose_term(expr):
    """A function that returns a tuple consisting of (1) a flag indicating
    whether the expression is linear, and (2) a list of tuples that
    represents the terms in the linear expression.

    Args:
        expr (expression): The root node of an expression tree

    Returns:
        A tuple with the form ``(flag, list)``.  If :attr:`flag` is
        :const:`False`, then a nonlinear term has been found, and
        :const:`list` is :const:`None`.  Otherwise, :const:`list` is a
        list of tuples: ``(coef, value)``.  If :attr:`value` is
        :const:`None`, then this represents a constant term with value
        :attr:`coef`.  Otherwise, :attr:`value` is a variable object,
        and :attr:`coef` is the numeric coefficient.

    """
    if expr.__class__ in nonpyomo_leaf_types or not expr.is_potentially_variable():
        return True, [(expr, None)]
    elif expr.is_variable_type():
        return True, [(1, expr)]
    else:
        try:
            terms = [t_ for t_ in _decompose_linear_terms(expr)]
            return True, terms
        except LinearDecompositionError:
            return False, None


class LinearDecompositionError(Exception):
    pass


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
        yield (multiplier * expr, None)
    elif expr.is_variable_type():
        yield (multiplier, expr)
    elif expr.__class__ is MonomialTermExpression:
        yield (multiplier * expr._args_[0], expr._args_[1])
    elif expr.__class__ is ProductExpression:
        if (
            expr._args_[0].__class__ in native_numeric_types
            or not expr._args_[0].is_potentially_variable()
        ):
            yield from _decompose_linear_terms(
                expr._args_[1], multiplier * expr._args_[0]
            )
        elif (
            expr._args_[1].__class__ in native_numeric_types
            or not expr._args_[1].is_potentially_variable()
        ):
            yield from _decompose_linear_terms(
                expr._args_[0], multiplier * expr._args_[1]
            )
        else:
            raise LinearDecompositionError(
                "Quadratic terms exist in a product expression."
            )
    elif expr.__class__ is DivisionExpression:
        if (
            expr._args_[1].__class__ in native_numeric_types
            or not expr._args_[1].is_potentially_variable()
        ):
            yield from _decompose_linear_terms(
                expr._args_[0], multiplier / expr._args_[1]
            )
        else:
            raise LinearDecompositionError("Unexpected nonlinear term (division)")
    elif isinstance(expr, SumExpression):
        for arg in expr.args:
            yield from _decompose_linear_terms(arg, multiplier)
    elif expr.__class__ is NegationExpression:
        yield from _decompose_linear_terms(expr._args_[0], -multiplier)
    else:
        raise LinearDecompositionError("Unexpected nonlinear term")


# -------------------------------------------------------
#
# Functions used to generate expressions
#
# -------------------------------------------------------


class ARG_TYPE(enum.IntEnum):
    MUTABLE = -2
    ASNUMERIC = -1
    INVALID = 0
    NATIVE = 1
    NPV = 2
    PARAM = 3
    VAR = 4
    MONOMIAL = 5
    LINEAR = 6
    SUM = 7
    OTHER = 8


_known_arg_types = {}


def register_arg_type(arg_class, etype):
    _known_arg_types.setdefault(arg_class, ARG_TYPE(etype))


def _categorize_arg_type(arg):
    if arg.__class__ in _known_arg_types:
        return _known_arg_types[arg.__class__]

    if arg.__class__ in native_numeric_types:
        ans = ARG_TYPE.NATIVE
    else:
        try:
            is_numeric = arg.is_numeric_type()
        except AttributeError:
            if check_if_numeric_type(arg):
                ans = ARG_TYPE.NATIVE
            else:
                ans = ARG_TYPE.INVALID
        else:
            if is_numeric:
                ans = None
            elif hasattr(arg, 'as_numeric'):
                ans = ARG_TYPE.ASNUMERIC
            else:
                ans = ARG_TYPE.INVALID

    if ans is None:
        if arg.is_expression_type():
            # Note: this makes a strong assumption that NPV is a class
            # attribute and not determined by the current expression
            # arguments / state.
            if not arg.is_potentially_variable():
                ans = ARG_TYPE.NPV
                # TODO: remove NPV_expression_types
                NPV_expression_types.add(arg.__class__)
            elif isinstance(arg, _MutableSumExpression):
                ans = ARG_TYPE.MUTABLE
            elif arg.__class__ is MonomialTermExpression:
                ans = ARG_TYPE.MONOMIAL
            elif isinstance(arg, LinearExpression):
                ans = ARG_TYPE.LINEAR
            elif isinstance(arg, SumExpression):
                ans = ARG_TYPE.SUM
            else:
                ans = ARG_TYPE.OTHER
        else:
            if not arg.is_potentially_variable():
                ans = ARG_TYPE.PARAM
            elif arg.is_variable_type():
                ans = ARG_TYPE.VAR
            else:
                ans = ARG_TYPE.OTHER
    register_arg_type(arg.__class__, ans)
    return ans


def _categorize_arg_types(*args):
    return tuple(_categorize_arg_type(arg) for arg in args)


def _invalid(*args):
    return NotImplemented


def _recast_mutable(expr):
    expr.make_immutable()
    if expr._nargs > 1:
        return expr
    elif not expr._nargs:
        return 0
    else:
        return expr._args_[0]


def _unary_op_dispatcher_type_mapping(dispatcher, updates):
    #
    # Special case (wrapping) operators
    #
    def _asnumeric(a):
        a = a.as_numeric()
        return dispatcher[a.__class__](a)

    def _mutable(a):
        a = _recast_mutable(a)
        return dispatcher[a.__class__](a)

    mapping = {
        ARG_TYPE.ASNUMERIC: _asnumeric,
        ARG_TYPE.MUTABLE: _mutable,
        ARG_TYPE.INVALID: _invalid,
    }

    mapping.update(updates)
    return mapping


def _binary_op_dispatcher_type_mapping(dispatcher, updates):
    #
    # Special case (wrapping) operators
    #
    def _any_asnumeric(a, b):
        b = b.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _asnumeric_any(a, b):
        a = a.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _asnumeric_asnumeric(a, b):
        a = a.as_numeric()
        b = b.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _any_mutable(a, b):
        b = _recast_mutable(b)
        return dispatcher[a.__class__, b.__class__](a, b)

    def _mutable_any(a, b):
        a = _recast_mutable(a)
        return dispatcher[a.__class__, b.__class__](a, b)

    def _mutable_mutable(a, b):
        if a is b:
            a = b = _recast_mutable(a)
        else:
            a = _recast_mutable(a)
            b = _recast_mutable(b)
        return dispatcher[a.__class__, b.__class__](a, b)

    mapping = {}
    mapping.update({(i, ARG_TYPE.ASNUMERIC): _any_asnumeric for i in ARG_TYPE})
    mapping.update({(ARG_TYPE.ASNUMERIC, i): _asnumeric_any for i in ARG_TYPE})
    mapping[ARG_TYPE.ASNUMERIC, ARG_TYPE.ASNUMERIC] = _asnumeric_asnumeric

    mapping.update({(i, ARG_TYPE.MUTABLE): _any_mutable for i in ARG_TYPE})
    mapping.update({(ARG_TYPE.MUTABLE, i): _mutable_any for i in ARG_TYPE})
    mapping[ARG_TYPE.MUTABLE, ARG_TYPE.MUTABLE] = _mutable_mutable

    mapping.update({(i, ARG_TYPE.INVALID): _invalid for i in ARG_TYPE})
    mapping.update({(ARG_TYPE.INVALID, i): _invalid for i in ARG_TYPE})

    mapping.update(updates)
    return mapping


#
# ADD: NATIVE handlers
#


def _add_native_native(a, b):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return a + b


def _add_native_npv(a, b):
    if not a:
        return b
    return NPV_SumExpression([a, b])


def _add_native_param(a, b):
    if b.is_constant():
        return a + b.value
    if not a:
        return b
    return NPV_SumExpression([a, b])


def _add_native_var(a, b):
    if not a:
        return b
    return LinearExpression([a, MonomialTermExpression((1, b))])


def _add_native_monomial(a, b):
    if not a:
        return b
    return LinearExpression([a, b])


def _add_native_linear(a, b):
    if not a:
        return b
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_native_sum(a, b):
    if not a:
        return b
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_native_other(a, b):
    if not a:
        return b
    return SumExpression([a, b])


#
# ADD: NPV handlers
#


def _add_npv_native(a, b):
    if not b:
        return a
    return NPV_SumExpression([a, b])


def _add_npv_npv(a, b):
    return NPV_SumExpression([a, b])


def _add_npv_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    return NPV_SumExpression([a, b])


def _add_npv_var(a, b):
    return LinearExpression([a, MonomialTermExpression((1, b))])


def _add_npv_monomial(a, b):
    return LinearExpression([a, b])


def _add_npv_linear(a, b):
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_npv_sum(a, b):
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_npv_other(a, b):
    return SumExpression([a, b])


#
# ADD: PARAM handlers
#


def _add_param_native(a, b):
    if a.is_constant():
        return a.value + b
    if not b:
        return a
    return NPV_SumExpression([a, b])


def _add_param_npv(a, b):
    if a.is_constant():
        a = a.value
        return a + b
    return NPV_SumExpression([a, b])


def _add_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a + b.value
        elif not a:
            return b
    elif b.is_constant():
        b = b.value
        if not b:
            return a
    return NPV_SumExpression([a, b])


def _add_param_var(a, b):
    if a.is_constant():
        a = a.value
        if not a:
            return b
    return LinearExpression([a, MonomialTermExpression((1, b))])


def _add_param_monomial(a, b):
    if a.is_constant():
        a = a.value
        if not a:
            return b
    return LinearExpression([a, b])


def _add_param_linear(a, b):
    if a.is_constant():
        a = a.value
        if not a:
            return b
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_param_sum(a, b):
    if a.is_constant():
        a = value(a)
        if not a:
            return b
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_param_other(a, b):
    if a.is_constant():
        a = a.value
        if not a:
            return b
    return SumExpression([a, b])


#
# ADD: VAR handlers
#


def _add_var_native(a, b):
    if not b:
        return a
    return LinearExpression([MonomialTermExpression((1, a)), b])


def _add_var_npv(a, b):
    return LinearExpression([MonomialTermExpression((1, a)), b])


def _add_var_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    return LinearExpression([MonomialTermExpression((1, a)), b])


def _add_var_var(a, b):
    return LinearExpression(
        [MonomialTermExpression((1, a)), MonomialTermExpression((1, b))]
    )


def _add_var_monomial(a, b):
    return LinearExpression([MonomialTermExpression((1, a)), b])


def _add_var_linear(a, b):
    args = b.args
    args.append(MonomialTermExpression((1, a)))
    return b.__class__(args)


def _add_var_sum(a, b):
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_var_other(a, b):
    return SumExpression([a, b])


#
# ADD: MONOMIAL handlers
#


def _add_monomial_native(a, b):
    if not b:
        return a
    return LinearExpression([a, b])


def _add_monomial_npv(a, b):
    return LinearExpression([a, b])


def _add_monomial_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    return LinearExpression([a, b])


def _add_monomial_var(a, b):
    return LinearExpression([a, MonomialTermExpression((1, b))])


def _add_monomial_monomial(a, b):
    return LinearExpression([a, b])


def _add_monomial_linear(a, b):
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_monomial_sum(a, b):
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_monomial_other(a, b):
    return SumExpression([a, b])


#
# ADD: LINEAR handlers
#


def _add_linear_native(a, b):
    if not b:
        return a
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_linear_npv(a, b):
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_linear_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_linear_var(a, b):
    args = a.args
    args.append(MonomialTermExpression((1, b)))
    return a.__class__(args)


def _add_linear_monomial(a, b):
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_linear_linear(a, b):
    args = a.args
    args.extend(b.args)
    return a.__class__(args)


def _add_linear_sum(a, b):
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_linear_other(a, b):
    return SumExpression([a, b])


#
# ADD: SUM handlers
#


def _add_sum_native(a, b):
    if not b:
        return a
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_sum_npv(a, b):
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_sum_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_sum_var(a, b):
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_sum_monomial(a, b):
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_sum_linear(a, b):
    args = a.args
    args.append(b)
    return a.__class__(args)


def _add_sum_sum(a, b):
    args = a.args
    args.extend(b.args)
    return a.__class__(args)


def _add_sum_other(a, b):
    args = a.args
    args.append(b)
    return a.__class__(args)


#
# ADD: OTHER handlers
#


def _add_other_native(a, b):
    if not b:
        return a
    return SumExpression([a, b])


def _add_other_npv(a, b):
    return SumExpression([a, b])


def _add_other_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    return SumExpression([a, b])


def _add_other_var(a, b):
    return SumExpression([a, b])


def _add_other_monomial(a, b):
    return SumExpression([a, b])


def _add_other_linear(a, b):
    return SumExpression([a, b])


def _add_other_sum(a, b):
    args = b.args
    args.append(a)
    return b.__class__(args)


def _add_other_other(a, b):
    return SumExpression([a, b])


def _register_new_add_handler(a, b):
    types = _categorize_arg_types(a, b)
    # Retrieve the appropriate handler, record it in the main
    # _add_dispatcher dict (so this method is not called a second time for
    # these types)
    _add_dispatcher[a.__class__, b.__class__] = handler = _add_type_handler_mapping[
        types
    ]
    # Call the appropriate handler
    return handler(a, b)


_add_dispatcher = collections.defaultdict(lambda: _register_new_add_handler)

_add_type_handler_mapping = _binary_op_dispatcher_type_mapping(
    _add_dispatcher,
    {
        (ARG_TYPE.NATIVE, ARG_TYPE.NATIVE): _add_native_native,
        (ARG_TYPE.NATIVE, ARG_TYPE.NPV): _add_native_npv,
        (ARG_TYPE.NATIVE, ARG_TYPE.PARAM): _add_native_param,
        (ARG_TYPE.NATIVE, ARG_TYPE.VAR): _add_native_var,
        (ARG_TYPE.NATIVE, ARG_TYPE.MONOMIAL): _add_native_monomial,
        (ARG_TYPE.NATIVE, ARG_TYPE.LINEAR): _add_native_linear,
        (ARG_TYPE.NATIVE, ARG_TYPE.SUM): _add_native_sum,
        (ARG_TYPE.NATIVE, ARG_TYPE.OTHER): _add_native_other,
        (ARG_TYPE.NPV, ARG_TYPE.NATIVE): _add_npv_native,
        (ARG_TYPE.NPV, ARG_TYPE.NPV): _add_npv_npv,
        (ARG_TYPE.NPV, ARG_TYPE.PARAM): _add_npv_param,
        (ARG_TYPE.NPV, ARG_TYPE.VAR): _add_npv_var,
        (ARG_TYPE.NPV, ARG_TYPE.MONOMIAL): _add_npv_monomial,
        (ARG_TYPE.NPV, ARG_TYPE.LINEAR): _add_npv_linear,
        (ARG_TYPE.NPV, ARG_TYPE.SUM): _add_npv_sum,
        (ARG_TYPE.NPV, ARG_TYPE.OTHER): _add_npv_other,
        (ARG_TYPE.PARAM, ARG_TYPE.NATIVE): _add_param_native,
        (ARG_TYPE.PARAM, ARG_TYPE.NPV): _add_param_npv,
        (ARG_TYPE.PARAM, ARG_TYPE.PARAM): _add_param_param,
        (ARG_TYPE.PARAM, ARG_TYPE.VAR): _add_param_var,
        (ARG_TYPE.PARAM, ARG_TYPE.MONOMIAL): _add_param_monomial,
        (ARG_TYPE.PARAM, ARG_TYPE.LINEAR): _add_param_linear,
        (ARG_TYPE.PARAM, ARG_TYPE.SUM): _add_param_sum,
        (ARG_TYPE.PARAM, ARG_TYPE.OTHER): _add_param_other,
        (ARG_TYPE.VAR, ARG_TYPE.NATIVE): _add_var_native,
        (ARG_TYPE.VAR, ARG_TYPE.NPV): _add_var_npv,
        (ARG_TYPE.VAR, ARG_TYPE.PARAM): _add_var_param,
        (ARG_TYPE.VAR, ARG_TYPE.VAR): _add_var_var,
        (ARG_TYPE.VAR, ARG_TYPE.MONOMIAL): _add_var_monomial,
        (ARG_TYPE.VAR, ARG_TYPE.LINEAR): _add_var_linear,
        (ARG_TYPE.VAR, ARG_TYPE.SUM): _add_var_sum,
        (ARG_TYPE.VAR, ARG_TYPE.OTHER): _add_var_other,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NATIVE): _add_monomial_native,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NPV): _add_monomial_npv,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.PARAM): _add_monomial_param,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.VAR): _add_monomial_var,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.MONOMIAL): _add_monomial_monomial,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.LINEAR): _add_monomial_linear,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.SUM): _add_monomial_sum,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.OTHER): _add_monomial_other,
        (ARG_TYPE.LINEAR, ARG_TYPE.NATIVE): _add_linear_native,
        (ARG_TYPE.LINEAR, ARG_TYPE.NPV): _add_linear_npv,
        (ARG_TYPE.LINEAR, ARG_TYPE.PARAM): _add_linear_param,
        (ARG_TYPE.LINEAR, ARG_TYPE.VAR): _add_linear_var,
        (ARG_TYPE.LINEAR, ARG_TYPE.MONOMIAL): _add_linear_monomial,
        (ARG_TYPE.LINEAR, ARG_TYPE.LINEAR): _add_linear_linear,
        (ARG_TYPE.LINEAR, ARG_TYPE.SUM): _add_linear_sum,
        (ARG_TYPE.LINEAR, ARG_TYPE.OTHER): _add_linear_other,
        (ARG_TYPE.SUM, ARG_TYPE.NATIVE): _add_sum_native,
        (ARG_TYPE.SUM, ARG_TYPE.NPV): _add_sum_npv,
        (ARG_TYPE.SUM, ARG_TYPE.PARAM): _add_sum_param,
        (ARG_TYPE.SUM, ARG_TYPE.VAR): _add_sum_var,
        (ARG_TYPE.SUM, ARG_TYPE.MONOMIAL): _add_sum_monomial,
        (ARG_TYPE.SUM, ARG_TYPE.LINEAR): _add_sum_linear,
        (ARG_TYPE.SUM, ARG_TYPE.SUM): _add_sum_sum,
        (ARG_TYPE.SUM, ARG_TYPE.OTHER): _add_sum_other,
        (ARG_TYPE.OTHER, ARG_TYPE.NATIVE): _add_other_native,
        (ARG_TYPE.OTHER, ARG_TYPE.NPV): _add_other_npv,
        (ARG_TYPE.OTHER, ARG_TYPE.PARAM): _add_other_param,
        (ARG_TYPE.OTHER, ARG_TYPE.VAR): _add_other_var,
        (ARG_TYPE.OTHER, ARG_TYPE.MONOMIAL): _add_other_monomial,
        (ARG_TYPE.OTHER, ARG_TYPE.LINEAR): _add_other_linear,
        (ARG_TYPE.OTHER, ARG_TYPE.SUM): _add_other_sum,
        (ARG_TYPE.OTHER, ARG_TYPE.OTHER): _add_other_other,
    },
)

#
# MUTABLENPVSUM __iadd__ handlers
#


def _iadd_mutablenpvsum_asnumeric(a, b):
    b = b.as_numeric()
    return _iadd_mutablenpvsum_dispatcher[b.__class__](a, b)


def _iadd_mutablenpvsum_mutable(a, b):
    b = _recast_mutable(b)
    return _iadd_mutablenpvsum_dispatcher[b.__class__](a, b)


def _iadd_mutablenpvsum_native(a, b):
    if not b:
        return a
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablenpvsum_npv(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablenpvsum_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablenpvsum_var(a, b):
    a.__class__ = _MutableLinearExpression
    return _iadd_mutablelinear_var(a, b)


def _iadd_mutablenpvsum_monomial(a, b):
    a.__class__ = _MutableLinearExpression
    return _iadd_mutablelinear_monomial(a, b)


def _iadd_mutablenpvsum_linear(a, b):
    a.__class__ = _MutableLinearExpression
    return _iadd_mutablelinear_linear(a, b)


def _iadd_mutablenpvsum_sum(a, b):
    a.__class__ = _MutableSumExpression
    return _iadd_mutablesum_sum(a, b)


def _iadd_mutablenpvsum_other(a, b):
    a.__class__ = _MutableSumExpression
    return _iadd_mutablesum_other(a, b)


_iadd_mutablenpvsum_type_handler_mapping = {
    ARG_TYPE.INVALID: _invalid,
    ARG_TYPE.ASNUMERIC: _iadd_mutablenpvsum_asnumeric,
    ARG_TYPE.MUTABLE: _iadd_mutablenpvsum_mutable,
    ARG_TYPE.NATIVE: _iadd_mutablenpvsum_native,
    ARG_TYPE.NPV: _iadd_mutablenpvsum_npv,
    ARG_TYPE.PARAM: _iadd_mutablenpvsum_param,
    ARG_TYPE.VAR: _iadd_mutablenpvsum_var,
    ARG_TYPE.MONOMIAL: _iadd_mutablenpvsum_monomial,
    ARG_TYPE.LINEAR: _iadd_mutablenpvsum_linear,
    ARG_TYPE.SUM: _iadd_mutablenpvsum_sum,
    ARG_TYPE.OTHER: _iadd_mutablenpvsum_other,
}


def _register_new_iadd_mutablenpvsum_handler(a, b):
    types = _categorize_arg_types(b)
    # Retrieve the appropriate handler, record it in the main
    # _iadd_mutablenpvsum_dispatcher dict (so this method is not called a second time for
    # these types)
    _iadd_mutablenpvsum_dispatcher[
        b.__class__
    ] = handler = _iadd_mutablenpvsum_type_handler_mapping[types[0]]
    # Call the appropriate handler
    return handler(a, b)


_iadd_mutablenpvsum_dispatcher = collections.defaultdict(
    lambda: _register_new_iadd_mutablenpvsum_handler
)


#
# MUTABLELINEAR __iadd__ handlers
#


def _iadd_mutablelinear_asnumeric(a, b):
    b = b.as_numeric()
    return _iadd_mutablelinear_dispatcher[b.__class__](a, b)


def _iadd_mutablelinear_mutable(a, b):
    b = _recast_mutable(b)
    return _iadd_mutablelinear_dispatcher[b.__class__](a, b)


def _iadd_mutablelinear_native(a, b):
    if not b:
        return a
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablelinear_npv(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablelinear_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablelinear_var(a, b):
    a._args_.append(MonomialTermExpression((1, b)))
    a._nargs += 1
    return a


def _iadd_mutablelinear_monomial(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablelinear_linear(a, b):
    a._args_.extend(b.args)
    a._nargs += b.nargs()
    return a


def _iadd_mutablelinear_sum(a, b):
    a.__class__ = _MutableSumExpression
    return _iadd_mutablesum_sum(a, b)


def _iadd_mutablelinear_other(a, b):
    a.__class__ = _MutableSumExpression
    return _iadd_mutablesum_other(a, b)


_iadd_mutablelinear_type_handler_mapping = {
    ARG_TYPE.INVALID: _invalid,
    ARG_TYPE.ASNUMERIC: _iadd_mutablelinear_asnumeric,
    ARG_TYPE.MUTABLE: _iadd_mutablelinear_mutable,
    ARG_TYPE.NATIVE: _iadd_mutablelinear_native,
    ARG_TYPE.NPV: _iadd_mutablelinear_npv,
    ARG_TYPE.PARAM: _iadd_mutablelinear_param,
    ARG_TYPE.VAR: _iadd_mutablelinear_var,
    ARG_TYPE.MONOMIAL: _iadd_mutablelinear_monomial,
    ARG_TYPE.LINEAR: _iadd_mutablelinear_linear,
    ARG_TYPE.SUM: _iadd_mutablelinear_sum,
    ARG_TYPE.OTHER: _iadd_mutablelinear_other,
}


def _register_new_iadd_mutablelinear_handler(a, b):
    types = _categorize_arg_types(b)
    # Retrieve the appropriate handler, record it in the main
    # _iadd_mutablelinear_dispatcher dict (so this method is not called a second time for
    # these types)
    _iadd_mutablelinear_dispatcher[
        b.__class__
    ] = handler = _iadd_mutablelinear_type_handler_mapping[types[0]]
    # Call the appropriate handler
    return handler(a, b)


_iadd_mutablelinear_dispatcher = collections.defaultdict(
    lambda: _register_new_iadd_mutablelinear_handler
)


#
# MUTABLESUM __iadd__ handlers
#


def _iadd_mutablesum_asnumeric(a, b):
    b = b.as_numeric()
    return _iadd_mutablesum_dispatcher[b.__class__](a, b)


def _iadd_mutablesum_mutable(a, b):
    b = _recast_mutable(b)
    return _iadd_mutablesum_dispatcher[b.__class__](a, b)


def _iadd_mutablesum_native(a, b):
    if not b:
        return a
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablesum_npv(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablesum_param(a, b):
    if b.is_constant():
        b = b.value
        if not b:
            return a
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablesum_var(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablesum_monomial(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablesum_linear(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


def _iadd_mutablesum_sum(a, b):
    a._args_.extend(b.args)
    a._nargs += b.nargs()
    return a


def _iadd_mutablesum_other(a, b):
    a._args_.append(b)
    a._nargs += 1
    return a


_iadd_mutablesum_type_handler_mapping = {
    ARG_TYPE.INVALID: _invalid,
    ARG_TYPE.ASNUMERIC: _iadd_mutablesum_asnumeric,
    ARG_TYPE.MUTABLE: _iadd_mutablesum_mutable,
    ARG_TYPE.NATIVE: _iadd_mutablesum_native,
    ARG_TYPE.NPV: _iadd_mutablesum_npv,
    ARG_TYPE.PARAM: _iadd_mutablesum_param,
    ARG_TYPE.VAR: _iadd_mutablesum_var,
    ARG_TYPE.MONOMIAL: _iadd_mutablesum_monomial,
    ARG_TYPE.LINEAR: _iadd_mutablesum_linear,
    ARG_TYPE.SUM: _iadd_mutablesum_sum,
    ARG_TYPE.OTHER: _iadd_mutablesum_other,
}


def _register_new_iadd_mutablesum_handler(a, b):
    types = _categorize_arg_types(b)
    # Retrieve the appropriate handler, record it in the main
    # _iadd_mutablesum_dispatcher dict (so this method is not called a
    # second time for these types)
    _iadd_mutablesum_dispatcher[
        b.__class__
    ] = handler = _iadd_mutablesum_type_handler_mapping[types[0]]
    # Call the appropriate handler
    return handler(a, b)


_iadd_mutablesum_dispatcher = collections.defaultdict(
    lambda: _register_new_iadd_mutablesum_handler
)


#
# NEGATION handlers
#


def _neg_native(a):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return -a


def _neg_npv(a):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return NPV_NegationExpression((a,))


def _neg_param(a):
    if a.is_constant():
        return -(a.value)
    return NPV_NegationExpression((a,))


def _neg_var(a):
    return MonomialTermExpression((-1, a))


def _neg_monomial(a):
    args = a.args
    return MonomialTermExpression((-args[0], args[1]))


def _neg_sum(a):
    if not a.nargs():
        return 0
    # return LinearExpression([-arg for arg in a.args])
    return NegationExpression((a,))


def _neg_other(a):
    return NegationExpression((a,))


def _register_new_neg_handler(a):
    types = _categorize_arg_types(a)
    # Retrieve the appropriate handler, record it in the main
    # _neg_dispatcher dict (so this method is not called a second time for
    # these types)
    _neg_dispatcher[a.__class__] = handler = _neg_type_handler_mapping[types[0]]
    # Call the appropriate handler
    return handler(a)


_neg_dispatcher = collections.defaultdict(lambda: _register_new_neg_handler)

_neg_type_handler_mapping = _unary_op_dispatcher_type_mapping(
    _neg_dispatcher,
    {
        ARG_TYPE.NATIVE: _neg_native,
        ARG_TYPE.NPV: _neg_npv,
        ARG_TYPE.PARAM: _neg_param,
        ARG_TYPE.VAR: _neg_var,
        ARG_TYPE.MONOMIAL: _neg_monomial,
        ARG_TYPE.LINEAR: _neg_sum,
        ARG_TYPE.SUM: _neg_sum,
        ARG_TYPE.OTHER: _neg_other,
    },
)


#
# MUL: NATIVE handlers
#


def _mul_native_native(a, b):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return a * b


def _mul_native_npv(a, b):
    if a in _zero_one_optimizations:
        return b if a else 0
    return NPV_ProductExpression((a, b))


def _mul_native_param(a, b):
    if a in _zero_one_optimizations:
        return b if a else 0
    if b.is_constant():
        return a * b.value
    return NPV_ProductExpression((a, b))


def _mul_native_var(a, b):
    if a in _zero_one_optimizations:
        return b if a else 0
    return MonomialTermExpression((a, b))


def _mul_native_monomial(a, b):
    if a in _zero_one_optimizations:
        return b if a else 0
    return MonomialTermExpression((a * b._args_[0], b._args_[1]))


def _mul_native_linear(a, b):
    if a in _zero_one_optimizations:
        return b if a else 0
    return ProductExpression((a, b))


def _mul_native_sum(a, b):
    if a in _zero_one_optimizations:
        return b if a else 0
    return ProductExpression((a, b))


def _mul_native_other(a, b):
    if a in _zero_one_optimizations:
        return b if a else 0
    return ProductExpression((a, b))


#
# MUL: NPV handlers
#


def _mul_npv_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 0
    return NPV_ProductExpression((a, b))


def _mul_npv_npv(a, b):
    return NPV_ProductExpression((a, b))


def _mul_npv_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return NPV_ProductExpression((a, b))


def _mul_npv_var(a, b):
    return MonomialTermExpression((a, b))


def _mul_npv_monomial(a, b):
    return MonomialTermExpression(
        (NPV_ProductExpression((a, b._args_[0])), b._args_[1])
    )


def _mul_npv_linear(a, b):
    return ProductExpression((a, b))


def _mul_npv_sum(a, b):
    return ProductExpression((a, b))


def _mul_npv_other(a, b):
    return ProductExpression((a, b))


#
# MUL: PARAM handlers
#


def _mul_param_native(a, b):
    if a.is_constant():
        return a.value * b
    if b in _zero_one_optimizations:
        return a if b else 0
    return NPV_ProductExpression((a, b))


def _mul_param_npv(a, b):
    if a.is_constant():
        a = a.value
        if a in _zero_one_optimizations:
            return b if a else 0
    return NPV_ProductExpression((a, b))


def _mul_param_param(a, b):
    if a.is_constant():
        a = a.value
        if a in _zero_one_optimizations:
            return b if a else 0
        if b.is_constant():
            return a * b.value
    elif b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return NPV_ProductExpression((a, b))


def _mul_param_var(a, b):
    if a.is_constant():
        a = a.value
        if a in _zero_one_optimizations:
            return b if a else 0
    return MonomialTermExpression((a, b))


def _mul_param_monomial(a, b):
    if a.is_constant():
        a = a.value
        if a in _zero_one_optimizations:
            return b if a else 0
    return MonomialTermExpression((a * b._args_[0], b._args_[1]))


def _mul_param_linear(a, b):
    if a.is_constant():
        a = a.value
        if a in _zero_one_optimizations:
            return b if a else 0
    return ProductExpression((a, b))


def _mul_param_sum(a, b):
    if a.is_constant():
        a = value(a)
        if a in _zero_one_optimizations:
            return b if a else 0
    return ProductExpression((a, b))


def _mul_param_other(a, b):
    if a.is_constant():
        a = a.value
        if a in _zero_one_optimizations:
            return b if a else 0
    return ProductExpression((a, b))


#
# MUL: VAR handlers
#


def _mul_var_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 0
    return MonomialTermExpression((b, a))


def _mul_var_npv(a, b):
    return MonomialTermExpression((b, a))


def _mul_var_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return MonomialTermExpression((b, a))


def _mul_var_var(a, b):
    return ProductExpression((a, b))


def _mul_var_monomial(a, b):
    return ProductExpression((a, b))


def _mul_var_linear(a, b):
    return ProductExpression((a, b))


def _mul_var_sum(a, b):
    return ProductExpression((a, b))


def _mul_var_other(a, b):
    return ProductExpression((a, b))


#
# MUL: MONOMIAL handlers
#


def _mul_monomial_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 0
    return MonomialTermExpression((a._args_[0] * b, a._args_[1]))


def _mul_monomial_npv(a, b):
    return MonomialTermExpression(
        (NPV_ProductExpression((a._args_[0], b)), a._args_[1])
    )


def _mul_monomial_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return MonomialTermExpression((a._args_[0] * b, a._args_[1]))


def _mul_monomial_var(a, b):
    return ProductExpression((a, b))


def _mul_monomial_monomial(a, b):
    return ProductExpression((a, b))


def _mul_monomial_linear(a, b):
    return ProductExpression((a, b))


def _mul_monomial_sum(a, b):
    return ProductExpression((a, b))


def _mul_monomial_other(a, b):
    return ProductExpression((a, b))


#
# MUL: LINEAR handlers
#


def _mul_linear_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 0
    return ProductExpression((a, b))


def _mul_linear_npv(a, b):
    return ProductExpression((a, b))


def _mul_linear_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return ProductExpression((a, b))


def _mul_linear_var(a, b):
    return ProductExpression((a, b))


def _mul_linear_monomial(a, b):
    return ProductExpression((a, b))


def _mul_linear_linear(a, b):
    return ProductExpression((a, b))


def _mul_linear_sum(a, b):
    return ProductExpression((a, b))


def _mul_linear_other(a, b):
    return ProductExpression((a, b))


#
# MUL: SUM handlers
#


def _mul_sum_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 0
    return ProductExpression((a, b))


def _mul_sum_npv(a, b):
    return ProductExpression((a, b))


def _mul_sum_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return ProductExpression((a, b))


def _mul_sum_var(a, b):
    return ProductExpression((a, b))


def _mul_sum_monomial(a, b):
    return ProductExpression((a, b))


def _mul_sum_linear(a, b):
    return ProductExpression((a, b))


def _mul_sum_sum(a, b):
    return ProductExpression((a, b))


def _mul_sum_other(a, b):
    return ProductExpression((a, b))


#
# MUL: OTHER handlers
#


def _mul_other_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 0
    return ProductExpression((a, b))


def _mul_other_npv(a, b):
    return ProductExpression((a, b))


def _mul_other_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return ProductExpression((a, b))


def _mul_other_var(a, b):
    return ProductExpression((a, b))


def _mul_other_monomial(a, b):
    return ProductExpression((a, b))


def _mul_other_linear(a, b):
    return ProductExpression((a, b))


def _mul_other_sum(a, b):
    return ProductExpression((a, b))


def _mul_other_other(a, b):
    return ProductExpression((a, b))


def _register_new_mul_handler(a, b):
    types = _categorize_arg_types(a, b)
    # Retrieve the appropriate handler, record it in the main
    # _mul_dispatcher dict (so this method is not called a second time for
    # these types)
    _mul_dispatcher[a.__class__, b.__class__] = handler = _mul_type_handler_mapping[
        types
    ]
    # Call the appropriate handler
    return handler(a, b)


_mul_dispatcher = collections.defaultdict(lambda: _register_new_mul_handler)

_mul_type_handler_mapping = _binary_op_dispatcher_type_mapping(
    _mul_dispatcher,
    {
        (ARG_TYPE.NATIVE, ARG_TYPE.NATIVE): _mul_native_native,
        (ARG_TYPE.NATIVE, ARG_TYPE.NPV): _mul_native_npv,
        (ARG_TYPE.NATIVE, ARG_TYPE.PARAM): _mul_native_param,
        (ARG_TYPE.NATIVE, ARG_TYPE.VAR): _mul_native_var,
        (ARG_TYPE.NATIVE, ARG_TYPE.MONOMIAL): _mul_native_monomial,
        (ARG_TYPE.NATIVE, ARG_TYPE.LINEAR): _mul_native_linear,
        (ARG_TYPE.NATIVE, ARG_TYPE.SUM): _mul_native_sum,
        (ARG_TYPE.NATIVE, ARG_TYPE.OTHER): _mul_native_other,
        (ARG_TYPE.NPV, ARG_TYPE.NATIVE): _mul_npv_native,
        (ARG_TYPE.NPV, ARG_TYPE.NPV): _mul_npv_npv,
        (ARG_TYPE.NPV, ARG_TYPE.PARAM): _mul_npv_param,
        (ARG_TYPE.NPV, ARG_TYPE.VAR): _mul_npv_var,
        (ARG_TYPE.NPV, ARG_TYPE.MONOMIAL): _mul_npv_monomial,
        (ARG_TYPE.NPV, ARG_TYPE.LINEAR): _mul_npv_linear,
        (ARG_TYPE.NPV, ARG_TYPE.SUM): _mul_npv_sum,
        (ARG_TYPE.NPV, ARG_TYPE.OTHER): _mul_npv_other,
        (ARG_TYPE.PARAM, ARG_TYPE.NATIVE): _mul_param_native,
        (ARG_TYPE.PARAM, ARG_TYPE.NPV): _mul_param_npv,
        (ARG_TYPE.PARAM, ARG_TYPE.PARAM): _mul_param_param,
        (ARG_TYPE.PARAM, ARG_TYPE.VAR): _mul_param_var,
        (ARG_TYPE.PARAM, ARG_TYPE.MONOMIAL): _mul_param_monomial,
        (ARG_TYPE.PARAM, ARG_TYPE.LINEAR): _mul_param_linear,
        (ARG_TYPE.PARAM, ARG_TYPE.SUM): _mul_param_sum,
        (ARG_TYPE.PARAM, ARG_TYPE.OTHER): _mul_param_other,
        (ARG_TYPE.VAR, ARG_TYPE.NATIVE): _mul_var_native,
        (ARG_TYPE.VAR, ARG_TYPE.NPV): _mul_var_npv,
        (ARG_TYPE.VAR, ARG_TYPE.PARAM): _mul_var_param,
        (ARG_TYPE.VAR, ARG_TYPE.VAR): _mul_var_var,
        (ARG_TYPE.VAR, ARG_TYPE.MONOMIAL): _mul_var_monomial,
        (ARG_TYPE.VAR, ARG_TYPE.LINEAR): _mul_var_linear,
        (ARG_TYPE.VAR, ARG_TYPE.SUM): _mul_var_sum,
        (ARG_TYPE.VAR, ARG_TYPE.OTHER): _mul_var_other,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NATIVE): _mul_monomial_native,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NPV): _mul_monomial_npv,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.PARAM): _mul_monomial_param,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.VAR): _mul_monomial_var,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.MONOMIAL): _mul_monomial_monomial,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.LINEAR): _mul_monomial_linear,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.SUM): _mul_monomial_sum,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.OTHER): _mul_monomial_other,
        (ARG_TYPE.LINEAR, ARG_TYPE.NATIVE): _mul_linear_native,
        (ARG_TYPE.LINEAR, ARG_TYPE.NPV): _mul_linear_npv,
        (ARG_TYPE.LINEAR, ARG_TYPE.PARAM): _mul_linear_param,
        (ARG_TYPE.LINEAR, ARG_TYPE.VAR): _mul_linear_var,
        (ARG_TYPE.LINEAR, ARG_TYPE.MONOMIAL): _mul_linear_monomial,
        (ARG_TYPE.LINEAR, ARG_TYPE.LINEAR): _mul_linear_linear,
        (ARG_TYPE.LINEAR, ARG_TYPE.SUM): _mul_linear_sum,
        (ARG_TYPE.LINEAR, ARG_TYPE.OTHER): _mul_linear_other,
        (ARG_TYPE.SUM, ARG_TYPE.NATIVE): _mul_sum_native,
        (ARG_TYPE.SUM, ARG_TYPE.NPV): _mul_sum_npv,
        (ARG_TYPE.SUM, ARG_TYPE.PARAM): _mul_sum_param,
        (ARG_TYPE.SUM, ARG_TYPE.VAR): _mul_sum_var,
        (ARG_TYPE.SUM, ARG_TYPE.MONOMIAL): _mul_sum_monomial,
        (ARG_TYPE.SUM, ARG_TYPE.LINEAR): _mul_sum_linear,
        (ARG_TYPE.SUM, ARG_TYPE.SUM): _mul_sum_sum,
        (ARG_TYPE.SUM, ARG_TYPE.OTHER): _mul_sum_other,
        (ARG_TYPE.OTHER, ARG_TYPE.NATIVE): _mul_other_native,
        (ARG_TYPE.OTHER, ARG_TYPE.NPV): _mul_other_npv,
        (ARG_TYPE.OTHER, ARG_TYPE.PARAM): _mul_other_param,
        (ARG_TYPE.OTHER, ARG_TYPE.VAR): _mul_other_var,
        (ARG_TYPE.OTHER, ARG_TYPE.MONOMIAL): _mul_other_monomial,
        (ARG_TYPE.OTHER, ARG_TYPE.LINEAR): _mul_other_linear,
        (ARG_TYPE.OTHER, ARG_TYPE.SUM): _mul_other_sum,
        (ARG_TYPE.OTHER, ARG_TYPE.OTHER): _mul_other_other,
    },
)


#
# DIV: NATIVE handlers
#


def _div_native_native(a, b):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return a / b


def _div_native_npv(a, b):
    if not a and a in _zero_one_optimizations:
        return 0
    return NPV_DivisionExpression((a, b))


def _div_native_param(a, b):
    if b.is_constant():
        return a / b.value
    if not a and a in _zero_one_optimizations:
        return 0
    return NPV_DivisionExpression((a, b))


def _div_native_var(a, b):
    if not a and a in _zero_one_optimizations:
        return 0
    return DivisionExpression((a, b))


def _div_native_monomial(a, b):
    if not a and a in _zero_one_optimizations:
        return 0
    return DivisionExpression((a, b))


def _div_native_linear(a, b):
    if not a and a in _zero_one_optimizations:
        return 0
    return DivisionExpression((a, b))


def _div_native_sum(a, b):
    if not a and a in _zero_one_optimizations:
        return 0
    return DivisionExpression((a, b))


def _div_native_other(a, b):
    if not a and a in _zero_one_optimizations:
        return 0
    return DivisionExpression((a, b))


#
# DIV: NPV handlers
#


def _div_npv_native(a, b):
    if b in _zero_one_optimizations and b:
        return a
    if not b:
        raise ZeroDivisionError()
    return NPV_DivisionExpression((a, b))


def _div_npv_npv(a, b):
    return NPV_DivisionExpression((a, b))


def _div_npv_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        if not b:
            raise ZeroDivisionError()
    return NPV_DivisionExpression((a, b))


def _div_npv_var(a, b):
    return DivisionExpression((a, b))


def _div_npv_monomial(a, b):
    return DivisionExpression((a, b))


def _div_npv_linear(a, b):
    return DivisionExpression((a, b))


def _div_npv_sum(a, b):
    return DivisionExpression((a, b))


def _div_npv_other(a, b):
    return DivisionExpression((a, b))


#
# DIV: PARAM handlers
#


def _div_param_native(a, b):
    if a.is_constant():
        return a.value / b
    if b in _zero_one_optimizations and b:
        return a
    if not b:
        raise ZeroDivisionError()
    return NPV_DivisionExpression((a, b))


def _div_param_npv(a, b):
    if a.is_constant():
        a = a.value
        if not a and a in _zero_one_optimizations:
            return 0
    return NPV_DivisionExpression((a, b))


def _div_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a / b.value
        if not a and a in _zero_one_optimizations:
            return 0
    elif b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        if not b:
            raise ZeroDivisionError()
    return NPV_DivisionExpression((a, b))


def _div_param_var(a, b):
    if a.is_constant():
        a = a.value
        if not a and a in _zero_one_optimizations:
            return 0
    return DivisionExpression((a, b))


def _div_param_monomial(a, b):
    if a.is_constant():
        a = a.value
        if not a and a in _zero_one_optimizations:
            return 0
    return DivisionExpression((a, b))


def _div_param_linear(a, b):
    if a.is_constant():
        a = a.value
        if not a and a in _zero_one_optimizations:
            return 0
    return DivisionExpression((a, b))


def _div_param_sum(a, b):
    if a.is_constant():
        a = value(a)
        if not a and a in _zero_one_optimizations:
            return 0
    return DivisionExpression((a, b))


def _div_param_other(a, b):
    if a.is_constant():
        a = a.value
        if not a and a in _zero_one_optimizations:
            return 0
    return DivisionExpression((a, b))


#
# DIV: VAR handlers
#


def _div_var_native(a, b):
    if b in _zero_one_optimizations and b:
        return a
    return MonomialTermExpression((1 / b, a))


def _div_var_npv(a, b):
    return MonomialTermExpression((NPV_DivisionExpression((1, b)), a))


def _div_var_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        return MonomialTermExpression((1 / b, a))
    return MonomialTermExpression((NPV_DivisionExpression((1, b)), a))


def _div_var_var(a, b):
    return DivisionExpression((a, b))


def _div_var_monomial(a, b):
    return DivisionExpression((a, b))


def _div_var_linear(a, b):
    return DivisionExpression((a, b))


def _div_var_sum(a, b):
    return DivisionExpression((a, b))


def _div_var_other(a, b):
    return DivisionExpression((a, b))


#
# DIV: MONOMIAL handlers
#


def _div_monomial_native(a, b):
    if b in _zero_one_optimizations and b:
        return a
    return MonomialTermExpression((a._args_[0] / b, a._args_[1]))


def _div_monomial_npv(a, b):
    return MonomialTermExpression(
        (NPV_DivisionExpression((a._args_[0], b)), a._args_[1])
    )


def _div_monomial_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        return MonomialTermExpression((a._args_[0] / b, a._args_[1]))
    return MonomialTermExpression(
        (NPV_DivisionExpression((a._args_[0], b)), a._args_[1])
    )


def _div_monomial_var(a, b):
    return DivisionExpression((a, b))


def _div_monomial_monomial(a, b):
    return DivisionExpression((a, b))


def _div_monomial_linear(a, b):
    return DivisionExpression((a, b))


def _div_monomial_sum(a, b):
    return DivisionExpression((a, b))


def _div_monomial_other(a, b):
    return DivisionExpression((a, b))


#
# DIV: LINEAR handlers
#


def _div_linear_native(a, b):
    if b in _zero_one_optimizations and b:
        return a
    if not b:
        raise ZeroDivisionError()
    return DivisionExpression((a, b))


def _div_linear_npv(a, b):
    return DivisionExpression((a, b))


def _div_linear_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        if not b:
            raise ZeroDivisionError()
    return DivisionExpression((a, b))


def _div_linear_var(a, b):
    return DivisionExpression((a, b))


def _div_linear_monomial(a, b):
    return DivisionExpression((a, b))


def _div_linear_linear(a, b):
    return DivisionExpression((a, b))


def _div_linear_sum(a, b):
    return DivisionExpression((a, b))


def _div_linear_other(a, b):
    return DivisionExpression((a, b))


#
# DIV: SUM handlers
#


def _div_sum_native(a, b):
    if b in _zero_one_optimizations and b:
        return a
    if not b:
        raise ZeroDivisionError()
    return DivisionExpression((a, b))


def _div_sum_npv(a, b):
    return DivisionExpression((a, b))


def _div_sum_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        if not b:
            raise ZeroDivisionError()
    return DivisionExpression((a, b))


def _div_sum_var(a, b):
    return DivisionExpression((a, b))


def _div_sum_monomial(a, b):
    return DivisionExpression((a, b))


def _div_sum_linear(a, b):
    return DivisionExpression((a, b))


def _div_sum_sum(a, b):
    return DivisionExpression((a, b))


def _div_sum_other(a, b):
    return DivisionExpression((a, b))


#
# DIV: OTHER handlers
#


def _div_other_native(a, b):
    if b in _zero_one_optimizations and b:
        return a
    if not b:
        raise ZeroDivisionError()
    return DivisionExpression((a, b))


def _div_other_npv(a, b):
    return DivisionExpression((a, b))


def _div_other_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations and b:
            return a
        if not b:
            raise ZeroDivisionError()
    return DivisionExpression((a, b))


def _div_other_var(a, b):
    return DivisionExpression((a, b))


def _div_other_monomial(a, b):
    return DivisionExpression((a, b))


def _div_other_linear(a, b):
    return DivisionExpression((a, b))


def _div_other_sum(a, b):
    return DivisionExpression((a, b))


def _div_other_other(a, b):
    return DivisionExpression((a, b))


def _register_new_div_handler(a, b):
    types = _categorize_arg_types(a, b)
    # Retrieve the appropriate handler, record it in the main
    # _div_dispatcher dict (so this method is not called a second time for
    # these types)
    _div_dispatcher[a.__class__, b.__class__] = handler = _div_type_handler_mapping[
        types
    ]
    # Call the appropriate handler
    return handler(a, b)


_div_dispatcher = collections.defaultdict(lambda: _register_new_div_handler)

_div_type_handler_mapping = _binary_op_dispatcher_type_mapping(
    _div_dispatcher,
    {
        (ARG_TYPE.NATIVE, ARG_TYPE.NATIVE): _div_native_native,
        (ARG_TYPE.NATIVE, ARG_TYPE.NPV): _div_native_npv,
        (ARG_TYPE.NATIVE, ARG_TYPE.PARAM): _div_native_param,
        (ARG_TYPE.NATIVE, ARG_TYPE.VAR): _div_native_var,
        (ARG_TYPE.NATIVE, ARG_TYPE.MONOMIAL): _div_native_monomial,
        (ARG_TYPE.NATIVE, ARG_TYPE.LINEAR): _div_native_linear,
        (ARG_TYPE.NATIVE, ARG_TYPE.SUM): _div_native_sum,
        (ARG_TYPE.NATIVE, ARG_TYPE.OTHER): _div_native_other,
        (ARG_TYPE.NPV, ARG_TYPE.NATIVE): _div_npv_native,
        (ARG_TYPE.NPV, ARG_TYPE.NPV): _div_npv_npv,
        (ARG_TYPE.NPV, ARG_TYPE.PARAM): _div_npv_param,
        (ARG_TYPE.NPV, ARG_TYPE.VAR): _div_npv_var,
        (ARG_TYPE.NPV, ARG_TYPE.MONOMIAL): _div_npv_monomial,
        (ARG_TYPE.NPV, ARG_TYPE.LINEAR): _div_npv_linear,
        (ARG_TYPE.NPV, ARG_TYPE.SUM): _div_npv_sum,
        (ARG_TYPE.NPV, ARG_TYPE.OTHER): _div_npv_other,
        (ARG_TYPE.PARAM, ARG_TYPE.NATIVE): _div_param_native,
        (ARG_TYPE.PARAM, ARG_TYPE.NPV): _div_param_npv,
        (ARG_TYPE.PARAM, ARG_TYPE.PARAM): _div_param_param,
        (ARG_TYPE.PARAM, ARG_TYPE.VAR): _div_param_var,
        (ARG_TYPE.PARAM, ARG_TYPE.MONOMIAL): _div_param_monomial,
        (ARG_TYPE.PARAM, ARG_TYPE.LINEAR): _div_param_linear,
        (ARG_TYPE.PARAM, ARG_TYPE.SUM): _div_param_sum,
        (ARG_TYPE.PARAM, ARG_TYPE.OTHER): _div_param_other,
        (ARG_TYPE.VAR, ARG_TYPE.NATIVE): _div_var_native,
        (ARG_TYPE.VAR, ARG_TYPE.NPV): _div_var_npv,
        (ARG_TYPE.VAR, ARG_TYPE.PARAM): _div_var_param,
        (ARG_TYPE.VAR, ARG_TYPE.VAR): _div_var_var,
        (ARG_TYPE.VAR, ARG_TYPE.MONOMIAL): _div_var_monomial,
        (ARG_TYPE.VAR, ARG_TYPE.LINEAR): _div_var_linear,
        (ARG_TYPE.VAR, ARG_TYPE.SUM): _div_var_sum,
        (ARG_TYPE.VAR, ARG_TYPE.OTHER): _div_var_other,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NATIVE): _div_monomial_native,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NPV): _div_monomial_npv,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.PARAM): _div_monomial_param,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.VAR): _div_monomial_var,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.MONOMIAL): _div_monomial_monomial,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.LINEAR): _div_monomial_linear,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.SUM): _div_monomial_sum,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.OTHER): _div_monomial_other,
        (ARG_TYPE.LINEAR, ARG_TYPE.NATIVE): _div_linear_native,
        (ARG_TYPE.LINEAR, ARG_TYPE.NPV): _div_linear_npv,
        (ARG_TYPE.LINEAR, ARG_TYPE.PARAM): _div_linear_param,
        (ARG_TYPE.LINEAR, ARG_TYPE.VAR): _div_linear_var,
        (ARG_TYPE.LINEAR, ARG_TYPE.MONOMIAL): _div_linear_monomial,
        (ARG_TYPE.LINEAR, ARG_TYPE.LINEAR): _div_linear_linear,
        (ARG_TYPE.LINEAR, ARG_TYPE.SUM): _div_linear_sum,
        (ARG_TYPE.LINEAR, ARG_TYPE.OTHER): _div_linear_other,
        (ARG_TYPE.SUM, ARG_TYPE.NATIVE): _div_sum_native,
        (ARG_TYPE.SUM, ARG_TYPE.NPV): _div_sum_npv,
        (ARG_TYPE.SUM, ARG_TYPE.PARAM): _div_sum_param,
        (ARG_TYPE.SUM, ARG_TYPE.VAR): _div_sum_var,
        (ARG_TYPE.SUM, ARG_TYPE.MONOMIAL): _div_sum_monomial,
        (ARG_TYPE.SUM, ARG_TYPE.LINEAR): _div_sum_linear,
        (ARG_TYPE.SUM, ARG_TYPE.SUM): _div_sum_sum,
        (ARG_TYPE.SUM, ARG_TYPE.OTHER): _div_sum_other,
        (ARG_TYPE.OTHER, ARG_TYPE.NATIVE): _div_other_native,
        (ARG_TYPE.OTHER, ARG_TYPE.NPV): _div_other_npv,
        (ARG_TYPE.OTHER, ARG_TYPE.PARAM): _div_other_param,
        (ARG_TYPE.OTHER, ARG_TYPE.VAR): _div_other_var,
        (ARG_TYPE.OTHER, ARG_TYPE.MONOMIAL): _div_other_monomial,
        (ARG_TYPE.OTHER, ARG_TYPE.LINEAR): _div_other_linear,
        (ARG_TYPE.OTHER, ARG_TYPE.SUM): _div_other_sum,
        (ARG_TYPE.OTHER, ARG_TYPE.OTHER): _div_other_other,
    },
)


#
# POW handlers
#


def _pow_native_native(a, b):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return a**b


def _pow_native_npv(a, b):
    return NPV_PowExpression((a, b))


def _pow_native_param(a, b):
    if b.is_constant():
        return a ** (b.value)
    return NPV_PowExpression((a, b))


def _pow_native_other(a, b):
    return PowExpression((a, b))


def _pow_npv_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 1
    return NPV_PowExpression((a, b))


def _pow_npv_npv(a, b):
    return NPV_PowExpression((a, b))


def _pow_npv_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 1
    return NPV_PowExpression((a, b))


def _pow_npv_other(a, b):
    return PowExpression((a, b))


def _pow_param_native(a, b):
    if a.is_constant():
        return a.value**b
    if b in _zero_one_optimizations:
        return a if b else 1
    return NPV_PowExpression((a, b))


def _pow_param_npv(a, b):
    if a.is_constant():
        a = a.value
    return NPV_PowExpression((a, b))


def _pow_param_param(a, b):
    if a.is_constant():
        a = a.value
        if b.is_constant():
            return a**b.value
    elif b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 1
    return NPV_PowExpression((a, b))


def _pow_param_other(a, b):
    if a.is_constant():
        a = a.value
    return PowExpression((a, b))


def _pow_other_native(a, b):
    if b in _zero_one_optimizations:
        return a if b else 1
    return PowExpression((a, b))


def _pow_other_npv(a, b):
    return PowExpression((a, b))


def _pow_other_param(a, b):
    if b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 1
    return PowExpression((a, b))


def _pow_other_other(a, b):
    return PowExpression((a, b))


def _register_new_pow_handler(a, b):
    types = _categorize_arg_types(a, b)
    # Retrieve the appropriate handler, record it in the main
    # _pow_dispatcher dict (so this method is not called a second time for
    # these types)
    _pow_dispatcher[a.__class__, b.__class__] = handler = _pow_type_handler_mapping[
        types
    ]
    # Call the appropriate handler
    return handler(a, b)


_pow_dispatcher = collections.defaultdict(lambda: _register_new_pow_handler)

_pow_type_handler_mapping = _binary_op_dispatcher_type_mapping(
    _pow_dispatcher,
    {
        (ARG_TYPE.NATIVE, ARG_TYPE.NATIVE): _pow_native_native,
        (ARG_TYPE.NATIVE, ARG_TYPE.NPV): _pow_native_npv,
        (ARG_TYPE.NATIVE, ARG_TYPE.PARAM): _pow_native_param,
        (ARG_TYPE.NATIVE, ARG_TYPE.VAR): _pow_native_other,
        (ARG_TYPE.NATIVE, ARG_TYPE.MONOMIAL): _pow_native_other,
        (ARG_TYPE.NATIVE, ARG_TYPE.LINEAR): _pow_native_other,
        (ARG_TYPE.NATIVE, ARG_TYPE.SUM): _pow_native_other,
        (ARG_TYPE.NATIVE, ARG_TYPE.OTHER): _pow_native_other,
        (ARG_TYPE.NPV, ARG_TYPE.NATIVE): _pow_npv_native,
        (ARG_TYPE.NPV, ARG_TYPE.NPV): _pow_npv_npv,
        (ARG_TYPE.NPV, ARG_TYPE.PARAM): _pow_npv_param,
        (ARG_TYPE.NPV, ARG_TYPE.VAR): _pow_npv_other,
        (ARG_TYPE.NPV, ARG_TYPE.MONOMIAL): _pow_npv_other,
        (ARG_TYPE.NPV, ARG_TYPE.LINEAR): _pow_npv_other,
        (ARG_TYPE.NPV, ARG_TYPE.SUM): _pow_npv_other,
        (ARG_TYPE.NPV, ARG_TYPE.OTHER): _pow_npv_other,
        (ARG_TYPE.PARAM, ARG_TYPE.NATIVE): _pow_param_native,
        (ARG_TYPE.PARAM, ARG_TYPE.NPV): _pow_param_npv,
        (ARG_TYPE.PARAM, ARG_TYPE.PARAM): _pow_param_param,
        (ARG_TYPE.PARAM, ARG_TYPE.VAR): _pow_param_other,
        (ARG_TYPE.PARAM, ARG_TYPE.MONOMIAL): _pow_param_other,
        (ARG_TYPE.PARAM, ARG_TYPE.LINEAR): _pow_param_other,
        (ARG_TYPE.PARAM, ARG_TYPE.SUM): _pow_param_other,
        (ARG_TYPE.PARAM, ARG_TYPE.OTHER): _pow_param_other,
        (ARG_TYPE.VAR, ARG_TYPE.NATIVE): _pow_other_native,
        (ARG_TYPE.VAR, ARG_TYPE.NPV): _pow_other_npv,
        (ARG_TYPE.VAR, ARG_TYPE.PARAM): _pow_other_param,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NATIVE): _pow_other_native,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.NPV): _pow_other_npv,
        (ARG_TYPE.MONOMIAL, ARG_TYPE.PARAM): _pow_other_param,
        (ARG_TYPE.LINEAR, ARG_TYPE.NATIVE): _pow_other_native,
        (ARG_TYPE.LINEAR, ARG_TYPE.NPV): _pow_other_npv,
        (ARG_TYPE.LINEAR, ARG_TYPE.PARAM): _pow_other_param,
        (ARG_TYPE.SUM, ARG_TYPE.NATIVE): _pow_other_native,
        (ARG_TYPE.SUM, ARG_TYPE.NPV): _pow_other_npv,
        (ARG_TYPE.SUM, ARG_TYPE.PARAM): _pow_other_param,
        (ARG_TYPE.OTHER, ARG_TYPE.NATIVE): _pow_other_native,
        (ARG_TYPE.OTHER, ARG_TYPE.NPV): _pow_other_npv,
        (ARG_TYPE.OTHER, ARG_TYPE.PARAM): _pow_other_param,
    },
)
_pow_type_handler_mapping.update(
    {
        (i, j): _pow_other_other
        for i in ARG_TYPE
        for j in ARG_TYPE
        if (i, j) not in _pow_type_handler_mapping
    }
)


#
# ABS handlers
#


def _abs_native(a):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return abs(a)


def _abs_npv(a):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return NPV_AbsExpression((a,))


def _abs_param(a):
    if a.is_constant():
        return abs(a.value)
    return NPV_AbsExpression((a,))


def _abs_other(a):
    return AbsExpression((a,))


def _register_new_abs_handler(a):
    types = _categorize_arg_types(a)
    # Retrieve the appropriate handler, record it in the main
    # _abs_dispatcher dict (so this method is not called a second time for
    # these types)
    _abs_dispatcher[a.__class__] = handler = _abs_type_handler_mapping[types[0]]
    # Call the appropriate handler
    return handler(a)


_abs_dispatcher = collections.defaultdict(lambda: _register_new_abs_handler)

_abs_type_handler_mapping = _unary_op_dispatcher_type_mapping(
    _abs_dispatcher,
    {
        ARG_TYPE.NATIVE: _abs_native,
        ARG_TYPE.NPV: _abs_npv,
        ARG_TYPE.PARAM: _abs_param,
        ARG_TYPE.VAR: _abs_other,
        ARG_TYPE.MONOMIAL: _abs_other,
        ARG_TYPE.LINEAR: _abs_other,
        ARG_TYPE.SUM: _abs_other,
        ARG_TYPE.OTHER: _abs_other,
    },
)


#
# INTRINSIC FUNCTION handlers
#


def _fcn_asnumeric(a, name, fcn):
    a = a.as_numeric()
    return _fcn_dispatcher[a.__class__](a, name, fcn)


def _fcn_mutable(a, name, fcn):
    a = _recast_mutable(a)
    return _fcn_dispatcher[a.__class__](a, name, fcn)


def _fcn_invalid(a, name, fcn):
    fcn(a)
    # returns None


def _fcn_native(a, name, fcn):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return fcn(a)


def _fcn_npv(a, name, fcn):
    # This can be hit because of the asnumeric / mutable wrapper handlers.
    return NPV_UnaryFunctionExpression((a,), name, fcn)


def _fcn_param(a, name, fcn):
    if a.is_constant():
        return fcn(a.value)
    return NPV_UnaryFunctionExpression((a,), name, fcn)


def _fcn_other(a, name, fcn):
    return UnaryFunctionExpression((a,), name, fcn)


def _register_new_fcn_dispatcher(a, name, fcn):
    types = _categorize_arg_types(a)
    # Retrieve the appropriate handler, record it in the main
    # _fcn_dispatcher dict (so this method is not called a second time for
    # these types)
    _fcn_dispatcher[a.__class__] = handler = _fcn_type_handler_mapping[types[0]]
    # Call the appropriate handler
    return handler(a, name, fcn)


_fcn_dispatcher = collections.defaultdict(lambda: _register_new_fcn_dispatcher)

_fcn_type_handler_mapping = {
    ARG_TYPE.ASNUMERIC: _fcn_asnumeric,
    ARG_TYPE.MUTABLE: _fcn_mutable,
    ARG_TYPE.INVALID: _fcn_invalid,
    ARG_TYPE.NATIVE: _fcn_native,
    ARG_TYPE.NPV: _fcn_npv,
    ARG_TYPE.PARAM: _fcn_param,
    ARG_TYPE.VAR: _fcn_other,
    ARG_TYPE.MONOMIAL: _fcn_other,
    ARG_TYPE.LINEAR: _fcn_other,
    ARG_TYPE.SUM: _fcn_other,
    ARG_TYPE.OTHER: _fcn_other,
}


#
# NOTE: abs() and pow() are not defined here, because they are
# Python operators.
#
def ceil(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'ceil', math.ceil)


def floor(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'floor', math.floor)


# e ** x
def exp(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'exp', math.exp)


def log(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'log', math.log)


def log10(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'log10', math.log10)


# FIXME: this is nominally the same as x ** 0.5, but follows a different
# path and produces a different NL file!
def sqrt(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'sqrt', math.sqrt)
    # return _pow_dispatcher[arg.__class__, float](arg, 0.5)


def sin(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'sin', math.sin)


def cos(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'cos', math.cos)


def tan(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'tan', math.tan)


def sinh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'sinh', math.sinh)


def cosh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'cosh', math.cosh)


def tanh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'tanh', math.tanh)


def asin(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'asin', math.asin)


def acos(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'acos', math.acos)


def atan(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'atan', math.atan)


def asinh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'asinh', math.asinh)


def acosh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'acosh', math.acosh)


def atanh(arg):
    return _fcn_dispatcher[arg.__class__](arg, 'atanh', math.atanh)


#
# Function interface to Expr_ifExpression
#


def _process_expr_if_arg(arg, kwargs, name):
    alt = kwargs.pop(name, None)
    if alt is not None:
        if arg is not None:
            raise ValueError(f'Cannot specify both {name}_ and {name}')
        arg = alt
    _type = _categorize_arg_type(arg)
    # Note that relational expressions get mapped to INVALID
    while _type < ARG_TYPE.INVALID:
        if _type is ARG_TYPE.MUTABLE:
            arg = _recast_mutable(arg)
        elif _type is ARG_TYPE.ASNUMERIC:
            arg = arg.as_numeric()
        else:
            raise DeveloperError('_categorize_arg_type() returned unexpected ARG_TYPE')
        _type = _categorize_arg_type(arg)
    return arg, _type


def Expr_if(IF_=None, THEN_=None, ELSE_=None, **kwargs):
    """
    Function used to construct a conditional numeric expression.

    This function accepts either of the following signatures:

       - Expr_if(IF={expr}, THEN={expr}, ELSE={expr})
       - Expr_if(IF_={expr}, THEN_={expr}, ELSE_={expr})

    (the former is historical, and the latter is required to support Cythonization)
    """
    _pv = False
    ELSE_, _type = _process_expr_if_arg(ELSE_, kwargs, 'ELSE')
    _pv |= _type >= ARG_TYPE.VAR or _type == ARG_TYPE.INVALID
    THEN_, _type = _process_expr_if_arg(THEN_, kwargs, 'THEN')
    _pv |= _type >= ARG_TYPE.VAR or _type == ARG_TYPE.INVALID
    IF_, _type = _process_expr_if_arg(IF_, kwargs, 'IF')
    _pv |= _type >= ARG_TYPE.VAR or _type == ARG_TYPE.INVALID
    if kwargs:
        raise ValueError('Unrecognized arguments: ' + ', '.join(kwargs))
    # Notes:
    # - side effect: IF is the last iteration, so _type == _categorize_arg_type(IF)
    # - we do NO error checking as to the actual arg types.  That is
    #   left to the writer (and as of writing [Jul 2023], the NL writer
    #   is the only writer that recognized Expr_if)
    if _type is ARG_TYPE.NATIVE:
        return THEN_ if IF_ else ELSE_
    elif _type is ARG_TYPE.PARAM and IF_.is_constant():
        return THEN_ if IF_.value else ELSE_
    elif _pv:
        return Expr_ifExpression((IF_, THEN_, ELSE_))
    else:
        return NPV_Expr_ifExpression((IF_, THEN_, ELSE_))


#
# Misc (legacy) functions
#


def _balanced_parens(arg):
    """Verify the string argument contains balanced parentheses.

    This checks that every open paren is balanced by a closed paren.
    That is, the infix string expression is likely to be valid.  This is
    primarily used to determine if a string that starts and ends with
    parens can have those parens removed.

    Examples:
        >>> a = "(( x + y ) * ( z - w ))"
        >>> _balanced_parens(a[1:-1])
        True
        >>> a = "( x + y ) * ( z - w )"
        >>> _balanced_parens(a[1:-1])
        False
    """
    _parenCount = 0
    for c in arg:
        if c == '(':
            _parenCount += 1
        elif c == ')':
            _parenCount -= 1
            if _parenCount < 0:
                return False
    return _parenCount == 0


# TODO: this is fragile (and not currently used anywhere).  It should be
# deprecated / removed.
NPV_expression_types = set(
    [
        NPV_NegationExpression,
        NPV_ExternalFunctionExpression,
        NPV_PowExpression,
        NPV_MinExpression,
        NPV_MaxExpression,
        NPV_ProductExpression,
        NPV_DivisionExpression,
        NPV_SumExpression,
        NPV_UnaryFunctionExpression,
        NPV_AbsExpression,
    ]
)
