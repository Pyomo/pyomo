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

__all__ = (
    'value',
    'is_constant',
    'is_fixed',
    'is_variable_type',
    'is_potentially_variable',
    'NumericValue',
    'ZeroConstant',
    'native_numeric_types',
    'native_types',
    'nonpyomo_leaf_types',
    'polynomial_degree',
)

import collections
import sys
import logging

from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.deprecation import (
    deprecated,
    deprecation_warning,
    relocated_module_attribute,
)
from pyomo.common.errors import PyomoException
from pyomo.core.expr.expr_common import (
    _add,
    _sub,
    _mul,
    _div,
    _pow,
    _neg,
    _abs,
    _radd,
    _rsub,
    _rmul,
    _rdiv,
    _rpow,
    _iadd,
    _isub,
    _imul,
    _idiv,
    _ipow,
    _lt,
    _le,
    _eq,
    ExpressionType,
)
import pyomo.common.numeric_types as _numeric_types

# TODO: update Pyomo to import these objects from common.numeric_types
#   (and not from here)
from pyomo.common.numeric_types import (
    nonpyomo_leaf_types,
    native_types,
    native_numeric_types,
    native_integer_types,
    native_logical_types,
    pyomo_constant_types,
)
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_errors import TemplateExpressionError

relocated_module_attribute(
    'native_boolean_types',
    'pyomo.common.numeric_types._native_boolean_types',
    version='6.6.0',
    f_globals=globals(),
    msg="The native_boolean_types set will be removed in the future: the set "
    "contains types that were convertible to bool, and not types that should "
    "be treated as if they were bool (as was the case for the other "
    "native_*_types sets).  Users likely should use native_logical_types.",
)
relocated_module_attribute(
    'RegisterNumericType',
    'pyomo.common.numeric_types.RegisterNumericType',
    version='6.6.0',
    f_globals=globals(),
)
relocated_module_attribute(
    'RegisterIntegerType',
    'pyomo.common.numeric_types.RegisterIntegerType',
    version='6.6.0',
    f_globals=globals(),
)
relocated_module_attribute(
    'RegisterBooleanType',
    'pyomo.common.numeric_types.RegisterBooleanType',
    version='6.6.0',
    f_globals=globals(),
)

logger = logging.getLogger('pyomo.core')


# Stub in the dispatchers
def _incomplete_import(*args):
    raise RuntimeError("incomplete import of Pyomo expression system")


_add_dispatcher = collections.defaultdict(_incomplete_import)
_mul_dispatcher = collections.defaultdict(_incomplete_import)
_div_dispatcher = collections.defaultdict(_incomplete_import)
_pow_dispatcher = collections.defaultdict(_incomplete_import)
_neg_dispatcher = collections.defaultdict(_incomplete_import)
_abs_dispatcher = collections.defaultdict(_incomplete_import)


def _generate_relational_expression(etype, lhs, rhs):
    raise RuntimeError("incomplete import of Pyomo expression system")


##------------------------------------------------------------------------
##
## Standard types of expressions
##
##------------------------------------------------------------------------


class NonNumericValue(object):
    """An object that contains a non-numeric value

    Constructor Arguments:
        value           The initial value.
    """

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


nonpyomo_leaf_types.add(NonNumericValue)


def value(obj, exception=True):
    """
    A utility function that returns the value of a Pyomo object or
    expression.

    Args:
        obj: The argument to evaluate. If it is None, a
            string, or any other primitive numeric type,
            then this function simply returns the argument.
            Otherwise, if the argument is a NumericValue
            then the __call__ method is executed.
        exception (bool): If :const:`True`, then an exception should
            be raised when instances of NumericValue fail to
            evaluate due to one or more objects not being
            initialized to a numeric value (e.g, one or more
            variables in an algebraic expression having the
            value None). If :const:`False`, then the function
            returns :const:`None` when an exception occurs.
            Default is True.

    Returns: A numeric value or None.
    """
    if obj.__class__ in native_types:
        return obj
    if obj.__class__ in pyomo_constant_types:
        #
        # I'm commenting this out for now, but I think we should never expect
        # to see a numeric constant with value None.
        #
        # if exception and obj.value is None:
        #    raise ValueError(
        #        "No value for uninitialized NumericConstant object %s"
        #        % (obj.name,))
        return obj.value
    #
    # Test if we have a duck typed Pyomo expression
    #
    try:
        obj.is_numeric_type()
    except AttributeError:
        #
        # TODO: Historically we checked for new *numeric* types and
        # raised exceptions for anything else.  That is inconsistent
        # with allowing native_types like None/str/bool to be returned
        # from value().  We should revisit if that is worthwhile to do
        # here.
        #
        if check_if_numeric_type(obj):
            return obj
        else:
            if not exception:
                return None
            raise TypeError(
                "Cannot evaluate object with unknown type: %s" % obj.__class__.__name__
            ) from None
    #
    # Evaluate the expression object
    #
    if exception:
        #
        # Here, we try to catch the exception
        #
        try:
            tmp = obj(exception=True)
            if tmp is None:
                raise ValueError(
                    "No value for uninitialized NumericValue object %s" % (obj.name,)
                )
            return tmp
        except TemplateExpressionError:
            # Template expressions work by catching this error type. So
            # we should defer this error handling and not log an error
            # message.
            raise
        except:
            logger.error(
                "evaluating object as numeric value: %s\n    (object: %s)\n%s"
                % (obj, type(obj), sys.exc_info()[1])
            )
            raise
    else:
        #
        # Here, we do not try to catch the exception
        #
        return obj(exception=False)


def is_constant(obj):
    """
    A utility function that returns a boolean that indicates
    whether the object is a constant.
    """
    # JDS: NB: I am not sure why we allow str to be a constant, but
    # since we have historically done so, we check for type membership
    # in native_types and not in native_numeric_types.
    #
    if obj.__class__ in native_types:
        return True
    try:
        return obj.is_constant()
    except AttributeError:
        pass
    # Now we need to confirm that we have an unknown numeric type
    #
    # As this branch is only hit for previously unknown (to Pyomo)
    # types that behave reasonably like numbers, we know they *must*
    # be constant.
    if check_if_numeric_type(obj):
        return True
    else:
        raise TypeError(
            "Cannot assess properties of object with unknown type: %s"
            % (type(obj).__name__,)
        )


def is_fixed(obj):
    """
    A utility function that returns a boolean that indicates
    whether the input object's value is fixed.
    """
    # JDS: NB: I am not sure why we allow str to be a constant, but
    # since we have historically done so, we check for type membership
    # in native_types and not in native_numeric_types.
    #
    if obj.__class__ in native_types:
        return True
    try:
        return obj.is_fixed()
    except AttributeError:
        pass
    # Now we need to confirm that we have an unknown numeric type
    #
    # As this branch is only hit for previously unknown (to Pyomo)
    # types that behave reasonably like numbers, we know they *must*
    # be fixed.
    if check_if_numeric_type(obj):
        return True
    else:
        raise TypeError(
            "Cannot assess properties of object with unknown type: %s"
            % (type(obj).__name__,)
        )


def is_variable_type(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object is a variable.
    """
    if obj.__class__ in native_types:
        return False
    try:
        return obj.is_variable_type()
    except AttributeError:
        return False


def is_potentially_variable(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object can reference variables.
    """
    if obj.__class__ in native_types:
        return False
    try:
        return obj.is_potentially_variable()
    except AttributeError:
        return False


def is_numeric_data(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object is numeric and not potentially
    variable.
    """
    if obj.__class__ in native_numeric_types:
        return True
    elif obj.__class__ in native_types:
        # this likely means it is a string
        return False
    try:
        # Test if this is an expression object that
        # is not potentially variable
        return not obj.is_potentially_variable()
    except AttributeError:
        pass
    # Now we need to confirm that we have an unknown numeric type
    #
    # As this branch is only hit for previously unknown (to Pyomo)
    # types that behave reasonably like numbers, we know they *must*
    # be numeric data (unless an exception is raised).
    return check_if_numeric_type(obj)


def polynomial_degree(obj):
    """
    A utility function that returns an integer
    that indicates the polynomial degree for an
    object. boolean indicating
    """
    if obj.__class__ in native_numeric_types:
        return 0
    elif obj.__class__ in native_types:
        raise TypeError(
            "Cannot evaluate the polynomial degree of a non-numeric type: %s"
            % (type(obj).__name__,)
        )
    try:
        return obj.polynomial_degree()
    except AttributeError:
        pass
    # Now we need to confirm that we have an unknown numeric type
    if check_if_numeric_type(obj):
        # As this branch is only hit for previously unknown (to Pyomo)
        # types that behave reasonably like numbers, we know they *must*
        # be a numeric constant.
        return 0
    else:
        raise TypeError(
            "Cannot assess properties of object with unknown type: %s"
            % (type(obj).__name__,)
        )


#
# It is very common to have only a few constants in a model, but those
# constants get repeated many times.  KnownConstants lets us re-use /
# share constants we have seen before.
#
# Note:
#   For now, all constants are coerced to floats.  This avoids integer
#   division in Python 2.x.  (At least some of the time.)
#
#   When we eliminate support for Python 2.x, we will not need this
#   coercion.  The main difference in the following code is that we will
#   need to index KnownConstants by both the class type and value, since
#   INT, FLOAT and LONG values sometimes hash the same.
#
_KnownConstants = {}


def as_numeric(obj):
    """
    A function that creates a NumericConstant object that
    wraps Python numeric values.

    This function also manages a cache of constants.

    NOTE:  This function is only intended for use when
        data is added to a component.

    Args:
        obj: The numeric value that may be wrapped.

    Raises: TypeError if the object is in native_types and not in
        native_numeric_types

    Returns: A NumericConstant object or the original object.
    """
    if obj.__class__ in native_numeric_types:
        val = _KnownConstants.get(obj, None)
        if val is not None:
            return val
        #
        # Coerce the value to a float, if possible
        #
        try:
            obj = float(obj)
        except:
            pass
        #
        # Create the numeric constant.  This really
        # should be the only place in the code
        # where these objects are constructed.
        #
        retval = NumericConstant(obj)
        #
        # Cache the numeric constants.  We used a bounded cache size
        # to avoid unexpectedly large lists of constants.  There are
        # typically a small number of constants that need to be cached.
        #
        # NOTE:  A LFU policy might be more sensible here, but that
        # requires a more complex cache.  It's not clear that that
        # is worth the extra cost.
        #
        if len(_KnownConstants) < 1024:
            # obj may (or may not) be hashable, so we need this try
            # block so that things proceed normally for non-hashable
            # "numeric" types
            try:
                _KnownConstants[obj] = retval
            except:
                pass
        #
        return retval
    #
    # Ignore objects that are duck typed to work with Pyomo expressions
    #
    try:
        if obj.is_numeric_type():
            return obj
        elif obj.is_expression_type(ExpressionType.RELATIONAL):
            deprecation_warning(
                "returning a relational expression from as_numeric().  "
                "Relational expressions are no longer numeric types.  "
                "In the future this will raise a TypeError.",
                version='6.4.3',
            )
            return obj
        else:
            try:
                _name = obj.name
            except AttributeError:
                _name = str(obj)
            raise TypeError(
                "The '%s' object '%s' is not a valid type for Pyomo "
                "numeric expressions" % (type(obj).__name__, _name)
            )

    except AttributeError:
        pass
    #
    # Test if the object looks like a number.  If so, re-call as_numeric
    # (this type will have been added to native_numeric_types).
    #
    if check_if_numeric_type(obj):
        return as_numeric(obj)
    #
    # Generate errors
    #
    if obj.__class__ in native_types:
        raise TypeError(
            "%s values ('%s') are not allowed in Pyomo "
            "numeric expressions" % (type(obj).__name__, str(obj))
        )
    raise TypeError(
        "Cannot treat the value '%s' as a numeric value because it has "
        "unknown type '%s'" % (str(obj), type(obj).__name__)
    )


def check_if_numeric_type(obj):
    """Test if the argument behaves like a numeric type.

    We check for "numeric types" by checking if we can add zero to it
    without changing the object's type.  If that works, then we register
    the type in native_numeric_types.

    """
    obj_class = obj.__class__
    # Do not re-evaluate known native types
    if obj_class in native_types:
        return obj_class in native_numeric_types

    try:
        obj_plus_0 = obj + 0
        obj_p0_class = obj_plus_0.__class__
        # ensure that the object is comparable to 0 in a meaningful way
        # (among other things, this prevents numpy.ndarray objects from
        # being added to native_numeric_types)
        if not ((obj < 0) ^ (obj >= 0)):
            return False
        # Native types *must* be hashable
        hash(obj)
    except:
        return False
    if obj_p0_class is obj_class or obj_p0_class in native_numeric_types:
        #
        # If we get here, this is a reasonably well-behaving
        # numeric type: add it to the native numeric types
        # so that future lookups will be faster.
        #
        _numeric_types.RegisterNumericType(obj_class)
        #
        # Generate a warning, since Pyomo's management of third-party
        # numeric types is more robust when registering explicitly.
        #
        logger.warning(
            f"""Dynamically registering the following numeric type:
    {obj_class.__module__}.{obj_class.__name__}
Dynamic registration is supported for convenience, but there are known
limitations to this approach.  We recommend explicitly registering
numeric types using RegisterNumericType() or RegisterIntegerType()."""
        )
        return True
    else:
        return False


@deprecated(
    "check_if_numeric_type_and_cache() has been deprecated in "
    "favor of just calling as_numeric()",
    version='6.4.3',
)
def check_if_numeric_type_and_cache(obj):
    """Test if the argument is a numeric type by checking if we can add
    zero to it.  If that works, then we cache the value and return a
    NumericConstant object.

    """
    if check_if_numeric_type(obj):
        return as_numeric(obj)
    else:
        return obj


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


class NumericConstant(NumericValue):
    """An object that contains a constant numeric value.

    Constructor Arguments:
        value           The initial value.
    """

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def is_constant(self):
        return True

    def is_fixed(self):
        return True

    def _compute_polynomial_degree(self, result):
        return 0

    def __str__(self):
        return str(self.value)

    def __call__(self, exception=True):
        """Return the constant value"""
        return self.value

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:  # pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))


pyomo_constant_types.add(NumericConstant)

# We use as_numeric() so that the constant is also in the cache
ZeroConstant = as_numeric(0)


#
# Note: the "if numpy_available" in the class definition also ensures
# that the numpy types are registered if numpy is in fact available
#


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
