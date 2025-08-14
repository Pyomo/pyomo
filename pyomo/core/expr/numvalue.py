#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import logging

from pyomo.common.deprecation import (
    deprecated,
    deprecation_warning,
    relocated_module_attribute,
)
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.expr_common import ExpressionType, _type_check_exception_arg
from pyomo.core.expr.numeric_expr import NumericValue

# TODO: update Pyomo to import these objects from common.numeric_types
#   (and not from here)
from pyomo.common.numeric_types import (
    nonpyomo_leaf_types,
    native_types,
    native_numeric_types,
    native_integer_types,
    native_logical_types,
    _pyomo_constant_types,
    check_if_numeric_type,
    value,
)
from pyomo.core.pyomoobject import PyomoObject

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
    'pyomo_constant_types',
    'pyomo.common.numeric_types._pyomo_constant_types',
    version='6.7.2',
    f_globals=globals(),
    msg="The pyomo_constant_types set will be removed in the future: the set "
    "contained only NumericConstant and _PythonCallbackFunctionID, and provided "
    "no meaningful value to clients or walkers.  Users should likely handle "
    "these types in the same manner as immutable Params.",
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
relocated_module_attribute(
    'NumericValue',
    'pyomo.core.expr.numeric_expr.NumericValue',
    version='6.6.2',
    f_globals=globals(),
)
relocated_module_attribute(
    'NumericNDArray',
    'pyomo.core.expr.numeric_expr.NumericNDArray',
    version='6.6.2',
    f_globals=globals(),
)

logger = logging.getLogger('pyomo.core')


##------------------------------------------------------------------------
##
## Standard types of expressions
##
##------------------------------------------------------------------------


class NonNumericValue(PyomoObject):
    """An object that contains a non-numeric value

    Constructor Arguments:
        value           The initial value.
    """

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)

    def __call__(self, exception=NOTSET):
        exception = _type_check_exception_arg(self, exception)
        return self.value

    def is_constant(self):
        return True

    def is_fixed(self):
        return True


nonpyomo_leaf_types.add(NonNumericValue)


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


# Note:
#   For now, all constants are coerced to floats.  This avoids integer
#   division in Python 2.x.  (At least some of the time.)
#
#   When we eliminate support for Python 2.x, we will not need this
#   coercion.  The main difference in the following code is that we will
#   need to index KnownConstants by both the class type and value, since
#   INT, FLOAT and LONG values sometimes hash the same.
#
# It is very common to have only a few constants in a model, but those
# constants get repeated many times.  KnownConstants lets us re-use /
# share constants we have seen before.
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

    def __call__(self, exception=NOTSET):
        """Return the constant value"""
        exception = _type_check_exception_arg(self, exception)
        return self.value

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:  # pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))


_pyomo_constant_types.add(NumericConstant)

# We use as_numeric() so that the constant is also in the cache
ZeroConstant = as_numeric(0)
