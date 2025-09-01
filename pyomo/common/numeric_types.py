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

import logging
import sys

from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import TemplateExpressionError

logger = logging.getLogger(__name__)

#: Python set used to identify numeric constants, boolean values, strings
#: and instances of
#: :class:`NonNumericValue <pyomo.core.expr.numvalue.NonNumericValue>`,
#: which is commonly used in code that walks Pyomo expression trees.
#:
#: :data:`nonpyomo_leaf_types` = :data:`native_types <pyomo.core.expr.numvalue.native_types>` + { :data:`NonNumericValue <pyomo.core.expr.numvalue.NonNumericValue>` }
nonpyomo_leaf_types = set()

# It is *significantly* faster to build the list of types we want to
# test against as a "static" set, and not to regenerate it locally for
# every call.  Plus, this allows us to dynamically augment the set
# with new "native" types (e.g., from NumPy)
#
# Note: These type sets are used in set_types.py for domain validation
#       For backward compatibility reasons, we include str in the set
#       of valid types for bool. We also avoid updating the numeric
#       and integer type sets when a new boolean type is registered
#       because not all boolean types exhibit numeric properties
#       (e.g., numpy.bool_)
#

#: Python set used to identify numeric constants.  This set includes
#: native Python types as well as numeric types from Python packages
#: like numpy, which may be registered by users.
#:
#: Note that :data:`native_numeric_types` does NOT include
#: :py:class:`complex`, as that is not a valid constant in Pyomo numeric
#: expressions.
native_numeric_types = {int, float}
native_integer_types = {int}
native_logical_types = {bool}
native_complex_types = {complex}

_native_boolean_types = {int, bool, str, bytes}
relocated_module_attribute(
    'native_boolean_types',
    'pyomo.common.numeric_types._native_boolean_types',
    version='6.6.0',
    msg="The native_boolean_types set will be removed in the future: the set "
    "contains types that were convertible to bool, and not types that should "
    "be treated as if they were bool (as was the case for the other "
    "native_*_types sets).  Users likely should use native_logical_types.",
)
_pyomo_constant_types = set()  # includes NumericConstant, _PythonCallbackFunctionID
relocated_module_attribute(
    'pyomo_constant_types',
    'pyomo.common.numeric_types._pyomo_constant_types',
    version='6.7.2',
    msg="The pyomo_constant_types set will be removed in the future: the set "
    "contained only NumericConstant and _PythonCallbackFunctionID, and provided "
    "no meaningful value to clients or walkers.  Users should likely handle "
    "these types in the same manner as immutable Params.",
)


#: Python set used to identify numeric constants and related native
#: types.  This set includes
#: native Python types as well as numeric types from Python packages
#: like numpy.
#:
#: :data:`native_types` = :data:`native_numeric_types <pyomo.core.expr.numvalue.native_numeric_types>` + { str }
native_types = {bool, str, type(None), slice, bytes}
native_types.update(native_numeric_types)
native_types.update(native_integer_types)
native_types.update(native_complex_types)
native_types.update(native_logical_types)
native_types.update(_native_boolean_types)

nonpyomo_leaf_types.update(native_types)


def RegisterNumericType(new_type: type):
    """Register the specified type as a "numeric type".

    A utility function for registering new types as "native numeric
    types" that can be leaf nodes in Pyomo numeric expressions.  The
    type should be compatible with :py:class:`float` (that is, store a
    scalar and be castable to a Python float).

    Parameters
    ----------
    new_type : type
        The new numeric type (e.g, `numpy.float64`)

    """
    native_numeric_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)


def RegisterIntegerType(new_type: type):
    """Register the specified type as an "integer type".

    A utility function for registering new types as "native integer
    types".  Integer types can be leaf nodes in Pyomo numeric
    expressions.  The type should be compatible with :py:class:`float`
    (that is, store a scalar and be castable to a Python float).

    Registering a type as an integer type implies
    :py:func:`RegisterNumericType`.

    Note that integer types are NOT registered as logical / Boolean types.

    Parameters
    ----------
    new_type : type
        The new integer type (e.g, `numpy.int64`)

    """
    native_numeric_types.add(new_type)
    native_integer_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)


@deprecated(
    "The native_boolean_types set (and hence RegisterBooleanType) "
    "is deprecated.  Users likely should use RegisterLogicalType.",
    version='6.6.0',
)
def RegisterBooleanType(new_type: type):
    """Register the specified type as a "logical type".

    A utility function for registering new types as "native logical
    types".  Logical types can be leaf nodes in Pyomo logical
    expressions.  The type should be compatible with :py:class:`bool`
    (that is, store a scalar and be castable to a Python bool).

    Note that logical types are NOT registered as numeric types.

    Parameters
    ----------
    new_type : type
        The new logical type (e.g, `numpy.bool_`)

    """
    _native_boolean_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)


def RegisterComplexType(new_type: type):
    """Register the specified type as an "complex type".

    A utility function for registering new types as "native complex
    types".  Complex types can NOT be leaf nodes in Pyomo numeric
    expressions.  The type should be compatible with :py:class:`complex`
    (that is, store a scalar complex value and be castable to a Python
    complex).

    Note that complex types are NOT registered as logical or numeric types.

    Parameters
    ----------
    new_type : type
        The new complex type (e.g, `numpy.complex128`)

    """
    native_types.add(new_type)
    native_complex_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)


def RegisterLogicalType(new_type: type):
    """Register the specified type as a "logical type".

    A utility function for registering new types as "native logical
    types".  Logical types can be leaf nodes in Pyomo logical
    expressions.  The type should be compatible with :py:class:`bool`
    (that is, store a scalar and be castable to a Python bool).

    Note that logical types are NOT registered as numeric types.

    Parameters
    ----------
    new_type : type
        The new logical type (e.g, `numpy.bool_`)

    """
    _native_boolean_types.add(new_type)
    native_logical_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)


def check_if_native_type(obj):
    if isinstance(obj, (str, bytes)):
        native_types.add(obj.__class__)
        return True
    if check_if_logical_type(obj):
        return True
    if check_if_numeric_type(obj):
        return True
    return False


def check_if_logical_type(obj):
    """Test if the argument behaves like a logical type.

    We check for "logical types" by checking if the type returns sane
    results for Boolean operators (``^``, ``|``, ``&``) and if it maps
    ``1`` and ``2`` both to the same equivalent instance.  If that
    works, then we register the type in :py:attr:`native_logical_types`.

    """
    obj_class = obj.__class__
    # Do not re-evaluate known native types
    if obj_class in native_types:
        return obj_class in native_logical_types

    try:
        # It is not an error if you can't initialize the type from an
        # int, but if you can, it should map !0 to True
        if obj_class(1) != obj_class(2):
            return False
    except:
        pass

    try:
        # Native logical types *must* be hashable
        hash(obj)
        # Native logical types must honor standard Boolean operators
        if all(
            (
                obj_class(False) != obj_class(True),
                obj_class(False) ^ obj_class(False) == obj_class(False),
                obj_class(False) ^ obj_class(True) == obj_class(True),
                obj_class(True) ^ obj_class(False) == obj_class(True),
                obj_class(True) ^ obj_class(True) == obj_class(False),
                obj_class(False) | obj_class(False) == obj_class(False),
                obj_class(False) | obj_class(True) == obj_class(True),
                obj_class(True) | obj_class(False) == obj_class(True),
                obj_class(True) | obj_class(True) == obj_class(True),
                obj_class(False) & obj_class(False) == obj_class(False),
                obj_class(False) & obj_class(True) == obj_class(False),
                obj_class(True) & obj_class(False) == obj_class(False),
                obj_class(True) & obj_class(True) == obj_class(True),
            )
        ):
            RegisterLogicalType(obj_class)
            return True
    except:
        pass
    return False


def check_if_numeric_type(obj):
    """Test if the argument behaves like a numeric type.

    We check for "numeric types" by checking if we can add zero to it
    without changing the object's type, and that the object compares to
    0 in a meaningful way.  If that works, then we register the type in
    :py:attr:`native_numeric_types`.

    """
    obj_class = obj.__class__
    # Do not re-evaluate known native types
    if obj_class in native_types:
        return obj_class in native_numeric_types

    try:
        obj_plus_0 = obj + 0
        obj_p0_class = obj_plus_0.__class__
        # Native numeric types *must* be hashable
        hash(obj)
    except:
        return False
    if obj_p0_class is not obj_class and obj_p0_class not in native_numeric_types:
        return False
    #
    # Check if the numeric type behaves like a complex type
    #
    try:
        if 1.41 < abs(obj_class(1j + 1)) < 1.42:
            RegisterComplexType(obj_class)
            return False
    except:
        pass
    #
    # Ensure that the object is comparable to 0 in a meaningful way
    #
    try:
        if not ((obj < 0) ^ (obj >= 0)):
            return False
    except:
        return False
    #
    # If we get here, this is a reasonably well-behaving
    # numeric type: add it to the native numeric types
    # so that future lookups will be faster.
    #
    RegisterNumericType(obj_class)
    try:
        if obj_class(0.4) == obj_class(0):
            RegisterIntegerType(obj_class)
    except:
        pass
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
    #
    # Test if we have a duck typed Pyomo expression
    #
    if not hasattr(obj, 'is_numeric_type'):
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
            )
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
                    "No value for uninitialized %s object %s"
                    % (type(obj).__name__, obj.name)
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
