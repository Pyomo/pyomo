#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


#: Python set used to identify numeric constants, boolean values, strings
#: and instances of
#: :class:`NonNumericValue <pyomo.core.expr.numvalue.NonNumericValue>`,
#: which is commonly used in code that walks Pyomo expression trees.
#:
#: :data:`nonpyomo_leaf_types` = :data:`native_types <pyomo.core.expr.numvalue.native_types>` + { :data:`NonNumericValue <pyomo.core.expr.numvalue.NonNumericValue>` }
nonpyomo_leaf_types = set([])

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
native_numeric_types = set([ int, float, bool ])
native_integer_types = set([ int, bool ])
native_boolean_types = set([ int, bool, str, bytes ])
native_logical_types = {bool, }
pyomo_constant_types = set()  # includes NumericConstant

#: Python set used to identify numeric constants and related native
#: types.  This set includes
#: native Python types as well as numeric types from Python packages
#: like numpy.
#:
#: :data:`native_types` = :data:`native_numeric_types <pyomo.core.expr.numvalue.native_numeric_types>` + { str }
native_types = set([ bool, str, type(None), slice, bytes])
native_types.update( native_numeric_types )
native_types.update( native_integer_types )
native_types.update( native_boolean_types )

nonpyomo_leaf_types.update( native_types )

def RegisterNumericType(new_type):
    """
    A utility function for updating the set of types that are
    recognized to handle numeric values.

    The argument should be a class (e.g, numpy.float64).
    """
    native_numeric_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)

def RegisterIntegerType(new_type):
    """
    A utility function for updating the set of types that are
    recognized to handle integer values. This also registers the type
    as numeric but does not register it as boolean.

    The argument should be a class (e.g., numpy.int64).
    """
    native_numeric_types.add(new_type)
    native_integer_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)

def RegisterBooleanType(new_type):
    """
    A utility function for updating the set of types that are
    recognized as handling boolean values. This function does not
    register the type of integer or numeric.

    The argument should be a class (e.g., numpy.bool_).
    """
    native_boolean_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)
