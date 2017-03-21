#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = []

from pyomo.core.base.numvalue import \
   RegisterNumericType, \
   RegisterIntegerType, \
   RegisterBooleanType

try:
    import numpy
    _has_numpy = True
except:
    _has_numpy = False

#
# Collect NumPy Types
#
# The full list of available types is dependent on numpy version and
# the compiler used to build numpy. The following list of types was
# taken from the NumPy v1.8 Manual at scipy.org
#

# Boolean
numpy_bool_names = []
numpy_bool_names.append('bool_')
numpy_bool = []
if _has_numpy:
    for _type_name in numpy_bool_names:
        try:
            _type = getattr(numpy,_type_name)
            numpy_bool.append(_type)
        except:
            pass

# Integers
numpy_int_names = []
numpy_int_names.append('int_')
numpy_int_names.append('intc')
numpy_int_names.append('intp')
numpy_int_names.append('int8')
numpy_int_names.append('int16')
numpy_int_names.append('int32')
numpy_int_names.append('int64')
numpy_int_names.append('uint8')
numpy_int_names.append('uint16')
numpy_int_names.append('uint32')
numpy_int_names.append('uint64')
numpy_int = []
if _has_numpy:
    for _type_name in numpy_int_names:
        try:
            _type = getattr(numpy,_type_name)
            numpy_int.append(_type)
        except:
            pass

# Reals
numpy_float_names = []
numpy_float_names.append('float_')
numpy_float_names.append('float16')
numpy_float_names.append('float32')
numpy_float_names.append('float64')
numpy_float = []
if _has_numpy:
    for _type_name in numpy_float_names:
        try:
            _type = getattr(numpy,_type_name)
            numpy_float.append(_type)
        except:
            pass

# Complex
numpy_complex_names = []
numpy_complex_names.append('complex_')
numpy_complex_names.append('complex64')
numpy_complex_names.append('complex128')
numpy_complex = []
if _has_numpy:
    for _type_name in numpy_complex_names:
        try:
            _type = getattr(numpy,_type_name)
            numpy_complex.append(_type)
        except:
            pass


#
# Register NumPy Types
#

# Register NumPy boolean types as Boolean
for _type in numpy_bool:
    RegisterBooleanType(_type)

# Register NumPy integer types as Integer (this will also update
# Numeric/Reals) and as Boolean
for _type in numpy_int:
    RegisterIntegerType(_type)
    RegisterBooleanType(_type)

# Register NumPy float types as Numeric and as Boolean
for _type in numpy_float:
    RegisterNumericType(_type)
    RegisterBooleanType(_type)

# Note: If complex types are to be registered it will need to be
#       with a different registration function than
#       RegisterNumericType because this updates the type set used
#       by Reals.

#for _type in numpy_complex:
#    pass

