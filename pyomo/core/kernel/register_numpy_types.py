#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    "pyomo.core.kernel.register_numpy_types is deprecated.  NumPy type "
    "registration is handled automatically by pyomo.common.dependencies.numpy",
    version='6.1',
)

from pyomo.core.expr.numvalue import (
    RegisterNumericType, RegisterIntegerType, RegisterBooleanType,
    native_numeric_types, native_integer_types, native_boolean_types,
)

from pyomo.common.dependencies import numpy, numpy_available as _has_numpy

# Ensure that the types were registered
bool(_has_numpy)

numpy_int_names = []
numpy_int = []
numpy_float_names = []
numpy_float = []
numpy_bool_names = []
numpy_bool = []

if _has_numpy:
    # Historically, the lists included several numpy aliases
    numpy_int_names.extend(('int_', 'intc', 'intp'))
    numpy_int.extend((numpy.int_, numpy.intc, numpy.intp))
    numpy_float_names.append('float_')
    numpy_float.append(numpy.float_)

# Re-build the old numpy_* lists
for t in native_boolean_types:
    if t.__module__ == 'numpy':
        if t in native_integer_types:
            if t.__name__ not in numpy_int_names:
                numpy_int.append(t)
                numpy_int_names.append(t.__name__)
        elif t in native_numeric_types:
            if t.__name__ not in numpy_float_names:
                numpy_float.append(t)
                numpy_float_names.append(t.__name__)
        else:
            if t.__name__ not in numpy_bool_names:
                numpy_bool.append(t)
                numpy_bool_names.append(t.__name__)


# Complex
numpy_complex_names = []
numpy_complex = []
if _has_numpy:
    numpy_complex_names.extend(('complex_', 'complex64', 'complex128'))
    for _type_name in numpy_complex_names:
        try:
            _type = getattr(numpy, _type_name)
            numpy_complex.append(_type)
        except:     #pragma:nocover
            pass


