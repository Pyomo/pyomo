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

"This is the deprecated pyomo.core.kernel.register_numpy_types module"

from pyomo.common.numeric_types import (
    RegisterNumericType,
    RegisterIntegerType,
    RegisterBooleanType,
    native_complex_types,
    native_numeric_types,
    native_integer_types,
    native_boolean_types,
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
numpy_complex_names = []
numpy_complex = []

if _has_numpy:
    # Historically, the lists included several numpy aliases
    numpy_int_names.extend(('int_', 'intc', 'intp'))
    numpy_int.extend((numpy.int_, numpy.intc, numpy.intp))
    if hasattr(numpy, 'float_'):
        numpy_float_names.append('float_')
        numpy_float.append(numpy.float_)
    if hasattr(numpy, 'complex_'):
        numpy_complex_names.append('complex_')
        numpy_complex.append(numpy.complex_)

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
for t in native_complex_types:
    if t.__module__ == 'numpy':
        if t.__name__ not in numpy_complex_names:
            numpy_complex.append(t)
            numpy_complex_names.append(t.__name__)
