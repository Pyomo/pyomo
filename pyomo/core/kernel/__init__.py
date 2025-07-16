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

from pyomo.core.expr import numvalue, numeric_expr, boolean_value, logical_expr
from pyomo.core.expr.numvalue import (
    value,
    is_constant,
    is_fixed,
    is_variable_type,
    is_potentially_variable,
    NumericValue,
    ZeroConstant,
    native_numeric_types,
    native_types,
    polynomial_degree,
)
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr import (
    log,
    log10,
    sin,
    cos,
    tan,
    cosh,
    sinh,
    tanh,
    asin,
    acos,
    atan,
    exp,
    sqrt,
    asinh,
    acosh,
    atanh,
    ceil,
    floor,
    Expr_if,
    inequality,
    linear_expression,
    nonlinear_expression,
    land,
    lor,
    equivalent,
    exactly,
    atleast,
    atmost,
    implies,
    lnot,
    xor,
)
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion

from pyomo.core.kernel import (
    base,
    homogeneous_container,
    heterogeneous_container,
    variable,
    constraint,
    matrix_constraint,
    parameter,
    expression,
    objective,
    sos,
    suffix,
    block,
    piecewise_library,
    set_types,
)


#
# declare deprecation paths for removed modules and attributes
#
from pyomo.common.deprecation import relocated_module_attribute, moved_module

relocated_module_attribute(
    'component_map',
    'pyomo.common.collections.component_map',
    msg='The pyomo.core.kernel.component_map module is deprecated.  '
    'Import ComponentMap from pyomo.common.collections.',
    version='5.7.1',
    f_globals=globals(),
)
relocated_module_attribute(
    'component_set',
    'pyomo.common.collections.component_set',
    msg='The pyomo.core.kernel.component_map module is deprecated.  '
    'Import ComponentMap from pyomo.common.collections.',
    version='5.7.1',
    f_globals=globals(),
)

moved_module(
    "pyomo.core.kernel.component_map",
    "pyomo._archive.component_map",
    msg='The pyomo.core.kernel.component_map module is deprecated.  '
    'Import ComponentMap from pyomo.common.collections.',
    version='5.7.1',
)
moved_module(
    "pyomo.core.kernel.component_set",
    "pyomo._archive.component_set",
    msg='The pyomo.core.kernel.component_set module is deprecated.  '
    'Import ComponentSet from pyomo.common.collections.',
    version='5.7.1',
)
moved_module(
    "pyomo.core.kernel.register_numpy_types",
    "pyomo._archive.register_numpy_types",
    msg="pyomo.core.kernel.register_numpy_types is deprecated.  NumPy type "
    "registration is handled automatically by pyomo.common.dependencies.numpy",
    version='6.1',
)
del relocated_module_attribute, moved_module
