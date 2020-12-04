#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr import numvalue, numeric_expr, boolean_value, logical_expr, current
from pyomo.core.expr.numvalue import (
    value, is_constant, is_fixed, is_variable_type,
    is_potentially_variable, NumericValue, ZeroConstant,
    native_numeric_types, native_types, polynomial_degree,
)
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr.numeric_expr import linear_expression, nonlinear_expression
from pyomo.core.expr.logical_expr import (land, lor, equivalent, exactly,
                                          atleast, atmost, implies, lnot,
                                          xor, inequality)
from pyomo.core.expr.current import (
    log, log10, sin, cos, tan, cosh, sinh, tanh,
    asin, acos, atan, exp, sqrt, asinh, acosh,
    atanh, ceil, floor,
    Expr_if,
)
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion

import pyomo.core.kernel.register_numpy_types
import pyomo.core.kernel.base
import pyomo.core.kernel.homogeneous_container
import pyomo.core.kernel.heterogeneous_container
import pyomo.core.kernel.variable
import pyomo.core.kernel.constraint
import pyomo.core.kernel.matrix_constraint
import pyomo.core.kernel.parameter
import pyomo.core.kernel.expression
import pyomo.core.kernel.objective
import pyomo.core.kernel.sos
import pyomo.core.kernel.suffix
import pyomo.core.kernel.block
import pyomo.core.kernel.piecewise_library
import pyomo.core.kernel.set_types

# TODO: These are included for backwards compatibility.  Accessing them
# will result in a deprecation warning
from pyomo.common.dependencies import attempt_import
component_map = attempt_import('pyomo.core.kernel.component_map')[0]
component_set = attempt_import('pyomo.core.kernel.component_set')[0]
