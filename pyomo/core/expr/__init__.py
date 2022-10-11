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

#
# The definition of __all__ is a bit funky here, because we want to
# expose symbols in pyomo.core.expr.current that are not included in
# pyomo.core.expr.  The idea is that pyomo.core.expr provides symbols
# that are used by general users, but pyomo.core.expr.current provides
# symbols that are used by developers.
# 

from . import (
    numvalue, visitor, numeric_expr, boolean_value, logical_expr,
    relational_expr, current,
)


# FIXME: we shouldn't need circular dependencies between modules
visitor.LinearExpression = numeric_expr.LinearExpression

# Initialize numvalue functions
numvalue._generate_sum_expression \
    = numeric_expr._generate_sum_expression
numvalue._generate_mul_expression \
    = numeric_expr._generate_mul_expression
numvalue._generate_other_expression \
    = numeric_expr._generate_other_expression

numvalue._generate_relational_expression \
    = relational_expr._generate_relational_expression

# Initialize logicalvalue functions
boolean_value._generate_logical_proposition \
    = logical_expr._generate_logical_proposition

from .numvalue import (
    value, is_constant, is_fixed, is_variable_type,
    is_potentially_variable, NumericValue, ZeroConstant,
    native_numeric_types, native_types, polynomial_degree,
)

from .boolean_value import BooleanValue

from .numeric_expr import linear_expression, nonlinear_expression
from .logical_expr import (
    land, lnot, lor, xor, equivalent, exactly, atleast, atmost, implies,
)
from .relational_expr import inequality
from .current import (
    log, log10, sin, cos, tan, cosh, sinh, tanh,
    asin, acos, atan, exp, sqrt, asinh, acosh,
    atanh, ceil, floor,
    Expr_if,
)

from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion
