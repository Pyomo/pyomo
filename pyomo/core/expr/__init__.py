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
    numvalue,
    visitor,
    numeric_expr,
    boolean_value,
    logical_expr,
    relational_expr,
    current,
)

# FIXME: remove circular dependencies between numvalue and numeric_expr
#
# We unfortunately have circular dependencies between the numvalue
# module (which defines the base class for all numeric expression
# components, and implements the operator overloading methods) and the
# numeric_expr module (the dispatchers and the expression node
# definitions)
numvalue._add_dispatcher = numeric_expr._add_dispatcher
numvalue._neg_dispatcher = numeric_expr._neg_dispatcher
numvalue._mul_dispatcher = numeric_expr._mul_dispatcher
numvalue._div_dispatcher = numeric_expr._div_dispatcher
numvalue._abs_dispatcher = numeric_expr._abs_dispatcher
numvalue._pow_dispatcher = numeric_expr._pow_dispatcher

# Initialize numvalue functions
numvalue._generate_relational_expression = (
    relational_expr._generate_relational_expression
)

# Initialize logicalvalue functions
boolean_value._generate_logical_proposition = logical_expr._generate_logical_proposition

from .numvalue import (
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

from .boolean_value import BooleanValue

from .numeric_expr import linear_expression, nonlinear_expression, mutable_expression
from .logical_expr import (
    land,
    lnot,
    lor,
    xor,
    equivalent,
    exactly,
    atleast,
    atmost,
    implies,
)
from .relational_expr import inequality
from .current import (
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
)

from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.taylor_series import taylor_series_expansion
