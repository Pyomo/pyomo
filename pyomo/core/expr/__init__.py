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

from . import (
    numvalue,
    visitor,
    numeric_expr,
    boolean_value,
    logical_expr,
    relational_expr,
)

#
# FIXME: remove circular dependencies between logical_expr and numeric_expr
#

# Initialize logicalvalue functions
boolean_value._generate_logical_proposition = logical_expr._generate_logical_proposition


from pyomo.common.numeric_types import (
    value,
    native_numeric_types,
    native_types,
    nonpyomo_leaf_types,
)
from pyomo.common.errors import TemplateExpressionError

from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import ExpressionType, Mode, OperatorAssociativity
from .logical_expr import (
    native_logical_types,
    special_boolean_atom_types,
    #
    BooleanValue,
    BooleanConstant,
    BooleanExpression,
    BooleanExpressionBase,
    #
    UnaryBooleanExpression,
    NotExpression,
    BinaryBooleanExpression,
    EquivalenceExpression,
    XorExpression,
    ImplicationExpression,
    NaryBooleanExpression,
    AndExpression,
    OrExpression,
    ExactlyExpression,
    AtMostExpression,
    AtLeastExpression,
    AllDifferentExpression,
    CountIfExpression,
    #
    land,
    lnot,
    lor,
    xor,
    equivalent,
    exactly,
    atleast,
    atmost,
    all_different,
    count_if,
    implies,
)
from .numeric_expr import (
    NumericValue,
    NumericExpression,
    # operators:
    AbsExpression,
    DivisionExpression,
    Expr_ifExpression,
    ExternalFunctionExpression,
    LinearExpression,
    MaxExpression,
    MinExpression,
    MonomialTermExpression,
    NegationExpression,
    PowExpression,
    ProductExpression,
    SumExpressionBase,  # TODO: deprecate / remove
    SumExpression,
    UnaryFunctionExpression,
    # TBD: remove export of NPV classes here?
    NPV_AbsExpression,
    NPV_DivisionExpression,
    NPV_Expr_ifExpression,
    NPV_ExternalFunctionExpression,
    NPV_MaxExpression,
    NPV_MinExpression,
    NPV_NegationExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
    # functions to generate expressions
    Expr_if,
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
    # Lgacy utilities
    NPV_expression_types,  # TODO: remove
    LinearDecompositionError,  # TODO: move to common.errors
    decompose_term,
    linear_expression,
    nonlinear_expression,
    mutable_expression,
)
from .numvalue import (
    as_numeric,
    is_constant,
    is_fixed,
    is_variable_type,
    is_potentially_variable,
    ZeroConstant,
    polynomial_degree,
)
from .relational_expr import (
    RelationalExpression,
    RangedExpression,
    InequalityExpression,
    EqualityExpression,
    NotEqualExpression,
    inequality,
)
from .symbol_map import SymbolMap
from .template_expr import (
    GetItemExpression,
    Numeric_GetItemExpression,
    Boolean_GetItemExpression,
    Structural_GetItemExpression,
    GetAttrExpression,
    Numeric_GetAttrExpression,
    Boolean_GetAttrExpression,
    Structural_GetAttrExpression,
    CallExpression,
    TemplateSumExpression,
    #
    NPV_Numeric_GetItemExpression,
    NPV_Boolean_GetItemExpression,
    NPV_Structural_GetItemExpression,
    NPV_Numeric_GetAttrExpression,
    NPV_Boolean_GetAttrExpression,
    NPV_Structural_GetAttrExpression,
    #
    IndexTemplate,
    resolve_template,
    ReplaceTemplateExpression,
    substitute_template_expression,
    substitute_getitem_with_param,
    substitute_template_with_value,
    templatize_rule,
    templatize_constraint,
)
from .visitor import (
    StreamBasedExpressionVisitor,
    SimpleExpressionVisitor,
    ExpressionValueVisitor,
    ExpressionReplacementVisitor,
    FixedExpressionError,
    NonConstantExpressionError,
    identify_components,
    identify_variables,
    identify_mutable_parameters,
    clone_expression,
    evaluate_expression,
    expression_to_string,
    polynomial_degree,
    replace_expressions,
    sizeof_expression,
)

from .calculus.derivatives import differentiate
from .taylor_series import taylor_series_expansion

#
# declare deprecation paths for removed modules and attributes
#
from pyomo.common.deprecation import moved_module

moved_module(
    "pyomo.core.expr.current",
    "pyomo._archive.current",
    msg="pyomo.core.expr.current is deprecated.  "
    "Please import expression symbols from pyomo.core.expr",
    version='6.6.2',
)
del moved_module
