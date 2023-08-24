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

import enum
import math

from pyomo.common.deprecation import deprecation_warning

deprecation_warning(
    "pyomo.core.expr.current is deprecated.  "
    "Please import expression symbols from pyomo.core.expr",
    version='6.6.2',
)

#
# Common intrinsic functions
#
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.expr_common import clone_counter, _mode

from pyomo.core.expr import (
    Mode,
    # from pyomo.core.expr.base
    ExpressionBase,
    # pyomo.core.expr.visitor
    evaluate_expression,
    expression_to_string,
    polynomial_degree,
    clone_expression,
    sizeof_expression,
    # pyomo.core.expr.numeric_expr
    NumericExpression,
    NumericValue,
    native_types,
    nonpyomo_leaf_types,
    native_numeric_types,
    value,
    nonlinear_expression,
    linear_expression,
    NegationExpression,
    NPV_NegationExpression,
    ExternalFunctionExpression,
    NPV_ExternalFunctionExpression,
    PowExpression,
    NPV_PowExpression,
    ProductExpression,
    NPV_ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    NPV_DivisionExpression,
    SumExpressionBase,
    NPV_SumExpression,
    SumExpression,
    Expr_ifExpression,
    NPV_Expr_ifExpression,
    UnaryFunctionExpression,
    NPV_UnaryFunctionExpression,
    AbsExpression,
    NPV_AbsExpression,
    LinearExpression,
    decompose_term,
    LinearDecompositionError,
    NPV_expression_types,
    Expr_if,
    ceil,
    floor,
    exp,
    log,
    log10,
    sqrt,
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    asin,
    acos,
    atan,
    asinh,
    acosh,
    atanh,
    # pyomo.core.expr.numvalue
    as_numeric,
    # pyomo.core.expr.logical_expr
    native_logical_types,
    BooleanValue,
    BooleanConstant,
    BooleanExpressionBase,
    lnot,
    equivalent,
    xor,
    implies,
    land,
    lor,
    exactly,
    atmost,
    atleast,
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
    special_boolean_atom_types,
    # pyomo.core.expr.relational_expr
    RelationalExpression,
    RangedExpression,
    InequalityExpression,
    EqualityExpression,
    inequality,
    # pyomo.core.expr.template_expr
    TemplateExpressionError,
    GetItemExpression,
    Numeric_GetItemExpression,
    Boolean_GetItemExpression,
    Structural_GetItemExpression,
    NPV_Numeric_GetItemExpression,
    NPV_Boolean_GetItemExpression,
    NPV_Structural_GetItemExpression,
    GetAttrExpression,
    Numeric_GetAttrExpression,
    Boolean_GetAttrExpression,
    Structural_GetAttrExpression,
    NPV_Numeric_GetAttrExpression,
    NPV_Boolean_GetAttrExpression,
    NPV_Structural_GetAttrExpression,
    CallExpression,
    TemplateSumExpression,
    IndexTemplate,
    resolve_template,
    ReplaceTemplateExpression,
    substitute_template_expression,
    substitute_getitem_with_param,
    substitute_template_with_value,
    templatize_rule,
    templatize_constraint,
    # pyomo.core.expr.visitor
    SymbolMap,
    StreamBasedExpressionVisitor,
    SimpleExpressionVisitor,
    ExpressionValueVisitor,
    replace_expressions,
    ExpressionReplacementVisitor,
    FixedExpressionError,
    NonConstantExpressionError,
    identify_components,
    identify_variables,
    identify_mutable_parameters,
)
