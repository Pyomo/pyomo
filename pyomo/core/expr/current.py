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

#
# Common intrinsic functions
#
from pyomo.core.expr import expr_common as common


#
# Provide a global value that indicates which expression system is being used
#
class Mode(enum.IntEnum):
    # coopr: Original Coopr/Pyomo expression system
    coopr_trees = 1
    # coopr3: leverage reference counts to reduce the amount of required
    # expression cloning to ensure independent expression trees.
    coopr3_trees = 3
    # pyomo4: rework the expression system to remove reliance on
    # reference counting.  This enables pypy support (which doesn't have
    # reference counting).  This version never became the default.
    pyomo4_trees = 4
    # pyomo5: refinement of pyomo4.  Expressions are now immutable by
    # contract, which tolerates "entangled" expression trees.  Added
    # specialized classes for NPV expressions and LinearExpressions.
    pyomo5_trees = 5
    # pyomo6: refinement of pyomo5 expression generation to leverage
    # multiple dispatch.  Standardized expression storage and argument
    # handling (significant rework of the LinearExpression structure).
    pyomo6_trees = 6


_mode = Mode.pyomo6_trees
# We no longer support concurrent expression systems.  _mode is left
# primarily so we can support expression system-specific baselines
assert _mode == Mode.pyomo6_trees

#
# Pull symbols from the appropriate expression system
#
from pyomo.core.expr import numvalue as _numvalue
from pyomo.core.expr import boolean_value as _logicalvalue
from pyomo.core.expr import numeric_expr as _numeric_expr

from pyomo.core.expr.expr_common import clone_counter
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.visitor import (
    evaluate_expression,
    expression_to_string,
    polynomial_degree,
    clone_expression,
    sizeof_expression,
)
from pyomo.core.expr.numeric_expr import (
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
)
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr import logical_expr as _logical_expr
from pyomo.core.expr.logical_expr import (
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
)
from pyomo.core.expr.relational_expr import (
    RelationalExpression,
    RangedExpression,
    InequalityExpression,
    EqualityExpression,
    inequality,
)
from pyomo.core.expr.template_expr import (
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
)
from pyomo.core.expr import visitor as _visitor
from pyomo.core.expr.visitor import (
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
