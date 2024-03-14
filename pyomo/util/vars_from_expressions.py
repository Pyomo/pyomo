#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
This module contains a function to generate a list of the Vars which appear
in expressions in the active tree. Note this is not the same as
component_data_objects(Var) because it does not look for Var objects which are
not used in any expressions and it does not care if the Vars it finds are
actually in the subtree or not.
"""
from pyomo.core import Block
from pyomo.core.expr.visitor import _StreamVariableVisitor
from pyomo.core.expr import identify_variables


def get_vars_from_components(
    block,
    ctype,
    include_fixed=True,
    active=None,
    sort=False,
    descend_into=Block,
    descent_order=None,
):
    """Returns a generator of all the Var objects which are used in Constraint
    expressions on the block. By default, this recurses into sub-blocks.

    Args:
        ctype: The type of component from which to get Vars, assumed to have
               an expr attribute.
        include_fixed: Whether or not to include fixed variables
        active: Whether to find Vars that appear in Constraints accessible
                via the active tree
        sort: sort method for iterating through Constraint objects
        descend_into: Ctypes to descend into when finding Constraints
        descent_order: Traversal strategy for finding the objects of type ctype
    """
    #visitor = _StreamVariableVisitor(
    #    include_fixed=include_fixed, descend_into_named_expressions=False
    #)
    #variables = []
    #for constraint in block.component_data_objects(
    #    ctype,
    #    active=active,
    #    sort=sort,
    #    descend_into=descend_into,
    #    descent_order=descent_order,
    #):
    #    variables.extend(visitor.walk_expression(constraint.expr))
    #    seen_named_exprs = set()
    #    named_expr_stack = list(visitor.named_expressions)
    #    while named_expr_stack:
    #        expr = named_expr_stack.pop()
    #        # Clear visitor's named expression cache so we only identify new
    #        # named expressions
    #        visitor.named_expressions.clear()
    #        variables.extend(visitor.walk_expression(expr.expr))
    #        for new_expr in visitor.named_expressions:
    #            if id(new_expr) not in seen_named_exprs:
    #                seen_named_exprs.add(id(new_expr))
    #                named_expr_stack.append(new_expr)
    #seen = set()
    #for var in variables:
    #    if id(var) not in seen:
    #        seen.add(id(var))
    #        yield var

    seen = set()
    named_expression_cache = {}
    for constraint in block.component_data_objects(
        ctype,
        active=active,
        sort=sort,
        descend_into=descend_into,
        descent_order=descent_order,
    ):
        for var in identify_variables(
            constraint.expr,
            include_fixed=include_fixed,
            named_expression_cache=named_expression_cache,
        ):
            if id(var) not in seen:
                seen.add(id(var))
                yield var
