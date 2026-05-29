# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""
This module contains functions to generate a list of the Vars appearing
in expressions in the active tree. Note this is not the same as
``component_data_objects(Var)`` because it does not look for Var objects which are
not used in any expressions and it does not care if the Vars it finds are
actually in the Block subtree or not.
"""

from pyomo.core import Block
from pyomo.core.expr.visitor import IdentifyVariableVisitor


def get_vars_from_components(
    block,
    ctype,
    include_fixed=True,
    active=None,
    sort=False,
    descend_into=Block,
    descent_order=None,
):
    """Returns a generator of all the Var objects which are used in
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
    seen = {}
    visitor = IdentifyVariableVisitor(include_fixed, {}, seen=seen)
    for constraint in block.component_data_objects(
        ctype,
        active=active,
        sort=sort,
        descend_into=descend_into,
        descent_order=descent_order,
    ):
        visitor.walk_expression(constraint.expr)
    return seen.values()


def get_vars(
    block,
    include_fixed=False,
    active=True,
    sort=False,
    descend_into=Block,
    descent_order=None,
):
    """Return all vars referenced through expressions in the specified block.

    This is a simple wrapper around :func:`get_vars_from_components()`
    that gathers all variables referenced by :class:`Constraint` and
    :class:`Objective` objects within the specified block.  Note that as
    it is designed to return the "variables used in the current model,"
    it uses different defaults for `active` and `include_fixed`.

    Parameters
    ----------
    include_fixed : bool
        If True, both fixed and free variables will be returned

    active : bool | None
        If True, only variables accessible through the active component
        tree will be erturned

    sort: SortOrder | None
        sort method for iterating through Constraint objects

    descend_into : None | type | tuple[type]
        Ctypes to descend into when finding Constraints

    descent_order : None | TraversalStrategy
        Traversal strategy for walking the block hierarchy

    """
    return get_vars_from_components(
        block,
        ctype=(Constraint, Objective),
        include_fixed=include_fixed,
        active=active,
        sort=sort,
        descend_into=descend_into,
        descent_order=descent_order,
    )
