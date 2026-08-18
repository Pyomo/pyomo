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

from pyomo.common.collections import ComponentSet
from pyomo.core import Block, Constraint, Objective
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
    """Returns a ComponentSet of all the Var objects that appear in
    expressions on the block. By default, this recurses into sub-blocks.

    Parameters
    ----------
    include_fixed : bool
        If True, both fixed and free variables will be returned

    ctype : type | tuple[type]
        The "ctype" of component from which to get Vars.  The components
        must expose a ``expr`` attribute that will be walked looking for
        variables.

    active : bool | None
        If True, only variables accessible through the active component
        tree will be returned.  If None, all variables accessible
        through either active or inactive components will be returned.

    sort: SortComponents | bool | None
        sort method for iterating through Constraint objects

    descend_into : type | tuple[type] | None
        "ctypes" to descend into when finding Constraints

    descent_order : TraversalStrategy | None
        Traversal strategy for walking the block hierarchy

    Returns
    -------
    ComponentSet : set of variables

    """
    var_cache = ComponentSet()
    visitor = IdentifyVariableVisitor(include_fixed, {}, var_cache=var_cache)
    for component in block.component_data_objects(
        ctype,
        active=active,
        sort=sort,
        descend_into=descend_into,
        descent_order=descent_order,
    ):
        visitor.walk_expression(component.expr)
    return var_cache


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
        tree will be returned.  If None, all variables accessible
        through either active or inactive components will be returned.

    sort: SortComponents | bool | None
        sort method for iterating through Constraint objects

    descend_into : type | tuple[type] | None
        "ctypes" to descend into when finding Constraints

    descent_order : TraversalStrategy | None
        Traversal strategy for walking the block hierarchy

    Returns
    -------
    ComponentSet : set of variables

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
