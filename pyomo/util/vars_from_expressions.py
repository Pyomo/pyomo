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

"""
This module contains a function to generate a list of the Vars which appear
in expressions in the active tree. Note this is not the same as
component_data_objects(Var) because it does not look for Var objects which are
not used in any expressions and it does not care if the Vars it finds are
actually in the subtree or not.
"""
from pyomo.core import Block
import pyomo.core.expr as EXPR


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
    seen = set()
    for constraint in block.component_data_objects(
        ctype,
        active=active,
        sort=sort,
        descend_into=descend_into,
        descent_order=descent_order,
    ):
        for var in EXPR.identify_variables(
            constraint.expr, include_fixed=include_fixed
        ):
            if id(var) not in seen:
                seen.add(id(var))
                yield var
