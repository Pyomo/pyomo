#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
This module contains a function to generate a list of the Vars which appear
in expressions in the active tree. Note this is not the same as
component_data_objects(Var) because it does not look for Var objects which are
not used in any expressions and it does not care if the Vars if finds are
actually in the subtree or not.
"""
from pyomo.core import Block, Constraint
from pyomo.core.expr import current as EXPR

def get_vars_from_constraints(block, include_fixed=True, active=None,
                              sort=False, descend_into=Block,
                              descent_order=None):
    """Returns a generator of all the Var objects which are used in Constraint
    expressions on the block. By default, this recurses into sub-blocks.

    Args:
        include_fixed: Whether or not to include fixed variables
        active: Whether to find Vars that appear in Constraints accessible 
                via the active tree
        sort: sort method for iterating through Constraint objects
        descend_into: Ctypes to descend into when finding Constraints
        descent_order: TODO: I have no idea?
    """
    for constraint in block.component_data_objects(Constraint,
                                                   active=active,
                                                   sort=sort,
                                                   descend_into=descend_into,
                                                   descent_order=descent_order):
        for var in EXPR.identify_variables(constraint.expr,
                                           include_fixed=include_fixed):
            yield var
