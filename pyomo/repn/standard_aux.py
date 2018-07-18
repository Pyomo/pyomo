#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division

__all__ = ['compute_standard_repn']


import pyomo.common
from pyomo.repn.standard_repn import preprocess_block_constraints, preprocess_block_objectives


@pyomo.common.pyomo_api(namespace='pyomo.repn')
def compute_standard_repn(data, model=None):
    """
    This plugin computes the standard representation for all objectives
    and constraints. All results are stored in a ComponentMap named
    "_repn" at the block level.

    We break out preprocessing of the objectives and constraints
    in order to avoid redundant and unnecessary work, specifically
    in contexts where a model is iteratively solved and modified.
    we don't have finer-grained resolution, but we could easily
    pass in a Constraint and an Objective if warranted.

    Required:
        model:      A concrete model instance.
    """
    idMap = {}
    for block in model.block_data_objects(active=True):
        preprocess_block_constraints(block, idMap=idMap)
        preprocess_block_objectives(block, idMap=idMap)
