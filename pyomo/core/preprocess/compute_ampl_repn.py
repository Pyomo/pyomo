#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import sys
import logging

import pyomo.core
from pyomo.core.base import Constraint, \
                             Objective, \
                             ComponentMap, \
                             active_components_data
from pyomo.core.expr import generate_ampl_repn


def preprocess_block_objectives(block):

    # Get/Create the ComponentMap for the repn
    if not hasattr(block,'_ampl_repn'):
        block._ampl_repn = ComponentMap()
    block_ampl_repn = block._ampl_repn

    for objective_data in active_components_data(block, Objective): #recursive = False

        if objective_data.expr is None:
            raise ValueError("No expression has been defined for objective %s" % str(key))

        try:
            ampl_repn = generate_ampl_repn(objective_data.expr)
        except Exception:
            err = sys.exc_info()[1]
            logging.getLogger('pyomo.core').error\
                ( "exception generating a ampl representation for objective %s: %s" \
                      % (objective_data.cname(True), str(err)) )
            raise

        block_ampl_repn[objective_data] = ampl_repn

def preprocess_block_constraints(block):

    # Get/Create the ComponentMap for the repn
    if not hasattr(block,'_ampl_repn'):
        block._ampl_repn = ComponentMap()
    block_ampl_repn = block._ampl_repn

    for constraint_data in active_components_data(block, Constraint): #recursive = False

        if constraint_data.body is None:
            raise ValueError("No expression has been defined for the body of constraint %s, index=%s" % (str(constraint.name), str(index)))

        try:
            ampl_repn = generate_ampl_repn(constraint_data.body)
        except Exception:
            err = sys.exc_info()[1]
            logging.getLogger('pyomo.core').error\
                ( "exception generating a ampl representation for constraint %s: %s" \
                      % (constraint_data.cname(True), str(err)) )
            raise

        block_ampl_repn[constraint_data] = ampl_repn

@pyomo.misc.pyomo_api(namespace='pyomo.model')
def compute_ampl_repn(data, model=None):
    """
    This plugin computes the ampl representation for all objectives
    and constraints. All results are stored in a ComponentMap named
    "_ampl_repn" at the block level.

    NOTE: this does not check for trivial constraints

    We break out preprocessing of the objectives and constraints
    in order to avoid redundant and unnecessary work, specifically
    in contexts where a model is iteratively solved and modified.
    we don't have finer-grained resolution, but we could easily
    pass in a Constraint and an Objective if warranted.

    Required:
        model:      A concrete model instance.
    """
    for block in model.all_blocks():
        preprocess_block_constraints(block)
        preprocess_block_objectives(block)

