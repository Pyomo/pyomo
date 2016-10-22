#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import logging

from pyomo.core.base import Constraint, Objective, ComponentMap, Block
import pyomo.repn
from pyomo.repn.canonical_repn import LinearCanonicalRepn
from pyomo.repn import generate_canonical_repn
import pyomo.core.base.connector

from six import iteritems

def preprocess_block_objectives(block, idMap=None):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block, '_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    for objective_data in block.component_data_objects(Objective,
                                                       active=True,
                                                       descend_into=False):

        if objective_data.expr is None:
            raise ValueError("No expression has been defined for objective %s"
                             % (objective_data.name))

        try:
            objective_data_repn = generate_canonical_repn(objective_data.expr, idMap=idMap)
        except Exception:
            err = sys.exc_info()[1]
            logging.getLogger('pyomo.core').error(
                "exception generating a canonical representation for objective %s: %s"
                % (objective_data.name, str(err)))
            raise

        block_canonical_repn[objective_data] = objective_data_repn

def preprocess_block_constraints(block, idMap=None):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block, '_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    for constraint in block.component_objects(Constraint,
                                              active=True,
                                              descend_into=False):

        preprocess_constraint(block,
                              constraint,
                              idMap=idMap,
                              block_canonical_repn=block_canonical_repn)

def preprocess_constraint(block,
                          constraint,
                          idMap=None,
                          block_canonical_repn=None):

    from pyomo.repn.beta.matrix import MatrixConstraint
    if isinstance(constraint, MatrixConstraint):
        return

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    for index, constraint_data in iteritems(constraint):

        if not constraint_data.active:
            continue

        if isinstance(constraint_data, LinearCanonicalRepn):
            continue

        if constraint_data.body is None:
            raise ValueError("No expression has been defined for "
                             "the body of constraint %s, index=%s"
                             % (str(constraint.name), str(index)))

        # FIXME: This is a huge hack to keep canonical_repn from
        #        trying to generate representations representations of
        #        Constraints with Connectors (which will be
        #        deactivated once they have been expanded
        #        anyways). This can go away when preprocess is moved
        #        out of the model.create() phase and into the future
        #        model validation phase. (ZBF)
        ignore_connector = False
        if hasattr(constraint_data.body,"_args") and constraint_data.body._args is not None:
            for arg in constraint_data.body._args:
                if arg.__class__ is pyomo.core.base.connector.SimpleConnector:
                    ignore_connector = True
        if ignore_connector:
            #print "Ignoring",constraint.name,index
            continue

        try:
            canonical_repn = generate_canonical_repn(constraint_data.body, idMap=idMap)
        except Exception:
            logging.getLogger('pyomo.core').error \
                ( "exception generating a canonical representation for constraint %s (index %s)" \
                  % (str(constraint.name), str(index)) )
            raise

        block_canonical_repn[constraint_data] = canonical_repn

def preprocess_constraint_data(block,
                               constraint_data,
                               idMap=None,
                               block_canonical_repn=None):

    if isinstance(constraint_data, LinearCanonicalRepn):
        return

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    if constraint_data.body is None:
        raise ValueError("No expression has been defined for "
                         "the body of constraint %s"
                         % (constraint_data.name))

    # FIXME: This is a huge hack to keep canonical_repn from trying to generate representations
    #        representations of Constraints with Connectors (which will be deactivated once they
    #        have been expanded anyways). This can go away when preprocess is moved out of the
    #        model.create() phase and into the future model validation phase. (ZBF)
    ignore_connector = False
    if hasattr(constraint_data.body,"_args") and constraint_data.body._args is not None:
        for arg in constraint_data.body._args:
            if arg.__class__ is pyomo.core.base.connector.SimpleConnector:
                ignore_connector = True
    if ignore_connector:
        #print "Ignoring",constraint.name,index
        return

    try:
        canonical_repn = generate_canonical_repn(constraint_data.body, idMap=idMap)
    except Exception:
        logging.getLogger('pyomo.core').error \
            ( "exception generating a canonical representation for constraint %s" \
              % (constraint_data.name))
        raise

    block_canonical_repn[constraint_data] = canonical_repn

@pyomo.util.pyomo_api(namespace='pyomo.repn')
def compute_canonical_repn(data, model=None):
    """
    This plugin computes the canonical representation for all
    objectives and constraints linear terms.  All results are stored
    in a ComponentMap named "_canonical_repn" at the block level.

    We break out preprocessing of the objectives and constraints
    in order to avoid redundant and unnecessary work, specifically
    in contexts where a model is iteratively solved and modified.
    we don't have finer-grained resolution, but we could easily
    pass in a Constraint and an Objective if warranted.

    Required:
        model:      A concrete model instance.
    """
    idMap = {}

    # FIXME: We should revisit the bilevel transformations to see why
    # the test requires "SubModels" to be preprocessed. [JDS 12/31/14]
    if model._type is not Block and model.active:
        preprocess_block_constraints(model, idMap=idMap)
        preprocess_block_objectives(model, idMap=idMap)

    # block_data_objects() returns the current block... no need to do special
    # handling of the top (model) block.
    #
    for block in model.block_data_objects(active=True):
        preprocess_block_constraints(block, idMap=idMap)
        preprocess_block_objectives(block, idMap=idMap)
