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
from pyomo.repn import generate_canonical_repn
import pyomo.core.base.connector

from six import iteritems

def preprocess_block_objectives(block, var_id_map):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block, '_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    active_objectives = block.component_map(Objective, active=True)
    for key, obj in iteritems(active_objectives):

        for ondx, objective_data in iteritems(obj._data):
            if not objective_data.active:
                continue
            if objective_data.expr is None:
                raise ValueError("No expression has been defined for objective %s" % str(key))

            try:
                objective_data_repn = generate_canonical_repn(objective_data.expr, var_id_map)
            except Exception:
                err = sys.exc_info()[1]
                logging.getLogger('pyomo.core').error\
                    ( "exception generating a canonical representation for objective %s (index %s): %s" \
                          % (str(key), str(ondx), str(err)) )
                raise

            block_canonical_repn[objective_data] = objective_data_repn

def preprocess_constraint_index(block,
                                constraint_data,
                                var_id_map,
                                block_canonical_repn=None,
                                block_lin_body=None):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    if constraint_data.body is None:
        raise ValueError("No expression has been defined for "
                         "the body of constraint %s"
                         % (constraint_data.cname(True)))

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
        canonical_repn = generate_canonical_repn(constraint_data.body, var_id_map)
    except Exception:
        logging.getLogger('pyomo.core').error \
            ( "exception generating a canonical representation for constraint %s" \
              % (constraint_data.cname(True)))
        raise

    block_canonical_repn[constraint_data] = canonical_repn

def preprocess_constraint(block,
                          constraint,
                          var_id_map={},
                          block_canonical_repn=None,
                          block_lin_body=None):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    has_lin_body = False
    if block_lin_body is None:
        if hasattr(block,"lin_body"):
            block_lin_body = block.lin_body
            has_lin_body = True
    else:
        has_lin_body = True

    for index, constraint_data in iteritems(constraint._data):

        if not constraint_data.active:
            continue

        if has_lin_body is True:
            lin_body = block_lin_body.get(constraint_data)
            if lin_body is not None:
                # if we already have the linear encoding of the
                # constraint body, skip canonical expression
                # generation.
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
            canonical_repn = generate_canonical_repn(constraint_data.body, var_id_map)
        except Exception:
            logging.getLogger('pyomo.core').error \
                ( "exception generating a canonical representation for constraint %s (index %s)" \
                  % (str(constraint.name), str(index)) )
            raise

        block_canonical_repn[constraint_data] = canonical_repn

def preprocess_block_constraints(block, var_id_map):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block, '_canonical_repn'):
        block._canonical_repn = ComponentMap()
    block_canonical_repn = block._canonical_repn

    for constraint in block.component_objects(Constraint,
                                              active=True,
                                              descend_into=False):
        preprocess_constraint(block,
                              constraint,
                              var_id_map=var_id_map,
                              block_canonical_repn=block_canonical_repn,
                              block_lin_body=getattr(block,"lin_body",None))

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
    var_id_map = {}

    # FIXME: We should revisit the bilevel transformations to see why
    # the test requires "SubModels" to be preprocessed. [JDS 12/31/14]
    if model._type is not Block and model.active:
        preprocess_block_constraints(model, var_id_map)
        preprocess_block_objectives(model, var_id_map)

    # block_data_objects() returns the current block... no need to do special
    # handling of the top (model) block.
    #
    for block in model.block_data_objects(active=True):
        preprocess_block_constraints(block, var_id_map)
        preprocess_block_objectives(block, var_id_map)
