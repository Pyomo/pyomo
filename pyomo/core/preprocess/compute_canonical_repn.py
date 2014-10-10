#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

import sys
import logging
import itertools

from six import iteritems

from pyomo.core.base import Constraint, Objective, ComponentMap, active_components
from pyomo.core.base import IPyomoPresolver, IPyomoPresolveAction
import pyomo.core.expr
from pyomo.core.expr import generate_canonical_repn
import pyomo.core.base.connector 


def preprocess_block_objectives(block, var_id_map):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'canonical_repn'):
        block.canonical_repn = ComponentMap()
    block_canonical_repn = block.canonical_repn

    if getattr(block,'skip_canonical_repn',False):
        return block
    
    active_objectives = block.active_components(Objective)
    for key, obj in iteritems(active_objectives):
        # number of objective indicies with non-trivial expressions
        num_nontrivial = 0

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
            if not pyomo.core.expr.canonical_is_constant(objective_data_repn):
                num_nontrivial += 1
            
            block_canonical_repn[objective_data] = objective_data_repn
            
        if num_nontrivial == 0:
            obj.trivial = True
        else:
            obj.trivial = False

def preprocess_constraint_index(block,
                                constraint_data,
                                var_id_map,
                                block_canonical_repn=None,
                                block_lin_body=None):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'canonical_repn'):
        block.canonical_repn = ComponentMap()
    block_canonical_repn = block.canonical_repn

    if constraint_data.body is None:
        raise ValueError("No expression has been defined for the body of constraint %s, index=%s" % (str(constraint._parent().name), str(constraint.index())))

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
            ( "exception generating a canonical representation for constraint %s (index %s)" \
              % (str(constraint.name), str(index)) )
        raise

    block_canonical_repn[constraint_data] = canonical_repn
        
def preprocess_constraint(block,
                          constraint,
                          var_id_map={},
                          block_canonical_repn=None,
                          block_lin_body=None):

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'canonical_repn'):
        block.canonical_repn = ComponentMap()
    block_canonical_repn = block.canonical_repn

    has_lin_body = False
    if block_lin_body is None:
        if hasattr(block,"lin_body"):
            block_lin_body = block.lin_body
            has_lin_body = True
    else:
        has_lin_body = True

    # number of constraint indicies with non-trivial bodies
    num_nontrivial = 0

    for index, constraint_data in iteritems(constraint._data):

        if not constraint_data.active:
            continue

        if has_lin_body is True:
            lin_body = block_lin_body.get(constraint_data)
            if lin_body is not None:
                # if we already have the linear encoding of the constraint body, skip canonical expression
                # generation. but we still need to assess constraint triviality.
                if not pyomo.core.expr.is_linear_expression_constant(lin_body):
                    num_nontrivial += 1
                continue

        if constraint_data.body is None:
            raise ValueError("No expression has been defined for the body of constraint %s, index=%s" % (str(constraint.name), str(index)))

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
            continue

        try:
            canonical_repn = generate_canonical_repn(constraint_data.body, var_id_map)
        except Exception:
            logging.getLogger('pyomo.core').error \
                ( "exception generating a canonical representation for constraint %s (index %s)" \
                  % (str(constraint.name), str(index)) )
            raise
        
        block_canonical_repn[constraint_data] = canonical_repn
        
        if not pyomo.core.expr.canonical_is_constant(canonical_repn):
            num_nontrivial += 1

    if num_nontrivial == 0:
        constraint.trivial = True
    else:
        constraint.trivial = False

def preprocess_block_constraints(block, var_id_map):

    if getattr(block,'skip_canonical_repn',False):
        return

    # Get/Create the ComponentMap for the canonical_repn
    if not hasattr(block,'canonical_repn'):
        block.canonical_repn = ComponentMap()
    block_canonical_repn = block.canonical_repn

    for constraint in active_components(block,Constraint):

        preprocess_constraint(block,
                              constraint,
                              var_id_map=var_id_map,
                              block_canonical_repn=block_canonical_repn,
                              block_lin_body=getattr(block,"lin_body",None))

@pyomo.misc.pyomo_api(namespace='pyomo.model')
def compute_canonical_repn(data, model=None):
    """
    This plugin computes the canonical representation for all
    objectives and constraints linear terms.  All results are stored
    in a ComponentMap named "canonical_repn" at the block level.

    NOTE: The idea of this module should be generaized. there are
    two key functionalities: computing a version of the expression
    tree in preparation for output and identification of trivial
    constraints.

    NOTE: this leaves the trivial constraints in the model.

    We break out preprocessing of the objectives and constraints
    in order to avoid redundant and unnecessary work, specifically
    in contexts where a model is iteratively solved and modified.
    we don't have finer-grained resolution, but we could easily
    pass in a Constraint and an Objective if warranted.

    Required:
        model:      A concrete model instance.
    """
    var_id_map = {}
    for block in model.all_blocks():
        preprocess_block_constraints(block, var_id_map)
        preprocess_block_objectives(block, var_id_map)
