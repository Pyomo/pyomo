#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# the purpose of this file is to collect all utility methods that compute
# attributes of blocks, based on their contents.

__all__ = ['has_discrete_variables']

import logging

from pyomo.core import Var, Constraint, TraversalStrategy

logger = logging.getLogger(__name__)


def has_discrete_variables(block):
    for vardata in block.component_data_objects(Var, active=True):
        if not vardata.is_continuous():
            return True
    return False


def log_model_constraints(m, logger=logger, active=True):
    """Prints the model constraints in the model."""
    for constr in m.component_data_objects(
            ctype=Constraint, active=active, descend_into=True,
            descent_order=TraversalStrategy.PrefixDepthFirstSearch):
        logger.info("%s %s" % (
            constr.name,
            ("active" if constr.active else "deactivated")
        ))
