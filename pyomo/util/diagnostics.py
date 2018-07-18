# -*- coding: UTF-8 -*-
"""Module with miscellaneous diagnostic tools"""
from pyomo.core.base.block import TraversalStrategy, Block
from pyomo.gdp import Disjunct
import logging


logger = logging.getLogger('pyomo.util.diagnostics')
logger.setLevel(logging.INFO)


def log_disjunct_values(m):
    """Prints the values of the disjunct indicator variables."""
    for disj in m.component_data_objects(
        ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
        descent_order=TraversalStrategy.PrefixDepthFirstSearch
    ):
        logger.info("%s %s%s" % (disj.name, disj.indicator_var.value,
                                 " fixed" if disj.indicator_var.fixed else ""))
