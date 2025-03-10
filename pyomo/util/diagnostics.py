#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
        ctype=Disjunct,
        active=True,
        descend_into=(Block, Disjunct),
        descent_order=TraversalStrategy.PrefixDepthFirstSearch,
    ):
        logger.info(
            "%s %s%s"
            % (
                disj.name,
                disj.indicator_var.value,
                " fixed" if disj.indicator_var.fixed else "",
            )
        )
