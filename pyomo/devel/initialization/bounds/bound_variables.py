# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import math

from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.core.base.block import BlockData
from pyomo.util.vars_from_expressions import get_vars

logger = logging.getLogger(__name__)


def bound_all_nonlinear_variables(m: BlockData, default_bound: float = 1.0e8):
    """
    Attempt to obtain valid bounds on all nonlinear variables based on the
    constraints in the model, m. If variable bounds cannot be obtained,
    we use default_bound.
    """
    fbbt(m)
    for v in get_vars(m):
        # If bounds are equal, treat first
        if v.lb == v.ub:
            if v.lb is None:
                v.setlb(-default_bound)
                v.setub(default_bound)
        # If bound is none or outside default bound, set to default
        else:
            lb = v.lb
            ub = v.ub
            if lb is None:
                lb = -math.inf
            if ub is None:
                ub = math.inf
            if lb < -default_bound:
                logger.debug(
                    f'Could not obtain a lower bound for {str(v)} better than {-default_bound}; setting the lower bound to {-default_bound}'
                )
                v.setlb(min(-default_bound, ub))

            if ub > default_bound:
                logger.debug(
                    f'Could not obtain an upper bound for {str(v)} better than {default_bound}; setting the upper bound to {default_bound}'
                )
                v.setub(max(default_bound, lb))
            if v.lb > v.ub + 1e-6 + 1e-6 * v.ub:
                raise RuntimeError(
                    f'Unexpectedly set invalid bounds for {str(v)}: ({v.lb}, {v.ub})'
                )
    fbbt(m)
    # Slightly shift the bounds to prevent math domain error in log or exp functions
    for v in get_vars(m):
        d = v.ub - v.lb
        d *= 1e-6
        d = min(d, 1e-6)
        v.setlb(v.lb + d)
        v.setub(v.ub - d)
