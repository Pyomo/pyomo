# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.core.base.block import BlockData
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.devel.initialization.utils import get_vars
import logging

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
            if v.lb == None:
                v.setlb(-default_bound)
                v.setub(default_bound)
            if abs(v.ub) < abs(default_bound):
                continue
            else:
                v.setlb(-default_bound)
                v.setub(default_bound)
        # If bound is none or outside default bound, set to default
        else:
            if v.lb is None or v.lb < -default_bound:
                logger.debug(
                    f'Could not obtain a lower bound for {str(v)} better than {-default_bound}; setting the lower bound to {-default_bound}'
                )
                v.setlb(-default_bound)

            if v.ub is None or v.ub > default_bound:
                logger.debug(
                    f'Could not obtain an upper bound for {str(v)} better than {default_bound}; setting the upper bound to {default_bound}'
                )
                v.setub(default_bound)
        # If changing v.lb makes it larger than v.lb, set them equal.
        if v.lb > v.ub:
            logger.debug(
                f'Lower bound was set higher than upper bound, which is not allowed.'
                'Setting upper bound equal to lower bound.'
            )
            v.setub(v.lb)
    fbbt(m)
    # Slightly shift the bounds to prevent math domain error in log or exp functions
    for v in get_vars(m):
        d = v.ub - v.lb
        d *= 1e-6
        d = min(d, 1e-6)
        v.setlb(v.lb + d)
        v.setub(v.ub - d)
