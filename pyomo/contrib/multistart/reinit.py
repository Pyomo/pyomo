"""Helper functions for variable reinitialization."""
from __future__ import division

import logging
import random

from six.moves import range

from pyomo.core import Var

logger = logging.getLogger('pyomo.contrib.multistart')


def rand(val, lb, ub):
    return random.uniform(lb, ub)  # uniform distribution between lb and ub


def midpoint_guess_and_bound(val, lb, ub):
    """Midpoint between current value and farthest bound."""
    far_bound = ub if ((ub - val) >= (val - lb)) else lb  # farther bound
    return (far_bound + val) / 2


def rand_guess_and_bound(val, lb, ub):
    """Random choice between current value and farthest bound."""
    far_bound = ub if ((ub - val) >= (val - lb)) else lb  # farther bound
    return random.uniform(val, far_bound)


def rand_distributed(val, lb, ub, divisions=9):
    """Random choice among evenly distributed set of values between bounds."""
    set_distributed_vals = linspace(lb, ub, divisions)
    return random.choice(set_distributed_vals)


def linspace(lower, upper, n):
    """Linearly spaced range."""
    return [lower + x * (upper - lower) / (n - 1) for x in range(n)]


def reinitialize_variables(model, config):
    """Reinitializes all variable values in the model.

    Excludes fixed, noncontinuous, and unbounded variables.

    """
    for var in model.component_data_objects(ctype=Var, descend_into=True):
        if var.is_fixed() or not var.is_continuous():
            continue
        if var.lb is None or var.ub is None:
            if not config.suppress_unbounded_warning:
                logger.warning(
                    'Unable to reinitialize value of unbounded variable '
                    '%s with bounds (%s, %s). '
                    'To suppress this message, set the '
                    'suppress_unbounded_warning flag.'
                    % (var.name, var.lb, var.ub))
            continue
        val = var.value if var.value is not None else (var.lb + var.ub) / 2
        # apply reinitialization strategy to variable
        strategies = {
            "rand": rand,
            "midpoint_guess_and_bound": midpoint_guess_and_bound,
            "rand_guess_and_bound": rand_guess_and_bound,
            "rand_distributed": rand_distributed
        }
        var.value = strategies[config.strategy](val, var.lb, var.ub)
