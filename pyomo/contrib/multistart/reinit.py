# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Helper functions for variable reinitialization."""

import logging
import random
from pyomo.common.dependencies import numpy as np
from pyomo.common.dependencies.scipy import stats
from pyomo.util.vars_from_expressions import get_vars_from_components

from pyomo.core import Var

logger = logging.getLogger('pyomo.contrib.multistart')


def rand(val, lb, ub, rng):
    sample = rng.uniform(lb, ub)  # uniform distribution between lb and ub
    return sample


def rand_vector(lbs, ubs, sampler):
    lowers = lbs
    uppers = ubs
    # Generate vector of samples using sampler
    samples = sampler.sample_vector(lowers, uppers)
    return samples


def midpoint_guess_and_bound(val, lb, ub, rng=None):
    """Midpoint between current value and farthest bound."""
    far_bound = ub if ((ub - val) >= (val - lb)) else lb  # farther bound
    return (far_bound + val) / 2


def rand_guess_and_bound(val, lb, ub, rng):
    """Random choice between current value and farthest bound."""
    far_bound = ub if ((ub - val) >= (val - lb)) else lb  # farther bound
    if far_bound == ub:
        return rng.uniform(val, far_bound)
    else:
        return rng.uniform(far_bound, val)


def rand_distributed(val, lb, ub, rng, divisions=9):
    """Random choice among evenly distributed set of values between bounds."""
    set_distributed_vals = linspace(lb, ub, divisions)
    return rng.choice(set_distributed_vals)


def simple_midpoint(val, lb, ub, rng=None):
    return (lb + ub) * 0.5


def linspace(lower, upper, n):
    """Linearly spaced range."""
    return [lower + x * (upper - lower) / (n - 1) for x in range(n)]


strategies = {
    "rand": rand,
    "rand_vector": rand_vector,
    "midpoint_guess_and_bound": midpoint_guess_and_bound,
    "rand_guess_and_bound": rand_guess_and_bound,
    "rand_distributed": rand_distributed,
    "midpoint": simple_midpoint,
}


def reinitialize_variables(model, config, sampler):
    """Reinitializes all variable values in the model.

    Excludes fixed, noncontinuous, and unbounded variables.

    """

    eligible_vars = []
    print(model._vars_list[0].value)
    for var in model._vars_list:
        if var.is_fixed() or not var.is_continuous():
            continue
        if var.lb is None or var.ub is None:
            if not config.suppress_unbounded_warning:
                logger.warning(
                    'Skipping reinitialization of unbounded variable '
                    '%s with bounds (%s, %s). '
                    'To suppress this message, set the '
                    'suppress_unbounded_warning flag.' % (var.name, var.lb, var.ub)
                )
            continue

        eligible_vars.append(var)

    if config.strategy == "rand_vector":

        if len(eligible_vars) == 0:
            raise ValueError(
                "No eligible variables to reinitialize." "Please add bounds."
            )
        # Collect lower and upper bounds for sampler
        lowers = [v.lb for v in eligible_vars]
        uppers = [v.ub for v in eligible_vars]

        samples = rand_vector(lowers, uppers, sampler)

        # assign samples to variables
        for var, sample in zip(eligible_vars, samples):
            var.set_value(sample, skip_validation=True)

        return

    # Otherwise
    else:
        for var in eligible_vars:
            val = var.value if var.value is not None else (var.lb + var.ub) / 2
            # print(f"val = {val}\n")
            # apply reinitialization strategy to variable
            var.set_value(
                strategies[config.strategy](val, var.lb, var.ub, sampler.rng),
                skip_validation=True,
            )
