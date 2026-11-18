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

logger = logging.getLogger('pyomo.contrib.multistart')


def rand(val, lb, ub, u):
    """Sampled point between the bounds."""
    return lb + u * (ub - lb)


def midpoint_guess_and_bound(val, lb, ub, u=None):
    """Midpoint between current value and farthest bound."""
    far_bound = ub if ((ub - val) >= (val - lb)) else lb  # farther bound
    return (far_bound + val) / 2


def rand_guess_and_bound(val, lb, ub, u):
    """Sampled point between current value and farthest bound."""
    far_bound = ub if ((ub - val) >= (val - lb)) else lb  # farther bound
    return val + u * (far_bound - val)


def rand_distributed(val, lb, ub, u, divisions=9):
    """Sampled choice among evenly distributed set of values between bounds."""
    set_distributed_vals = linspace(lb, ub, divisions)
    return set_distributed_vals[min(int(u * divisions), divisions - 1)]


def simple_midpoint(val, lb, ub, u=None):
    return (lb + ub) * 0.5


def linspace(lower, upper, n):
    """Linearly spaced range."""
    return [lower + x * (upper - lower) / (n - 1) for x in range(n)]


strategies = {
    "rand": rand,
    "midpoint_guess_and_bound": midpoint_guess_and_bound,
    "rand_guess_and_bound": rand_guess_and_bound,
    "rand_distributed": rand_distributed,
    "midpoint": simple_midpoint,
}


def reinitialize_variables(var_list, config, sampler):
    """Reinitializes the values of the variables in var_list.

    Excludes fixed, noncontinuous, and unbounded variables. One point is
    drawn from ``sampler`` for all eligible variables together, and the
    strategy maps each coordinate onto that variable's range.

    """

    eligible_vars = []
    for var in var_list:
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

    if len(eligible_vars) == 0:
        return

    point = sampler.unit_point(len(eligible_vars))
    strategy = strategies[config.strategy]
    for var, u in zip(eligible_vars, point):
        val = var.value if var.value is not None else (var.lb + var.ub) / 2
        var.set_value(strategy(val, var.lb, var.ub, u), skip_validation=True)
