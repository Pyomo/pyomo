#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from __future__ import division

import logging
import math
import random

from six.moves import range

from pyomo.common.config import (
    ConfigBlock, ConfigValue, In, add_docstring_list
)
from pyomo.core import Var, maximize, value
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

logger = logging.getLogger('pyomo.contrib.multistart')


@SolverFactory.register('multistart',
                        doc='MultiStart solver for NLPs')
class MultiStart(object):
    """Solver wrapper that initializes at multiple starting points.

    # TODO: also return appropriate duals

    For theoretical underpinning, see
    https://www.semanticscholar.org/paper/How-many-random-restarts-are-enough-Dick-Wong/55b248b398a03dc1ac9a65437f88b835554329e0

    Keyword arguments below are specified for the ``solve`` function.

    """

    CONFIG = ConfigBlock("MultiStart")
    CONFIG.declare("strategy", ConfigValue(
        default="rand", domain=In([
            "rand", "midpoint_guess_and_bound",
            "rand_guess_and_bound", "rand_distributed"]),
        description="Specify the restart strategy. Defaults to rand.",
        doc="""Specify the restart strategy.

        - "rand": pure random choice
        - "midpoint_guess_and_bound": midpoint between current and farthest bound
        - "rand_guess_and_bound": midpoint between current and farthest bound
        - "rand_distributed": random choice among evenly distributed values
        """
    ))
    CONFIG.declare("solver", ConfigValue(
        default="ipopt",
        description="solver to use, defaults to ipopt"
    ))
    CONFIG.declare("iterations", ConfigValue(
        default=10,
        description="Specify the number of iterations, defaults to 10. "
        "If -1 is specified, the high confidence stopping rule will be used"
    ))
    CONFIG.declare("stopping_mass", ConfigValue(
        default=0.5,
        description="Maximum allowable estimated missing mass of optima.",
        doc="""Maximum allowable estimated missing mass of optima for the
        high confidence stopping rule, only used with the random strategy.
        The lower the parameter, the stricter the rule.
        Value bounded in (0, 1]."""
    ))
    CONFIG.declare("stopping_delta", ConfigValue(
        default=0.5,
        description="1 minus the confidence level required for the stopping rule.",
        doc="""1 minus the confidence level required for the stopping rule for the
        high confidence stopping rule, only used with the random strategy.
        The lower the parameter, the stricter the rule.
        Value bounded in (0, 1]."""
    ))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def solve(self, model, **kwds):
        # initialize keyword args
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        # initialize the solver
        solver = SolverFactory(config.solver)

        # store objective values and their respective models, and
        # results from solver
        objectives = []
        models = []
        results = []
        # store the model from the initial values
        model._vars_list = list(model.component_data_objects(
            ctype=Var, descend_into=True))
        m = model.clone()
        result = solver.solve(m)
        if result.solver.status is SolverStatus.ok and result.solver.termination_condition is TerminationCondition.optimal:
            val = value(m.obj.expr)
            objectives.append(val)
            models.append(m)
            results.append(result)
        num_iter = 0
        # if HCS rule is specified, reinitialize completely randomly until rule specifies stopping
        if config.iterations == -1:
            while not self.should_stop(objectives, (
                    config.stopping_mass, config.stopping_delta)):
                num_iter += 1
                m = model.clone()
                self.reinitialize_all(m, 'rand')
                result = solver.solve(m)
                if result.solver.status is SolverStatus.ok and result.solver.termination_condition is TerminationCondition.optimal:
                    val = value(m.obj.expr)
                    objectives.append(val)
                    models.append(m)
                    results.append(result)
        # if HCS rule is not specified, iterate, while reinitializing with given strategy
        else:
            for i in range(config.iterations):
                m = model.clone()

                self.reinitialize_all(m, config.strategy)
                result = solver.solve(m)
                if result.solver.status is SolverStatus.ok and result.solver.termination_condition is TerminationCondition.optimal:
                    val = value(m.obj.expr)
                    objectives.append(val)
                    models.append(m)
                    results.append(result)
        if model.obj.sense == maximize:
            i = argmax(objectives)
            newmodel = models[i]
            opt_result = results[i]
        else:
            i = argmin(objectives)
            newmodel = models[i]
            opt_result = results[i]

        # reassign the given models vars to the new models vars
        for i, var in enumerate(model._vars_list):
            if not var.is_fixed() and not var.is_binary() and not var.is_integer() \
                    and not (var is None or var.lb is None or var.ub is None or config.strategy is None):
                var.value = newmodel._vars_list[i].value

        return opt_result

    def reinitialize_all(self, model, strategy):
        def rand(val, lb, ub):
            return (ub - lb) * random.random() + lb

        def midpoint_guess_and_bound(val, lb, ub):
            bound = ub if ((ub - val) >= (val - lb)) else lb
            return (bound + val) // 2

        def rand_guess_and_bound(val, lb, ub):
            bound = ub if ((ub - val) >= (val - lb)) else lb
            return (abs(bound - val) * random.random()) + min(bound, val)

        def rand_distributed(val, lb, ub, divisions=9):
            lsp = linspace(lb, ub, divisions)
            return random.choice(lsp)

        for var in model.component_data_objects(ctype=Var, descend_into=True):
            if not var.is_fixed() and not var.is_binary() and not var.is_integer() \
                    and not (var is None or var.lb is None or var.ub is None or strategy is None):
                val = value(var)
                lb = var.lb
                ub = var.ub
                # apply strategy to bounds/variable
                strategies = {"rand": rand,
                              "midpoint_guess_and_bound": midpoint_guess_and_bound,
                              "rand_guess_and_bound": rand_guess_and_bound,
                              "rand_distributed": rand_distributed
                              }
                var.value = strategies[strategy](val, lb, ub)

# High Confidence Stopping rule
# This stopping operates by estimating the amount of missing optima, and stops
# once the estimated mass of missing optima is within an acceptable range, with
# some confidence.

    def num_one_occurrences(self, lst):
        """
        Determines the number of optima that have only been observed once.
        Needed to estimate missing mass of optima.
        """
        dist = {}
        for x in lst:
            if x in dist:
                dist[x] += 1
            else:
                dist[x] = 1
        one_offs = [x for x in dist if dist[x] == 1]
        return len(one_offs)

    def should_stop(self, solutions, hcs_param):
        """
        Determines if the missing mass of unseen local optima is acceptable
        based on the High Confidence stopping rule.
        """
        f = self.num_one_occurrences(solutions)
        n = len(solutions)
        (stopping_mass, stopping_delta) = hcs_param
        d = stopping_delta
        c = stopping_mass
        confidence = f / n + (2 * math.sqrt(2) + math.sqrt(3)
                              ) * math.sqrt(math.log(3 / d) / n)
        return confidence < c


# Helper functions to remove numpy dependency
def argmax(lst):
    """Returns index of largest value in list lst"""
    return max(range(len(lst)), key=lambda i: lst[i])


def argmin(lst):
    """Returns index of smallest value in list lst"""
    return min(range(len(lst)), key=lambda i: lst[i])


def linspace(lower, upper, n):
    """Linearly spaced range."""
    return [lower + x * (upper - lower) / (n - 1) for x in range(n)]
