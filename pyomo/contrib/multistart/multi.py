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

import copy
import logging
import math
import random
import textwrap

import numpy as np

from pyomo.common.plugin import alias
from pyomo.core.base.objective import maximize
from pyomo.core.base.var import Var
from pyomo.core.kernel.numvalue import value
from pyomo.environ import *
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *

logger = logging.getLogger('pyomo.contrib.multistart')


class MultiStart(OptSolver):
    """Solver wrapper that initializes at multiple starting points.

    For theoretical underpinning, see https://www.semanticscholar.org/paper/How-many-random-restarts-are-enough-Dick-Wong/55b248b398a03dc1ac9a65437f88b835554329e0

    Keywords:
        strategy: specify the restart strategy, defaults to rand
            - "rand": pure random choice
            - "midpoint_guess_and_bound": midpoint between current and
                                       farthest bound
            - "rand_guess_and_bound": midpoint between current and
                                       farthest bound
            - "rand_distributed": random choice among evenly distributed values
        solver: solver to use, defaults to ipopt
        iterations: specify the number of iterations, defaults to 10.
                        if -1 is specified, the high confidence stopping rule will be used
        HCS_param: specify the tuple (m,d)
            defaults to (m,d) = (.5,.5)
            only use with random strategy
            The stopping mass m is the maximum allowable estimated missing mass of optima
            The stopping delta d = 1-the confidence level required for the stopping rule
            For both parameters, the lower the parameter the more stricter the rule.
            both are bounded 0<x<=1

    """
    alias('multistart', doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def solve(self, model, **kwds):
        # initialize keyword args
        strategy = kwds.pop('strategy', 'rand')
        solver = kwds.pop('solver', 'ipopt')
        iterations = kwds.pop('iterations', 10)
        hcs_param = kwds.pop('HCS_param', (.5, .5))

        # initialize the solver
        solver = SolverFactory(solver)

        # store objective values and their respective models, and
        # results from solver
        objectives = []
        models = []
        results = []
        # store the model from the initial values
        m = copy.deepcopy(model)
        result = solver.solve(m)
        if result.solver.status is SolverStatus.ok and result.solver.termination_condition is TerminationCondition.optimal:
            val = value(m.obj.expr)
            objectives.append(val)
            models.append(m)
            results.append(result)
        num_iter = 0
        # if HCS rule is specified, reinitialize completely randomly until rule specifies stopping
        if iterations == -1:
            while not self.should_stop(objectives, hcs_param):
                num_iter += 1
                m = copy.deepcopy(model)
                self.reinitialize_all(m, 'rand')
                result = solver.solve(m)
                if result.solver.status is SolverStatus.ok and result.solver.termination_condition is TerminationCondition.optimal:
                    val = value(m.obj.expr)
                    objectives.append(val)
                    models.append(m)
                    results.append(result)
        # if HCS rule is not specified, iterate, while reinitializing with given strategy
        else:
            for i in xrange(iterations):
                m = copy.deepcopy(model)
                self.reinitialize_all(m, strategy)
                result = solver.solve(m)
                if result.solver.status is SolverStatus.ok and result.solver.termination_condition is TerminationCondition.optimal:
                    val = value(m.obj.expr)
                    objectives.append(val)
                    models.append(m)
                    results.append(result)
        if model.obj.sense == maximize:
            i = np.argmax(objectives)
            newmodel = models[i]
            opt_result = results[i]
        else:
            i = np.argmin(objectives)
            newmodel = models[i]
            opt_result = results[i]

        oldvars = list(model.component_data_objects(
            ctype=Var, descend_into=True))
        newvars = list(newmodel.component_data_objects(
            ctype=Var, descend_into=True))

        # reassign the given models vars to the new models vars
        for i in xrange(len(oldvars)):
            var = oldvars[i]
            newvar = newvars[i]
            if not var.is_fixed() and not var.is_binary() and not var.is_integer() \
                    and not (var is None or var.lb is None or var.ub is None or strategy is None):
                var.value = value(newvar)
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
            linspace = np.linspace(lb, ub, divisions)
            return np.random.choice(linspace)

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

    # determines the number of optima that have only been observed once.
    # needed to estimate missing mass of optima
    def num_one_occurrences(self, lst):
        dist = {}
        for x in lst:
            if x in dist:
                dist[x] += 1
            else:
                dist[x] = 1
        one_offs = [x for x in dist if dist[x] == 1]
        return len(one_offs)

    # determines if the missing mass of unseen local optima is acceptable
    # based on the High Confidence stopping rule.
    def should_stop(self, solutions, (stopping_mass, stopping_delta)):
        f = self.num_one_occurrences(solutions)
        n = len(solutions)
        d = stopping_delta
        c = stopping_mass
        confidence = f / n + (2 * math.sqrt(2) + math.sqrt(3)
                              ) * math.sqrt(math.log(3 / d) / n)
        return confidence < c
