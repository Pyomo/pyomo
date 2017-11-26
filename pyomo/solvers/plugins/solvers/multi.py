from __future__ import division
import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
import logging
import numpy as np
import random
import pyomo.util.plugin
import textwrap
import copy
from pyomo.core.base.var import Var
from pyomo.core.kernel.numvalue import value
from pyomo.core.base.objective import maximize
from pyomo.environ import *
import math
logger = logging.getLogger('pyomo.solver')

stopping_mass = .5
stopping_delta = .5

class Multistart(OptSolver):
    '''Solver wrapper that can check multiple starting points
    '''
    pyomo.util.plugin.alias('multistart',doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        pass
    def solve (self,model,**kwds):
        strategy = kwds.pop('strategy','rand')
        solver = kwds.pop('solver','ipopt')
        iterations = kwds.pop('iterations',-1)
        solver = SolverFactory(solver)
        objectives = []
        models = []
        m = copy.deepcopy(model)
        result = solver.solve(m)
        if result.solver.status is SolverStatus.ok and result.solver.termination_condition is  TerminationCondition.optimal:
            val = value(m.obj.expr)
            objectives.append(val)
            models.append(m)
        if iterations == -1:
            while not self.should_stop(objectives):
                m = copy.deepcopy(model)
                self.reinitialize_all(m,strategy)
                result = solver.solve(m)
                if result.solver.status is SolverStatus.ok and result.solver.termination_condition is  TerminationCondition.optimal:
                    val = value(m.obj.expr)
                    objectives.append(val)
                    models.append(m)

        else:
            for i in xrange(iterations):
                m = copy.deepcopy(model)
                self.reinitialize_all(m,strategy)
                result = solver.solve(m)
                if result.solver.status is SolverStatus.ok and result.solver.termination_condition is  TerminationCondition.optimal:
                    val = value(m.obj.expr)
                    objectives.append(val)
                    models.append(m)

        if model.obj.sense == maximize:
            newmodel = models[np.argmax(objectives)]
        else:
            newmodel = models[np.argmin(objectives)]

        oldvars = list(model.component_data_objects(ctype=Var, descend_into=True))
        newvars = list(newmodel.component_data_objects(ctype=Var, descend_into=True))
        for i in xrange(len(oldvars)):
            var = oldvars[i]
            newvar = newvars[i]
            if not var.is_fixed() and not var.is_binary() and not var.is_integer() \
                    and not (var is None or var.lb is None or var.ub is None or strategy is None):
                var.value = value(newvar)



    def reinitialize_all(self,model,strategy):
        def rand(val, lb, ub):
            return (ub - lb) * random.random() + lb

        def midpoint_guess_and_bound(val, lb, ub):
            bound = ub if ((ub - val) >= (val - lb)) else lb
            return (bound + val) // 2

        def rand_guess_and_bound(val, lb, ub):
            bound = ub if ((ub - val) >= (val - lb)) else lb
            return (abs(bound - val) * random.random()) + min(bound, val)

        def rand_distributed(val, lb, ub, divisions=9):
            linspace=np.linspace(lb,ub,divisions)
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
    def num_one_occurrences(self,lst):
        dist = {}
        for x in lst:
            if x in dist:
                dist[x] += 1
            else:
                dist[x] = 1
        one_offs = [x for x in dist if dist[x] == 1]
        return len(one_offs)

    def should_stop(self,solutions):
        f = self.num_one_occurrences(solutions)
        n = len(solutions)
        d = stopping_delta
        c = stopping_mass
        confidence = f/n + (2*math.sqrt(2) + math.sqrt(3))* math.sqrt(math.log(3/d)/n)
        return confidence<c
