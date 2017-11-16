import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
import logging
import numpy as np
import pyomo.environ as pe
import pyomo.util.plugin
import textwrap

logger = logging.getLogger('pyomo.solver')



class Multistart(Solver):
    '''NLP Solver wrapper that can check multiple starting points
    '''
    alias('multistart',doc=textwrap.fill(textwrap.dedent(__doc__.strip())))
#TODO: CHange to keywords
    def solve (self,model,**kwds):
        strategy = kwds.pop('strategy','rand_distributed')
        solver = kwds.pop('solver','ipopt')
        iterations = kwds.pop('iterations',5)
        solver = SolverFactory(solver)
        objectives = []
        models = []
        for i in xrange(iterations):
            m = deepcopy(model)
            result = solver.solve(model)
            if result.solver.status is SolverStatus.ok and result.solver.termination_condition is  TerminationCondition.optimal:
                val = pe.value(m.obj.expr)
                objectives.append(val)
                models.append(m)
            reinitialize_all(model,strategy = strategy)

        if model.obj.sense == maximize:
            model = models[np.argmax(objectives)]
        else:
            model = models[np.argmin(objectives)]


    def reinitialize_all(model,strategy):
        def midpoint(val, lb, ub):
            return (lb + ub) // 2

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

        # iterate thorugh all units in flowsheet
        for o in itervalues(self.units):
            # iterate through all variables in that unit
            for var in self.component_data_objects(ctype=pe.Var, descend_into=True):
                if not var.is_fixed() and not var.is_binary() and not var.is_integer() \
                        and not (var is None or var.lb is None or var.ub is None or strategy is None):
                    val = pe.value(var)
                    lb = var.lb
                    ub = var.ub
                    # apply strategy to bounds/variable
                    strategies = {"midpoint": midpoint,
                                  "rand": rand,
                                  "midpoint_guess_and_bound": midpoint_guess_and_bound,
                                  "rand_guess_and_bound": rand_guess_and_bound,
                                  "rand_distributed": rand_distributed
                                  }
                    var.value = strategies[strategy](val, lb, ub)
