import logging

import pyomo.util.plugin
from pyomo.opt import OptSolver SolverFactory

logger = logging.getLogger('pyomo.solver')


def solve (model,solver,strategy = "rand_distributed",iterations = 5):
	solver = SolverFactory(solver)
	solver.solve(model)
	reinitialize_all(model,strategy = strategy)
    solve(model,solver,strategy)







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
        linspace=[]
        spacing = (ub - lb) / divisions
        for i in range(divisions - 1):
            linspace.append(spacing * (i + 1) + lb)
        return random.choice(linspace)

    # iterate thorugh all units in flowsheet
    print("attempting reinitialization on all variables")
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