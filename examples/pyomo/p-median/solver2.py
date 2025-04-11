#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Imports from Pyomo
import pyomo.environ as pyo
from pyomo.common.plugin_base import alias, implements
from pyomo.opt import SolverStatus, SolutionStatus, ProblemSense
import random
import copy


@plugin_factory
class MySolver(object):
    alias('random')

    # Declare that this is an IOptSolver plugin
    implements(IOptSolver)

    # Solve the specified problem and return
    # a SolverResults object
    def solve(self, instance, **kwds):
        print("Starting random heuristic")
        val, sol = self._random(instance)
        n = pyo.value(instance.N)
        # Setup results
        results = pyo.SolverResults()
        results.problem.name = instance.name
        results.problem.sense = ProblemSense.minimize
        results.problem.num_constraints = 1
        results.problem.num_variables = n
        results.problem.num_objectives = 1
        results.solver.status = SolverStatus.ok
        soln = results.solution.add()
        soln.value = val
        soln.status = SolutionStatus.feasible
        for j in pyo.sequence(n):
            soln.variable[instance.y[j].name] = {"Value": sol[j - 1], "Id": j}
        # Return results
        return results

    # Perform a random search
    def _random(self, instance):
        sol = [0] * instance.N.value
        for j in range(instance.P.value):
            sol[j] = 1
        # Generate 100 random solutions, and keep the best
        best = None
        best_sol = []
        for kk in range(100):
            random.shuffle(sol)
            # Compute value
            val = 0.0
            for j in pyo.sequence(instance.M.value):
                val += min(
                    [
                        instance.d[i, j].value
                        for i in pyo.sequence(instance.N.value)
                        if sol[i - 1] == 1
                    ]
                )
            if best is None or val < best:
                best = val
                best_sol = copy.copy(sol)
        return [best, best_sol]
