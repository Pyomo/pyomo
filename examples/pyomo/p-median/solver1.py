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
import copy
import pyomo.environ as pyo
from pyomo.common.plugin_base import alias, implements
from pyomo.opt import SolverStatus, SolutionStatus, ProblemSense


@plugin_factory(pyo.SolverFactory)
class MySolver(object):
    alias('greedy')

    # Declare that this is an IOptSolver plugin
    implements(IOptSolver)

    # Solve the specified problem and return
    # a SolverResults object
    def solve(self, instance, **kwds):
        print("Starting greedy heuristic")
        val, instance = self._greedy(instance)
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
            if instance.y[j].value is 1:
                soln.variable[instance.y[j].name] = {"Value": 1, "Id": j}
        return results

    # Perform a greedy search
    def _greedy(self, instance):
        p = pyo.value(instance.P)
        n = pyo.value(instance.N)
        m = pyo.value(instance.M)
        fixed = set()
        # Initialize
        for j in pyo.sequence(n):
            instance.y[j].value = 0
        # Greedily fix the next best facility
        for i in pyo.sequence(p):
            best = None
            ndx = j
            for j in pyo.sequence(n):
                if j in fixed:
                    continue
                instance.y[j].value = 1
                # Compute value
                val = 0.0
                for kk in pyo.sequence(m):
                    tmp = copy.copy(fixed)
                    tmp.add(j)
                    tbest = None
                    for jj in tmp:
                        if tbest is None or instance.d[jj, kk].value < tbest:
                            tbest = instance.d[jj, kk].value
                    val += tbest
                # Keep best greedy choice
                if best is None or val < best:
                    best = val
                    ndx = j
                instance.y[j].value = 0
            fixed.add(ndx)
            instance.y[ndx].value = 1
        return [best, instance]
