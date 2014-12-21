#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.opt
import pyutilib.misc


class MPEC_Solver1(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('mpec_ils', doc='MPEC solver with iterative local search')

    def __init__(self, **kwds):
        kwds['type'] = 'mpec_ils'
        pyomo.opt.OptSolver.__init__(self,**kwds)

    def _presolve(self, *args, **kwds):
        #
        # Cache the instance
        #
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        #
        # Transform instance
        #
        # TODO: This transformation needs to penalize small
        # bounds on the complementarity conditions.
        #
        instance = self._instance.transform('mpec.simple_nonlinear')
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            solver = 'ipopt'
        opt = pyomo.opt.SolverFactory(solver)
        #
        bound = 1e-1
        while (True):
            instance.mpec_bound.value = bound
            res = opt.solve(instance, tee=self.tee,
                                 timelimit=self._timelimit)
            instance.load(res)
            bound /= 10.0
            if bound < 1e-3:
                break
        #
        # Transform the result back into the original model
        #
        self.results = instance.update_results(res)
        # TODO: This doesn't work yet.

        #
        # Return the sub-solver return condition value and log
        #
        return pyutilib.misc.Bunch(rc=getattr(opt,'_rc', None), log=getattr(opt,'_log',None))

    def _postsolve(self):
        #
        # Uncache the instance
        #
        self._instance = None
        #
        # Return the results
        #
        # TODO: initialize the SolverResults or use the results data from
        #   the subsover???  Probably the latter.
        #
        return self.results

    def X_postsolve(self):
        results = SolverResults()
        solv = results.solver
        solv.name = self.options.subsolver
        #solv.status = self._glpk_get_solver_status()
        #solv.memory_used = "%d bytes, (%d KiB)" % (peak_mem, peak_mem/1024)
        solv.wallclock_time = self._ans.elapsed['solver_time']
        solv.cpu_time = self._ans.elapsed['solver_cputime']
        solv.termination_condition = TerminationCondition.maxIterations
        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables = self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables = self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives

        from pyomo.core import maximize
        if self.problem.sense == maximize:
            prob.sense = ProblemSense.maximize
        else:
            prob.sense = ProblemSense.minimize

        if not sstatus in ( SolutionStatus.error, ):
            soln = Solution()
            soln.status = sstatus

            if type(self._ans.ff) in (list, tuple):
                oval = float(self._ans.ff[0])
            else:
                oval = float(self._ans.ff)
            if self.problem.sense == maximize:
                soln.objective[ self.problem._f_name[0] ].value = - oval
            else:
                soln.objective[ self.problem._f_name[0] ].value = oval

            id = 0
            for var_label in self._ans.xf.keys():
                if self._ans.xf[var_label].is_integer():
                    soln.variable[ var_label.name ] = {'Value': int(self._ans.xf[var_label]), 'Id':id}
                else:
                    soln.variable[ var_label.name ] = {'Value': float(self._ans.xf[var_label]), 'Id':id}
                id += 1

            results.solution.insert( soln )

        return results

