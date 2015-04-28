#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import time
import pyutilib.misc
import pyomo.opt


class MPEC_Solver1(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('mpec_nlp', doc='MPEC solver that optimizes a nonlinear transformation')

    def __init__(self, **kwds):
        kwds['type'] = 'mpec_nlp'
        pyomo.opt.OptSolver.__init__(self,**kwds)

    def _presolve(self, *args, **kwds):
        #
        # Cache the instance
        #
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()
        #
        # Transform instance
        #
        instance = self._instance.transform('mpec.simple_nonlinear')
        #instance.pprint()
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            self.options.solver = solver = 'ipopt'
        opt = pyomo.opt.SolverFactory(solver)
        #
        self.results = []
        epsilon_final = self.options.get('epsilon_final', 1e-7)
        epsilon = self.options.get('epsilon_initial', epsilon_final)
        while (True):
            instance.mpec_bound.value = epsilon
            res = opt.solve( instance, tee=self.tee,
                             timelimit=self._timelimit )
            self.results.append(res)
            instance.load(res)
            epsilon /= 10.0
            if epsilon < epsilon_final:
                break
        #
        # Reclassify the Complementarity components
        #
        from pyomo.mpec import Complementarity
        for cuid in self._instance._transformation_data['mpec.simple_nonlinear'].compl_cuids:
            cobj = cuid.find_component(self._instance)
            cobj.parent_block().reclassify_component_type(cobj, Complementarity)
        #
        # Update timing
        #
        stop_time = time.time()
        self.wall_time = stop_time - start_time
        #
        # Return the sub-solver return condition value and log
        #
        return pyutilib.misc.Bunch(rc=getattr(opt,'_rc', None), log=getattr(opt,'_log',None))

    def _postsolve(self):
        #
        # Create a results object
        #
        results = pyomo.opt.SolverResults()
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.options.subsolver
        solv.wallclock_time = self.wall_time
        cpu_ = []
        for res in self.results:
            if not getattr(res.solver, 'cpu_time', None) is None:
                cpu_.append( res.solver.cpu_time )
        if len(cpu_) > 0:
            solv.cpu_time = sum(cpu_)
        #solv.termination_condition = pyomo.opt.TerminationCondition.maxIterations
        #
        # PROBLEM
        #
        self._instance.compute_statistics()
        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables = self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables = self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives
        #
        # SOLUTION(S)
        #
        soln, results._symbol_map = self._instance.get_solution()
        results.solution.insert( soln )
        #
        # Uncache the instance and return the results
        #
        self._instance = None
        print(results)
        return results

