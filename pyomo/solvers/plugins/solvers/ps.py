#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from six.moves import xrange

from pyutilib.misc import Bunch

import pyomo.util.plugin
from pyomo.opt import SolverResults, ProblemSense, SolverStatus, SolutionStatus, TerminationCondition
from pyomo.opt.blackbox import solver
from pyomo.opt.blackbox import problem


class PatternSearch(solver.COLINSolver):
    """A pattern search optimizer"""

    pyomo.util.plugin.alias('ps', doc='Pyomo\'s simple pattern search optimizer')

    def __init__(self, **kwds):
        """ Constructor """
        kwds['type'] = "PatternSearch"
        solver.COLINSolver.__init__(self, **kwds)
        self.tmp_point = problem.MixedIntVars()
        #
        # TODO: add a mechanism for managing optiosn
        #
        self.min_function_value = 0.0
        self.max_iterations = 100
        self.min_delta = 1e-5
        self.lower=[]
        self.upper=[]
        self.scale=[]
        self.initial_delta = 1.0
        self.initial_point = None
        self.debug=False

    def reset(self):
        self.delta = self.initial_delta
        if self.problem is not None:
            self.lower = self.problem.real_lower
            self.upper = self.problem.real_upper
        if self.initial_point is None:
            #
            # TODO: set the solver state with this value
            #
            raise ValueError("Pattern Search optimizer must have an initial point specified")
        if not self.feasible(self.initial_point):
            #
            # TODO: set the solver state with this value
            #
            raise ValueError("Pattern Search optimizer initial point must be feasible: "+str(self.initial_point))
        self.x = self.initial_point
        self.scale=[1.0]*len(self.x)
        for i in xrange(0,len(self.x)):
            if type(self.lower[i]) in (float,int) and type(self.upper[i]) in (float,int):
                self.scale[i] = (self.upper[i]-self.lower[i])/10.0

    def feasible(self,point):
        if len(self.lower) == 0:
            return True
        for i in xrange(0,len(self.lower)):
            if self.lower[i] is not None and point[i] < self.lower[i]:
                return False
            if self.upper[i] is not None and point[i] > self.upper[i]:
                return False
        return True

    def evaluate(self,point):
        self.tmp_point.reals = point
        return self.problem.function_value(self.tmp_point)

    def _apply_solver(self):
        """ The routine in a derived class that performs the solve """
        x = self.x
        best = self.evaluate(x)
        ndim=len(x)
        curr_iter=0
        tstatus = TerminationCondition.maxIterations
        tmsg=None
        while curr_iter < self.max_iterations:
            #
            # TODO: use logging here
            #
            if self.debug:      #pragma:nocover
                print("PatternSearch",curr_iter,x,best,self.delta)
            #
            # TODO: add a mechanism to archive the reason for termination
            #
            if best <= self.min_function_value:
                tstatus = TerminationCondition.minFunctionValue
                tmsg = "%s < %s" % (best,self.min_function_value)
                break
            if self.delta <= self.min_delta:
                tstatus = TerminationCondition.minStepLength
                tmsg = "%s < %s" % (self.delta,self.min_delta)
                break
            #
            # Iterate, and keep the best point
            #
            best_new_step=None
            best_new_value=best
            for j in xrange(0,ndim):
                orig=x[j]
                #
                # x[j]-Delta
                #
                x[j] = orig-self.delta*self.scale[j]
                #
                # TODO: use logging here
                #
                if self.debug:          #pragma:nocover
                    print("  Trial Point: ",x)
                if self.lower[j] is None or x[j] >= self.lower[j]:
                    tmp = self.evaluate(x)
                    if tmp < best_new_value:
                        best_new_step=(j,-1)
                        best_new_value=tmp
                #
                # x[j]+Delta
                #
                x[j] = orig+self.delta*self.scale[j]
                #
                # TODO: use logging here
                #
                if self.debug:      #pragma:nocover
                    print("  Trial Point: ",x)
                if self.upper[j] is None or x[j] <= self.upper[j]:
                    tmp = self.evaluate(x)
                    if tmp < best_new_value:
                        best_new_step=(j,1)
                        best_new_value=tmp
                x[j]=orig
            if best_new_value < best:
                #
                # Move to an improving point
                #
                x[best_new_step[0]] = x[best_new_step[0]] + best_new_step[1]*self.delta*self.scale[best_new_step[0]]
                best = best_new_value
                self.delta *= 2.0
            else:
                #
                # Contract about a non-improving point
                #
                self.delta /= 3.0
            curr_iter += 1
        #
        # Archive results in a SolverResults object
        #
        results = SolverResults()
        #
        results.solver.number_of_iterations = curr_iter
        results.solver.final_step_length = self.delta
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = tstatus
        results.solver.termination_message = tmsg
        #
        results.problem.number_of_objectives=1
        results.problem.number_of_constraints=0
        results.problem.number_of_variables=ndim
        results.problem.sense=ProblemSense.minimize
        results.problem.upper_bound=best
        #
        soln = results.solution.add()
        soln.objective['f'] = {'Value': best}
        soln.status = SolutionStatus.bestSoFar
        for i in xrange(ndim):
            soln.variable["x"+str(i)] = {"Value" : x[i]}
        #
        # Return final results
        #
        self.results = results

        # the pattern search has neither a proper return code or a log:
        return Bunch(rc=None, log=None)
