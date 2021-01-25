#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import time
import pyomo.opt
from pyomo.opt import SolverFactory
from pyomo.core import TransformationFactory
from pyomo.common.collections import Bunch


@SolverFactory.register('mpec_nlp', doc='MPEC solver that optimizes a nonlinear transformation')
class MPEC_Solver1(pyomo.opt.OptSolver):


    def __init__(self, **kwds):
        kwds['type'] = 'mpec_nlp'
        pyomo.opt.OptSolver.__init__(self,**kwds)
        self._metasolver = True

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
        xfrm = TransformationFactory('mpec.simple_nonlinear')
        xfrm.apply_to(self._instance)
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:                 #pragma:nocover
            self.options.solver = solver = 'ipopt'

        # use the with block here so that deactivation of the
        # solver plugin always occurs thereby avoiding memory
        # leaks caused by plugins!
        with pyomo.opt.SolverFactory(solver) as opt:
            self.results = []
            epsilon_final = self.options.get('epsilon_final', 1e-7)
            epsilon = self.options.get('epsilon_initial', epsilon_final)
            while (True):
                self._instance.mpec_bound.value = epsilon
                #
                # **NOTE: It would be better to override _presolve on the
                #         base class of this solver as you might be
                #         missing a number of keywords that were passed
                #         into the solve method (e.g., none of the
                #         io_options are getting relayed to the subsolver
                #         here).
                #
                res = opt.solve(self._instance,
                                tee=self._tee,
                                timelimit=self._timelimit)
                self.results.append(res)
                epsilon /= 10.0
                if epsilon < epsilon_final:
                    break
            #
            # Reclassify the Complementarity components
            #
            from pyomo.mpec import Complementarity
            for cuid in self._instance._transformation_data['mpec.simple_nonlinear'].compl_cuids:
                cobj = cuid.find_component_on(self._instance)
                cobj.parent_block().reclassify_component_type(cobj, Complementarity)
            #
            # Update timing
            #
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            #
            # Return the sub-solver return condition value and log
            #
            return Bunch(rc=getattr(opt,'_rc', None),
                                       log=getattr(opt,'_log',None))

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
        for res in self.results:                    #pragma:nocover
            if not getattr(res.solver, 'cpu_time', None) is None:
                cpu_.append( res.solver.cpu_time )
        if len(cpu_) > 0:                           #pragma:nocover
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
        self._instance.solutions.store_to(results)
        #
        # Uncache the instance and return the results
        #
        self._instance = None
        return results
