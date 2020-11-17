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
from pyomo.core import TransformationFactory, Var, Set
from pyomo.common.collections import Bunch


@pyomo.opt.SolverFactory.register('bilevel_blp_local',
    doc='Local solver for continuous bilevel linear problems')
class BILEVEL_Solver3(pyomo.opt.OptSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'bilevel_blp_local'
        pyomo.opt.OptSolver.__init__(self,**kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()
        #
        # Transform the instance
        #
        xfrm = TransformationFactory('bilevel.linear_mpec')
        xfrm.apply_to(self._instance)
        xfrm = TransformationFactory('mpec.simple_nonlinear')
        xfrm.apply_to(self._instance, mpec_bound=self.options.get('mpec_bound',1e-7))
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            solver = 'glpk'

        # use the with block here so that deactivation of the
        # solver plugin always occurs thereby avoiding memory
        # leaks caused by plugins!
        with pyomo.opt.SolverFactory(solver) as opt:
            #
            self.results = []
            #
            # **NOTE: It would be better to override _presolve on the
            #         base class of this solver as you might be
            #         missing a number of keywords that were passed
            #         into the solve method (e.g., none of the
            #         io_options are getting relayed to the subsolver
            #         here).
            #
            self.results.append(opt.solve(self._instance,
                                          tee=self._tee,
                                          timelimit=self._timelimit))
        #
        # Load the result back into the original model
        #
        ##self._instance.load(self.results[0], ignore_invalid_labels=True)
        #
        stop_time = time.time()
        self.wall_time = stop_time - start_time
        #
        # Deactivate the block that contains the optimality conditions,
        # and reactivate SubModel
        #
        submodel = self._instance._transformation_data[
            'bilevel.linear_mpec'].submodel_cuid.find_component_on(
                self._instance)
        for (name, data) in submodel.component_map(active=False).items():
            if not isinstance(data,Var) and not isinstance(data,Set):
                data.activate()
        # TODO: delete this subblock
        self._instance._transformation_data[
            'bilevel.linear_mpec'].block_cuid.find_component_on(
                self._instance).deactivate()
        #
        # Return the sub-solver return condition value and log
        #
        return Bunch(rc=getattr(opt,'_rc', None), log=getattr(opt,'_log',None))

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
        #solv.status = self._glpk_get_solver_status()
        #solv.memory_used = "%d bytes, (%d KiB)" % (peak_mem, peak_mem/1024)
        solv.wallclock_time = self.wall_time
        cpu_ = []
        for res in self.results:
            if not getattr(res.solver, 'cpu_time', None) is None:
                cpu_.append( res.solver.cpu_time )
        if len(cpu_) > 0:
            solv.cpu_time = sum(cpu_)
        #
        # TODO: detect infeasibilities, etc
        #
        solv.termination_condition = pyomo.opt.TerminationCondition.optimal
        #
        # PROBLEM
        #
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
        # Uncache the instance
        #
        self._instance = None
        return results
