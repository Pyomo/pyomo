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
from pyomo.core import TransformationFactory, Var, ComponentUID, Block, Objective
import pyomo.opt
from pyomo.bilevel.components import SubModel
import pyomo.util


class BILEVEL_Solver1(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('bilevel_ld', doc='Solver for bilevel problems using linear duality')

    def __init__(self, **kwds):
        kwds['type'] = 'bilevel_ld'
        pyomo.opt.OptSolver.__init__(self,**kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()
        #
        # Cache the instance
        #
        xfrm = TransformationFactory('bilevel.linear_dual')
        xfrm.apply_to(self._instance)
        #
        # Apply an additional transformation to remap bilinear terms
        #
        if self.options.transform is None:
            xfrm = None
        else:
            xfrm = TransformationFactory(self.options.transform)
            xfrm.apply_to(self._instance)
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
            # Transform the result back into the original model
            #
            tdata = self._instance._transformation_data['bilevel.linear_dual']
            unfixed_cuids = set()
            # Copy variable values and fix them
            for vuid in tdata.fixed:
                for index_, data_ in vuid.find_component_on(self._instance).iteritems():
                    if not data_.fixed:
                        data_.value = self._instance.find_component(data_).value
                        data_.fixed = True
                        unfixed_cuids.add(ComponentUID(data_))
            # Reclassify the SubModel components and resolve
            for name_ in tdata.submodel:
                submodel = getattr(self._instance, name_)
                submodel.activate()
                dual_submodel = getattr(self._instance, name_+'_dual')
                dual_submodel.deactivate()
                pyomo.util.PyomoAPIFactory('pyomo.repn.compute_canonical_repn')({}, model=submodel)
                self._instance.reclassify_component_type(name_, Block)
                # use the with block here so that deactivation of the
                # solver plugin always occurs thereby avoiding memory
                # leaks caused by plugins!
                with pyomo.opt.SolverFactory(solver) as opt_inner:
                    #
                    # **NOTE: It would be better to override _presolve on the
                    #         base class of this solver as you might be
                    #         missing a number of keywords that were passed
                    #         into the solve method (e.g., none of the
                    #         io_options are getting relayed to the subsolver
                    #         here).
                    #
                    self.results.append(opt_inner.solve(self._instance,
                                                        tee=self._tee,
                                                        timelimit=self._timelimit,
                                                        select=None))
                    self._instance.solutions.select(0, ignore_fixed_vars=True)
                    data_.parent_component().parent_block().reclassify_component_type(name_, SubModel)
            # Unfix variables
            for vuid in tdata.fixed:
                for index_, data_ in vuid.find_component_on(self._instance).iteritems():
                    if ComponentUID(data_) in unfixed_cuids:
                        data_.fixed = False
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            # Reactivate top level objective
            for oname, odata in self._instance.component_map(Objective).items():
                odata.activate()
            #
            # Return the sub-solver return condition value and log
            #
            return pyutilib.misc.Bunch(rc=getattr(opt,'_rc', None),
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
        from pyomo.core import maximize
        ##if self._instance.sense == maximize:
            ##prob.sense = pyomo.opt.ProblemSense.maximize
        ##else:
            ##prob.sense = pyomo.opt.ProblemSense.minimize
        #
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        #
        # Uncache the instance
        #
        self._instance = None
        return results

