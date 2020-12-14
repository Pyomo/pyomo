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
from pyomo.core import TransformationFactory, Var, ComponentUID, Block, Objective, Set
import pyomo.opt
import pyomo.common
from pyomo.common.collections import Bunch


@pyomo.opt.SolverFactory.register('bilevel_ld',
    doc='Solver for bilevel problems using linear duality')
class BILEVEL_Solver1(pyomo.opt.OptSolver):

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
        # Verify whether the objective is linear
        #
        nonlinear=False
        for odata in self._instance.component_objects(Objective, active=True):
            nonlinear = odata.expr.polynomial_degree() != 1
            # Stop after the first objective
            break
        #
        # Apply an additional transformation to remap bilinear terms
        #
        if nonlinear:
            gdp_xfrm = TransformationFactory("gdp.bilinear")
            gdp_xfrm.apply_to(self._instance)
            mip_xfrm = TransformationFactory("gdp.bigm")
            mip_xfrm.apply_to(self._instance, bigM=self.options.get('bigM',100000))
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
            #print("POST-SOLVE - BEGIN")
            #self._instance.write("tmp.lp", io_options={"symbolic_solver_labels":True})
            #self._instance.pprint()
            #self._instance.display()
            #print("POST-SOLVE - END")
            #
            # If the problem was bilinear, then reactivate the original data
            #
            if nonlinear:
                i = 0
                for v in self._instance.bilinear_data_.vlist.itervalues():
                    #print(v)
                    #print(v.name)
                    #print(type(v))
                    #print(v.value)
                    if abs(v.value) <= 1e-7:
                        self._instance.bilinear_data_.vlist_boolean[i] = 0
                    else:
                        self._instance.bilinear_data_.vlist_boolean[i] = 1
                    i = i + 1
                #
                self._instance.bilinear_data_.deactivate()
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
                for (name, data) in submodel.component_map(active=False).items():
                    if not isinstance(data,Var) and not isinstance(data,Set):
                        data.activate()
                dual_submodel = getattr(self._instance, name_+'_dual')
                dual_submodel.deactivate()
                pyomo.common.PyomoAPIFactory('pyomo.repn.compute_standard_repn')({}, model=submodel)
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
                    results = opt_inner.solve(self._instance, tee=self._tee, timelimit=self._timelimit)
                                                        #select=None)
            # Unfix variables
            for vuid in tdata.fixed:
                for index_, data_ in vuid.find_component_on(self._instance).iteritems():
                    if ComponentUID(data_) in unfixed_cuids:
                        data_.fixed = False
            #
            self._instance.solutions.select(0, ignore_fixed_vars=True)
            self.results.append(results)
            #
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            self.results_obj = self._setup_results_obj()
            #
            # Reactivate top level objective
            # and reclassify the submodel
            #
            for oname, odata in self._instance.component_map(Objective).items():
                odata.activate()
            # TODO: rework the Block logic to allow for searching SubModel objects for variables, etc.
            #data_.parent_component().parent_block().reclassify_component_type(name_, SubModel)
            #
            # Return the sub-solver return condition value and log
            #
            return Bunch(rc=getattr(opt,'_rc', None),
                                       log=getattr(opt,'_log',None))

    def _postsolve(self):
        #
        # Uncache the instance
        #
        self._instance = None
        #
        # Return the results object
        #
        return self.results_obj

    def _setup_results_obj(self):
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
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        return results
