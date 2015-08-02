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
from pyomo.core import TransformationFactory


class MPEC_Solver2(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('mpec_minlp', doc='MPEC solver transforms to a MINLP')

    def __init__(self, **kwds):
        kwds['type'] = 'mpec_minlp'
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
        xfrm = TransformationFactory('mpec.simple_disjunction')
        xfrm.apply_to(self._instance)

        xfrm = TransformationFactory('gdp.bigm')
        xfrm.apply_to(self._instance, default_bigM=self.options.get('bigM',10**6))
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:                     #pragma:nocover
            self.options.solver = solver = 'glpk'

        # use the with block here so that deactivation of the
        # solver plugin always occurs thereby avoiding memory
        # leaks caused by plugins!
        with pyomo.opt.SolverFactory(solver) as opt:
            #
            # **NOTE: It would be better to override _presolve on the
            #         base class of this solver as you might be
            #         missing a number of keywords that were passed
            #         into the solve method (e.g., none of the
            #         io_options are getting relayed to the subsolver
            #         here).
            #
            self.results = opt.solve(self._instance,
                                     tee=self._tee,
                                     timelimit=self._timelimit)
            #
            # Reclassify the Complementarity components
            #
            from pyomo.mpec import Complementarity
            for cuid in self._instance._transformation_data['mpec.simple_disjunction'].compl_cuids:
                cobj = cuid.find_component(self._instance)
                cobj.parent_block().reclassify_component_type(cobj, Complementarity)
            #
            # Transform the result back into the original model
            #
            ##self._instance.solutions.load_from(self.results, ignore_invalid_labels=True)
            #
            # Update timing
            #
            stop_time = time.time()
            self.wall_time = stop_time - start_time

            #
            # Return the sub-solver return condition value and log
            #
            return pyutilib.misc.Bunch(rc=getattr(opt,'_rc', None),
                                       log=getattr(opt,'_log',None))

    def _postsolve(self):
        #
        # SOLVER
        #
        solv = self.results.solver
        solv.name = self.options.subsolver
        solv.wallclock_time = self.wall_time
        #
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(self.results)
        #
        # Uncache the instance
        #
        self._instance = None
        #
        # Return the results
        #
        return self.results

