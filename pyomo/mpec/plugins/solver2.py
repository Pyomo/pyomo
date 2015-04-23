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

class MPEC_Solver2(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('mpec_lg', doc='Global solver for linear MPEC problems')

    def __init__(self, **kwds):
        kwds['type'] = 'mpec_lg'
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
        instance = self._instance.transform('mpec.simple_disjunction')
        instance2 = instance.transform('gdp.bigm', default_bigM=self.options.get('bigM',10^6))
        #
        # Solve with a specified solver
        #
        solver = self.options.solver
        if not self.options.solver:
            self.options.solver = solver = 'glpk'
        opt = pyomo.opt.SolverFactory(solver)
        self.results = opt.solve(instance2,
                                tee=self.tee,
                                timelimit=self._timelimit)
        #
        # Transform the result back into the original model
        #
        self._instance.load(self.results)
        self.results = instance2.update_results(self.results)
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
        return self.results

