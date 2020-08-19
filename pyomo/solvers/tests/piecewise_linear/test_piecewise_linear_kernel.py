#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import json
import os
from os.path import dirname, abspath, join
thisDir = dirname( abspath(__file__) )

import pyutilib.th as unittest
from pyutilib.misc import import_file

from pyomo.kernel import SolverFactory, variable, maximize, minimize
from pyomo.solvers.tests.solvers import test_solver_cases

problems = ['convex_var',
            'concave_var',
            'piecewise_var',
            'step_var']

testing_solvers = {}
#testing_solvers['cplex','lp'] = False
testing_solvers['cplex','nl'] = False
#testing_solvers['ipopt','nl'] = False
#testing_solvers['cplex','python'] = False
#testing_solvers['_cplex_persistent','python'] = False
for _solver, _io in test_solver_cases():
    if (_solver, _io) in testing_solvers and \
        test_solver_cases(_solver, _io).available:
        testing_solvers[_solver, _io] = True

def createTestMethod(pName,problem,solver,writer,kwds):

    def testMethod(obj):

        if not testing_solvers[solver, writer]:
            obj.skipTest("Solver %s (interface=%s) is not available"
                         % (solver, writer))

        m = import_file(os.path.join(thisDir,
                                                   'kernel_problems',
                                                   problem),
                                      clear_cache=True)

        model = m.define_model(**kwds)

        opt = SolverFactory(solver, solver_io=writer)
        results = opt.solve(model)

        # non-recursive
        new_results = ((var.name, var.value)
                       for var in model.components(ctype=variable.ctype,
                                                   active=True,
                                                   descend_into=False))
        baseline_results = getattr(obj,problem+'_results')
        for name, value in new_results:
            if abs(baseline_results[name]-value) > 0.00001:
                raise IOError("Difference in baseline solution values and "
                              "current solution values using:\n" + \
                "Solver: "+solver+"\n" + \
                "Writer: "+writer+"\n" + \
                "Variable: "+name+"\n" + \
                "Solution: "+str(value)+"\n" + \
                "Baseline: "+str(baseline_results[name])+"\n")

    return testMethod

def assignTests(cls, problem_list):
    for solver,writer in testing_solvers:
        for PROBLEM in problem_list:
            aux_list = [{'simplify': True},
                        {'simplify': False}]
            for AUX in aux_list:
                for REPN in ['sos2','mc','inc','cc','dcc','dlog','log']:
                    for BOUND_TYPE in ['lb','ub','eq']:
                        for SENSE in [maximize, minimize]:
                            if not( ((BOUND_TYPE == 'lb') and (SENSE ==  maximize)) or \
                                    ((BOUND_TYPE == 'ub') and (SENSE ==  minimize)) or \
                                    ((REPN == 'mc') and ('step' in PROBLEM)) ):
                                kwds = {}
                                kwds['sense'] = SENSE
                                kwds['repn'] = REPN
                                kwds['bound'] = BOUND_TYPE
                                if SENSE == maximize:
                                    attrName = "test_{0}_{1}_{2}_{3}_{4}_{5}".format(PROBLEM,REPN,BOUND_TYPE,'maximize',solver,writer)
                                else:
                                    assert SENSE == minimize
                                    attrName = "test_{0}_{1}_{2}_{3}_{4}_{5}".format(PROBLEM,REPN,BOUND_TYPE,'minimize',solver,writer)
                                assert len(AUX) == 1
                                kwds.update(AUX)
                                attrName += '_simplify_'+str(AUX['simplify'])
                                setattr(cls,attrName,createTestMethod(attrName,PROBLEM,solver,writer,kwds))
                                with open(join(thisDir,'kernel_baselines',PROBLEM+'_baseline_results.json'),'r') as f:
                                    baseline_results = json.load(f)
                                    setattr(cls,PROBLEM+'_results',baseline_results)

class PiecewiseLinearKernelTest(unittest.TestCase):
    pass
assignTests(PiecewiseLinearKernelTest, problems)

if __name__ == "__main__":
    unittest.main()
