#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test transformations for linear duality
#

import os
from os.path import abspath, dirname, normpath, join

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir,'..','..','..','examples','pyomo','core'))

import pyutilib.th as unittest

from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.scripting.util import cleanup
import pyomo.scripting.pyomo_main as main


from six import iteritems

solver = None
class CommonTests(object):

    solve = True

    def run_bilevel(self, *_args, **kwds):
        if self.solve:
            args = ['solve']
            _solver = kwds.get('solver','glpk')
            args.append('--solver=%s' % _solver)
            args.append('--save-results=result.yml')
            args.append('--results-format=json')
        else:
            args = ['convert']
        if 'transform' in kwds:
            args.append('--transform=%s' % kwds['transform'])
        args.append('-c')
        args.append('--symbolic-solver-labels')
        args.append('--file-determinism=2')

        if False:
            args.append('--stream-solver')
            args.append('--tempdir='+currdir)
            args.append('--keepfiles')
            args.append('--debug')
            args.append('--logging=verbose')

        args = args + list(_args)
        os.chdir(currdir)

        print('***')
        #print(' '.join(args))
        output = main.main(args)
        try:
            output = main.main(args)
        except:
            output = None
        cleanup()
        print('***')
        return output

    def check(self, problem, solver):
        pass

    def referenceFile(self, problem, solver):
        return join(currdir, problem+'.txt')

    def getObjective(self, fname):
        FILE = open(fname)
        data = yaml.load(FILE, **yaml_load_args)
        FILE.close()
        solutions = data.get('Solution', [])
        ans = []
        for x in solutions:
            ans.append(x.get('Objective', {}))
        return ans

    def updateDocStrings(self):
        for key in dir(self):
            if key.startswith('test'):
                getattr(self,key).__doc__ = " (%s)" % getattr(self,key).__name__

    def test_t5(self):
        self.problem='test_t5'
        self.run_bilevel(join(exdir,'t5.py'))
        self.check( 't5', 'linear_dual' )

    def test_t1(self):
        self.problem='test_t1'
        self.run_bilevel(join(exdir,'t1.py'))
        self.check( 't1', 'linear_dual' )


class Reformulate(unittest.TestCase, CommonTests):

    solve=False

    def tearDown(self):
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def run_bilevel(self,  *args, **kwds):
        args = list(args)
        args.append('--output='+self.problem+'_result.lp')
        kwds['transform'] = 'duality.linear_dual'
        CommonTests.run_bilevel(self, *args, **kwds)

    def referenceFile(self, problem, solver):
        return join(currdir, problem+"_"+solver+'.lp')

    def check(self, problem, solver):
        self.assertFileEqualsBaseline( join(currdir,self.problem+'_result.lp'),
                                           self.referenceFile(problem,solver), tolerance=1e-5 )


class Solver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def tearDown(self):
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))

    def check(self, problem, solver):
        refObj = self.getObjective(self.referenceFile(problem,solver))
        ansObj = self.getObjective(join(currdir,'result.yml'))
        self.assertEqual(len(refObj), len(ansObj))
        for i in range(len(refObj)):
            self.assertEqual(len(refObj[i]), len(ansObj[i]))
            for key,val in iteritems(refObj[i]):
                self.assertAlmostEqual(val['Value'], ansObj[i].get(key,None)['Value'], places=3)


class Solve_GLPK(Solver, CommonTests):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = pyomo.opt.check_available_solvers('glpk')

    def setUp(self):
        if (not yaml_available) or (not 'glpk' in solvers):
            self.skipTest("YAML is not available or "
                          "the 'glpk' executable is not available")

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'glpk'
        CommonTests.run_bilevel(self, *args, **kwds)


class Solve_CPLEX(Solver, CommonTests):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = pyomo.opt.check_available_solvers('cplex')

    def setUp(self):
        if (not yaml_available) or (not 'cplex' in solvers):
            self.skipTest("YAML is not available or "
                          "the 'cplex' executable is not available")

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'cplex'
        CommonTests.run_bilevel(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
