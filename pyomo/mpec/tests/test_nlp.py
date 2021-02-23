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
# Test the mpec_nlp solver
#

import os
from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'mpec'))

import pyutilib.th as unittest

from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup

from six import iteritems

solvers = pyomo.opt.check_available_solvers('ipopt')


class CommonTests:

    solve = True
    solver='mpec_nlp'

    def run_solver(self, *_args, **kwds):
        if self.solve:
            args = ['solve']
            if 'solver' in kwds:
                _solver = kwds.get('solver','glpk')
                args.append('--solver='+self.solver)
                args.append('--solver-options="solver=%s"' % _solver)
            args.append('--save-results=result.yml')
            args.append('--results-format=yaml')
        else:
            args = ['convert']
        args.append('-c')

        # These were being ignored by the solvers for this package,
        # which now causes a helpful error message.
        #args.append('--symbolic-solver-labels')
        #args.append('--file-determinism=2')

        if False:
            args.append('--stream-solver')
            args.append('--tempdir='+currdir)
            args.append('--keepfiles')
            args.append('--logging=debug')

        args = args + list(_args)
        os.chdir(currdir)

        print('***')
        #print(' '.join(args))
        try:
            output = pyomo_main.main(args)
        except SystemExit:
            output = None
        except:
            output = None
            raise
        cleanup()
        print('***')
        return output

    def referenceFile(self, problem, solver):
        return join(currdir, problem+'.txt')

    def getObjective(self, fname):
        FILE = open(fname,'r')
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

    def test_linear1(self):
        self.problem='test_linear1'
        self.run_solver( join(exdir,'linear1.py') )
        self.check( 'linear1', self.solver )

    def test_bard1(self):
        self.problem='test_bard1'
        self.run_solver( join(exdir,'bard1.py') )
        self.check( 'bard1', self.solver )

    def test_scholtes4(self):
        self.problem='test_scholtes4'
        self.run_solver( join(exdir,'scholtes4.py') )
        self.check( 'scholtes4', self.solver )

    def check(self, problem, solver):
        refObj = self.getObjective(self.referenceFile(problem,solver))
        ansObj = self.getObjective(join(currdir,'result.yml'))
        self.assertEqual(len(refObj), len(ansObj))
        for i in range(len(refObj)):
            self.assertEqual(len(refObj[i]), len(ansObj[i]))
            for key,val in iteritems(refObj[i]):
                self.assertAlmostEqual(val['Value'], ansObj[i].get(key,None)['Value'], places=2)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'ipopt' in solvers, "The 'ipopt' executable is not available")
class Solve_IPOPT(unittest.TestCase, CommonTests):

    def tearDown(self):
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))

    def run_solver(self,  *args, **kwds):
        kwds['solver'] = 'ipopt'
        CommonTests.run_solver(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
