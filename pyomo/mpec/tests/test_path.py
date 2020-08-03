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
# Test the path solver
#

import os
from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir,'..','..','..','examples','mpec'))

import six
import pyutilib.th as unittest

from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup

from six import iteritems

solvers = pyomo.opt.check_available_solvers('path')

class CommonTests:

    solve = True
    solver='path'

    def run_solver(self, *_args, **kwds):
        if self.solve:
            args = ['solve']
            args.append('--solver='+self.solver)
            args.append('--save-results=result.yml')
            args.append('--results-format=yaml')
            args.append('--solver-options="lemke_start=automatic output_options=yes"')
        else:
            args = ['convert']
        args.append('-c')
        args.append('--symbolic-solver-labels')
        args.append('--file-determinism=2')

        if False:
            args.append('--stream-solver')
            args.append('--tempdir='+currdir)
            args.append('--keepfiles')
            args.append('--logging=debug')

        args = args + list(_args)
        os.chdir(currdir)

        print('***')
        print(' '.join(args))
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

    def test_munson1a(self):
        self.problem='test_munson1a'
        self.run_solver( join(exdir,'munson1a.py') )
        self.check( 'munson1a', self.solver )

    def test_munson1b(self):
        self.problem='test_munson1b'
        self.run_solver( join(exdir,'munson1b.py') )
        self.check( 'munson1b', self.solver )

    def test_munson1c(self):
        self.problem='test_munson1c'
        self.run_solver( join(exdir,'munson1c.py') )
        self.check( 'munson1c', self.solver )

    def test_munson1d(self):
        self.problem='test_munson1d'
        self.run_solver( join(exdir,'munson1d.py') )
        self.check( 'munson1d', self.solver )

    def check(self, problem, solver):
        refObj = self.getObjective(self.referenceFile(problem,solver))
        ansObj = self.getObjective(join(currdir,'result.yml'))
        self.assertEqual(len(refObj), len(ansObj))
        for i in range(len(refObj)):
            self.assertEqual(len(refObj[i]), len(ansObj[i]))
            if isinstance(refObj[i], six.string_types):
                continue
            for key,val in iteritems(refObj[i]):
                self.assertAlmostEqual(val['Value'], ansObj[i].get(key,None)['Value'], places=2)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'path' in solvers, "The 'path' executable is not available")
class Solve_PATH(unittest.TestCase, CommonTests):

    def tearDown(self):
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))

    def run_solver(self,  *args, **kwds):
        CommonTests.run_solver(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
