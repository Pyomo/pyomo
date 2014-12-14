#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

#
# Test transformations for linear duality
#

#try:
#    import new
#except:
#    import types as new
import os
import sys
import unittest
from os.path import abspath, dirname, normpath, join

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir,'..','..','..','examples','bilevel'))

from six import iteritems
import re
import pyutilib.services
import pyutilib.subprocess
import pyutilib.common
import pyutilib.th as unittest
from pyutilib.misc import setup_redirect, reset_redirect
try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

from pyomo.environ import *
import pyomo.opt
import pyomo.scripting.pyomo_command as pyomo_main
from pyomo.bilevel.plugins.driver import bilevel_exec
from pyomo.scripting.util import cleanup
from pyomo.util.plugin import ExtensionPoint

solver = pyomo.opt.load_solvers('cplex', 'glpk')


class CommonTests:

    def run_bilevel(self, *_args, **kwds):
        args = []
        args.append('-c')
        pproc = None
        if 'solver' in kwds:
            _solver = kwds.get('solver','glpk')
            args.append('--solver=bilevel_ld')
            args.append('--solver-options="solver=%s"' % _solver)
        elif 'preprocess' in kwds:
            pp = kwds['preprocess']
            if pp == 'linear_dual':
                import pyomo.bilevel.linear_dual
                pproc = pyomo.bilevel.linear_dual.transform
        args.append('--symbolic-solver-labels')
        args.append('--save-results=result.yml')
        args.append('--file-determinism=2')

        if False:
            args.append('--stream-solver')
            args.append('--tempdir='+currdir)
            args.append('--keepfiles')
            args.append('--debug')
            args.append('--verbose')

        args = args + list(_args)
        os.chdir(currdir)

        print('***')
        if pproc:
            pproc.activate()
            print("Activating " + kwds['preprocess'])
        #print(' '.join(args))
        #output = pyomo_main.run(args)
        try:
            if pproc:
                output = pyomo_main.run(args)
            else:
                output = bilevel_exec(args)
        except:
            output = None
        cleanup()
        if pproc:
            pproc.deactivate()
        print('***')
        return output

    def check(self, problem, solver):
        pass

    def referenceFile(self, problem, solver):
        return join(currdir, problem+'.txt')

    def getObjective(self, fname):
        FILE = open(fname)
        data = yaml.load(FILE)
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
        self.run_bilevel( join(exdir,'t5.py'))
        self.check( 't5', 'linear_dual' )

    def test_t1(self):
        self.problem='test_t1'
        self.run_bilevel( join(exdir,'t1.py'))
        self.check( 't1', 'linear_dual' )

    def Xtest_t2(self):
        self.problem='test_t2'
        self.run_bilevel( join(exdir,'t2.py'))
        self.check( 't2', 'linear_dual' )


class Reformulate(unittest.TestCase, CommonTests):

    def run_bilevel(self,  *args, **kwds):
        args = list(args)
        args.append('--save-model='+self.problem+'_result.lp')
        args.append('--instance-only')
        kwds['preprocess'] = 'linear_dual'
        CommonTests.run_bilevel(self, *args, **kwds)

    def referenceFile(self, problem, solver):
        return join(currdir, problem+"_"+solver+'.lp')

    def check(self, problem, solver):
        self.assertFileEqualsBaseline( join(currdir,self.problem+'_result.lp'),
                                           self.referenceFile(problem,solver), tolerance=1e-5 )


class Solver(unittest.TestCase):

    def check(self, problem, solver):
        refObj = self.getObjective(self.referenceFile(problem,solver))
        ansObj = self.getObjective(join(currdir,'result.yml'))
        self.assertEqual(len(refObj), len(ansObj))
        for i in range(len(refObj)):
            self.assertEqual(len(refObj[i]), len(ansObj[i]))
            for key,val in iteritems(refObj[i]):
                self.assertEqual(val['Id'], ansObj[i].get(key,None)['Id'])
                self.assertAlmostEqual(val['Value'], ansObj[i].get(key,None)['Value'], places=3)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(solver['glpk'] is None, "The 'glpk' executable is not available")
class Solve_GLPK(Solver, CommonTests):

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'glpk'
        CommonTests.run_bilevel(self, *args, **kwds)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
class Solve_CPLEX(Solver, CommonTests):

    def run_bilevel(self,  *args, **kwds):
        kwds['solver'] = 'cplex'
        CommonTests.run_bilevel(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
