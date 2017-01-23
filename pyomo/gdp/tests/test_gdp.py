#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Test the pyomo.gdp transformations
#

import os
import sys
from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir,'..','..','..','examples', 'gdp'))

try:
    import new
except:
    import types as new

import pyutilib.th as unittest

import pyomo.opt
import pyomo.scripting.pyomo_main as main
from pyomo.environ import *

from six import iteritems

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

solvers = pyomo.opt.check_available_solvers('cplex', 'glpk')


if False:
    if os.path.exists(sys.exec_prefix+os.sep+'bin'+os.sep+'coverage'):
        executable=sys.exec_prefix+os.sep+'bin'+os.sep+'coverage -x '
    else:
        executable=sys.executable

    def copyfunc(func):
        return new.function(func.__code__, func.func_globals, func.func_name,
                            func.func_defaults, func.func_closure)

    class Labeler(type):
        def __new__(meta, name, bases, attrs):
            for key in attrs.keys():
                if key.startswith('test_'):
                    for base in bases:
                        original = getattr(base, key, None)
                        if original is not None:
                            copy = copyfunc(original)
                            copy.__doc__ = attrs[key].__doc__ + " (%s)" % copy.__name__
                            attrs[key] = copy
                            break
            for base in bases:
                for key in dir(base):
                    if key.startswith('test_') and key not in attrs:
                        original = getattr(base, key)
                        copy = copyfunc(original)
                        copy.__doc__ = original.__doc__ + " (%s)" % name
                        attrs[key] = copy
            return type.__new__(meta, name, bases, attrs)


class CommonTests:
    #__metaclass__ = Labeler

    solve=True

    def pyomo(self, *args, **kwds):
        if self.solve:
            args=['solve']+list(args)
            if 'solver' in kwds:
                args.append('--solver='+kwds['solver'])
            else:
                args.append('--solver=glpk')
            args.append('--save-results=result.yml')
        else:
            args=['convert']+list(args)
        if 'preprocess' in kwds:
            pp = kwds['preprocess']
            if pp == 'bigm':
                args.append('--transform=gdp.bigm')
            elif pp == 'chull':
                args.append('--transform=gdp.chull')
        args.append('-c')
        args.append('--symbolic-solver-labels')
        os.chdir(currdir)

        print('***')
        #if pproc is not None:
        #    pproc.activate()
        #    print("Activating " + kwds['preprocess'])
        #else:
        #    print("ERROR: no transformation activated: " + pp)
        print(' '.join(args))
        output = main.main(args)
        #if pproc is not None:
        #    pproc = None
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

    def test_bigm_jobshop_small(self):
        self.problem='test_bigm_jobshop_small'
        # Run the small jobshop example using the BigM transformation
        self.pyomo( join(exdir,'jobshop.py'), join(exdir,'jobshop-small.dat'),
                    preprocess='bigm' )
        self.check( 'jobshop_small', 'bigm' )

    def test_bigm_jobshop_large(self):
        self.problem='test_bigm_jobshop_large'
        # Run the large jobshop example using the BigM transformation
        self.pyomo( join(exdir,'jobshop.py'), join(exdir,'jobshop.dat'),
                    preprocess='bigm')
        self.check( 'jobshop_large', 'bigm' )

    def test_chull_jobshop_small(self):
        self.problem='test_chull_jobshop_small'
        # Run the small jobshop example using the CHull transformation
        self.pyomo( join(exdir,'jobshop.py'), join(exdir,'jobshop-small.dat'),
                    preprocess='chull')
        self.check( 'jobshop_small', 'chull' )

    def test_chull_jobshop_large(self):
        self.problem='test_chull_jobshop_large'
        # Run the large jobshop example using the CHull transformation
        self.pyomo( join(exdir,'jobshop.py'), join(exdir,'jobshop.dat'),
                    preprocess='chull')
        self.check( 'jobshop_large', 'chull' )


class Reformulate(unittest.TestCase, CommonTests):

    solve=False

    def tearDown(self):
        if os.path.exists(os.path.join(currdir,'result.yml')):
            os.remove(os.path.join(currdir,'result.yml'))

    def pyomo(self,  *args, **kwds):
        args = list(args)
        args.append('--output='+self.problem+'_result.lp')
        CommonTests.pyomo(self, *args, **kwds)

    def referenceFile(self, problem, solver):
        return join(currdir, problem+"_"+solver+'.lp')

    def check(self, problem, solver):
        self.assertFileEqualsBaseline( join(currdir,self.problem+'_result.lp'),
                                           self.referenceFile(problem,solver) )


class Solver(unittest.TestCase):

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
                self.assertEqual(val, ansObj[i].get(key,None))


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'glpk' in solvers, "The 'glpk' executable is not available")
class Solve_GLPK(Solver, CommonTests):

    def pyomo(self,  *args, **kwds):
        kwds['solver'] = 'glpk'
        CommonTests.pyomo(self, *args, **kwds)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'cplex' in solvers, "The 'cplex' executable is not available")
class Solve_CPLEX(Solver, CommonTests):

    def pyomo(self,  *args, **kwds):
        kwds['solver'] = 'cplex'
        CommonTests.pyomo(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
