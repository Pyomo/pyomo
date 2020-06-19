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
# Test the pyomo.gdp transformations
#

import os
import sys
from os.path import abspath, dirname, normpath, join
from pyutilib.misc import import_file
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir,'..','..','..','examples', 'gdp'))

try:
    import new
except:
    import types as new

import pyutilib.th as unittest

from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
from pyomo.environ import SolverFactory, TransformationFactory

from six import iteritems

solvers = pyomo.opt.check_available_solvers('cplex', 'glpk','gurobi')


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
                            copy.__doc__ = attrs[key].__doc__ + \
                                           " (%s)" % copy.__name__
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
        exfile = import_file(join(exdir, 'jobshop.py'))
        m_jobshop = exfile.build_model()
        # This is awful, but it's the convention of the old method, so it will
        # work for now
        datafile = args[0]
        m = m_jobshop.create_instance(join(exdir, datafile))

        if 'preprocess' in kwds:
            transformation = kwds['preprocess']

        TransformationFactory('gdp.%s' % transformation).apply_to(m)
        m.write(join(currdir, '%s_result.lp' % self.problem),
                io_options={'symbolic_solver_labels': True})

        if self.solve:
            solver = 'glpk'
            if 'solver' in kwds:
                solver = kwds['solver']
            results = SolverFactory(solver).solve(m)
            m.solutions.store_to(results)
            results.write(filename=join(currdir, 'result.yml'))

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

    def test_bigm_jobshop_small(self):
        self.problem='test_bigm_jobshop_small'
        # Run the small jobshop example using the BigM transformation
        self.pyomo('jobshop-small.dat', preprocess='bigm')
        # ESJ: TODO: Right now the indicator variables have names they won't
        # have when they don't have to be reclassified. So I think this LP file
        # will need to change again.
        self.check( 'jobshop_small', 'bigm' )

    def test_bigm_jobshop_large(self):
        self.problem='test_bigm_jobshop_large'
        # Run the large jobshop example using the BigM transformation
        self.pyomo('jobshop.dat', preprocess='bigm')
        # ESJ: TODO: this LP file also will need to change with the
        # indicator variable change.
        self.check( 'jobshop_large', 'bigm' )

    # def test_bigm_constrained_layout(self):
    #     self.problem='test_bigm_constrained_layout'
    #     # Run the constrained layout example with the bigm transformation
    #     self.pyomo( join(exdir,'ConstrainedLayout.py'), 
    #                 join(exdir,'ConstrainedLayout_BigM.dat'), 
    #                 preprocess='bigm', solver='cplex')
    #     self.check( 'constrained_layout', 'bigm')

    def test_hull_jobshop_small(self):
        self.problem='test_hull_jobshop_small'
        # Run the small jobshop example using the Hull transformation
        self.pyomo('jobshop-small.dat', preprocess='hull')
        self.check( 'jobshop_small', 'hull' )

    def test_hull_jobshop_large(self):
        self.problem='test_hull_jobshop_large'
        # Run the large jobshop example using the Hull transformation
        self.pyomo('jobshop.dat', preprocess='hull')
        self.check( 'jobshop_large', 'hull' )

    @unittest.skip("cutting plane LP file tests are too fragile")
    @unittest.skipIf('gurobi' not in solvers, 'Gurobi solver not available')
    def test_cuttingplane_jobshop_small(self):
        self.problem='test_cuttingplane_jobshop_small'
        self.pyomo('jobshop-small.dat', preprocess='cuttingplane')
        self.check( 'jobshop_small', 'cuttingplane' )

    @unittest.skip("cutting plane LP file tests are too fragile")
    @unittest.skipIf('gurobi' not in solvers, 'Gurobi solver not available')
    def test_cuttingplane_jobshop_large(self):
        self.problem='test_cuttingplane_jobshop_large'
        self.pyomo('jobshop.dat', preprocess='cuttingplane')
        self.check( 'jobshop_large', 'cuttingplane' )


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
        if os.path.exists(join(currdir,self.problem+'_result.lp')):
            os.remove(join(currdir,self.problem+'_result.lp'))


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
                self.assertAlmostEqual(
                    val.get('Value', None),
                    ansObj[i].get(key,{}).get('Value', None),
                    6
                )
        # Clean up test files
        if os.path.exists(join(currdir,self.problem+'_result.lp')):
            os.remove(join(currdir,self.problem+'_result.lp'))


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'glpk' in solvers, "The 'glpk' executable is not available")
class Solve_GLPK(Solver, CommonTests):

    def pyomo(self,  *args, **kwds):
        kwds['solver'] = 'glpk'
        CommonTests.pyomo(self, *args, **kwds)


@unittest.skipIf(not yaml_available, "YAML is not available")
@unittest.skipIf(not 'cplex' in solvers, 
                 "The 'cplex' executable is not available")
class Solve_CPLEX(Solver, CommonTests):

    def pyomo(self,  *args, **kwds):
        kwds['solver'] = 'cplex'
        CommonTests.pyomo(self, *args, **kwds)


if __name__ == "__main__":
    unittest.main()
