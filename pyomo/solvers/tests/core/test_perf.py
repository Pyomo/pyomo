#
# Test the Pyomo command-line interface
#

import os
import gc
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep
datadir = os.path.normpath(currdir+'../../../../pyomo/examples/pyomo/p-median/')+os.sep

import pyutilib.th as unittest
import pyomo.scripting.convert
import pyomo.scripting.pyomo_command as main
import pyomo.opt
import pyomo.environ

solvers = pyomo.opt.check_available_solvers('cplex')

#def tearDownModule():
#    gc.collect()
#    for name in ['AbstractModel','ConcreteModel']:
#        try:
#            os.remove(currdir+name+'_backrefs.png')
#        except OSError:
#            pass
#    try:
#        import objgraph
#        for name in ['AbstractModel','ConcreteModel']:
#            objgraph.show_backrefs(objgraph.by_type(name),filename=currdir+name+'_backrefs.png')
#    except ImportError:
#        pass

@unittest.category('performance')
class Test(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        os.chdir(currdir)

    def tearDown(self):
        os.chdir(self.cwd)


#@unittest.category('performance')
class Test1(object):

    def test1(self):
        res = pyomo.scripting.convert.pyomo2nl(['--output',currdir+'test1.nl',datadir+'pmedian.py',datadir+'pmedian.dat'])
        if not os.path.exists(currdir+'test1.nl'):
            raise ValueError("Missing file test1.nl generated in test1")
        os.remove(currdir+'test1.nl')
        if res.errorcode:
            raise ValueError("pyomo2nl returned nonzero return code (%s)" % res.errorcode)
        if not res.retval.local.max_memory is None:
            self.recordTestData('maximum memory used', res.retval.local.max_memory)

    def test2(self):
        res = pyomo.scripting.convert.pyomo2lp(['--output',currdir+'test2.lp',datadir+'pmedian.py',datadir+'pmedian.dat'])
        if not os.path.exists(currdir+'test2.lp'):
            raise ValueError("Missing file test2.lp generated in test2")
        os.remove(currdir+'test2.lp')
        if res.errorcode:
            raise ValueError("pyomo2lp returned nonzero return code (%s)" % res.errorcode)
        if not res.retval.local.max_memory is None:
            self.recordTestData('maximum memory used', res.retval.local.max_memory)

@unittest.category('performance')
class Test2(Test):
    pass


@unittest.category('performance')
class Test3(Test):
    pass

@unittest.category('performance')
class Test4(Test):
    pass

@unittest.nottest
def nl_test(self, name):
    fname = currdir+name+'.nl'
    root = name.split('_')[-1]
    options = self.get_options(name)
    if os.path.exists(datadir+root+'.dat'):
        options.append(datadir+root+'.dat')
    res = pyomo.scripting.convert.pyomo2nl(['--output',fname,'-c']+options)
    if not os.path.exists(fname):
        raise ValueError("Missing file %s generated in test2" % fname)
    os.remove(fname)
    if res.errorcode:
        raise ValueError("pyomo2nl returned nonzero return code (%s)" % res.errorcode)
    if not res.retval.local.max_memory is None:
        self.recordTestData('maximum memory used', res.retval.local.max_memory)

@unittest.nottest
def lp_test(self, name):
    fname = currdir+name+'.lp'
    root = name.split('_')[-1]
    options = self.get_options(name)
    if os.path.exists(datadir+root+'.dat'):
        options.append(datadir+root+'.dat')
    res = pyomo.scripting.convert.pyomo2lp(['--output',fname,'-c']+options)
    if not os.path.exists(fname):
        raise ValueError("Missing file %s generated in test2" % fname)
    os.remove(fname)
    if res.errorcode:
        raise ValueError("pyomo2lp returned nonzero return code (%s)" % res.errorcode)
    if not res.retval.local.max_memory is None:
        self.recordTestData('maximum memory used', res.retval.local.max_memory)

@unittest.nottest
def lp_with_cplex_solve_test(self, name):
    root = name.split('_')[-1]
    options = self.get_options(name)
    if os.path.exists(datadir+root+'.dat'):
        options.append(datadir+root+'.dat')
    res=main.run(['--solver=cplex','-c'] + options)
    if res.errorcode:
        raise ValueError("pyomo returned nonzero return code (%s)" % res.errorcode)
    if not res.retval.local.max_memory is None:
        self.recordTestData('maximum memory used', res.retval.local.max_memory)

# add the unit tests...

for i in [6,7,8]:

    name = 'test'+str(i)

    # Standard label output variants
    Test2.add_fn_test(fn=nl_test, name='nl_pmedian.'+name, options=[datadir+'pmedian.py'])
    Test2.add_fn_test(fn=nl_test, name='nl-O_pmedian.'+name, options=['--disable-gc', datadir+'pmedian.py'] )

    # standard label output variants
    Test2.add_fn_test(fn=lp_test, name='lp_pmedian.'+name, options=[datadir+'pmedian.py'])
    Test2.add_fn_test(fn=lp_test, name='lp-O_pmedian.'+name, options=['--disable-gc', datadir+'pmedian.py'] )

    # symbolic label output variants
    Test2.add_fn_test(fn=lp_test, name='lp_symbolic_labels_pmedian.'+name, options=['--symbolic-solver-labels', datadir+'pmedian.py'])
    Test2.add_fn_test(fn=lp_test, name='lp-O_symbolic_labels_pmedian.'+name, options=['--symbolic-solver-labels', '--disable-gc', datadir+'pmedian.py'] )

for name in ['diagA100000', 'diagB100000', 'diagC100000', 'bilinearA100000', 'bilinearB100000', 'bilinearC100000']:

    # standard label output variants
    Test3.add_fn_test(fn=nl_test, name='nl_'+name, options=[currdir+'performance'+os.sep+name+'.py'])
    Test3.add_fn_test(fn=nl_test, name='nl_O_'+name, options=['--disable-gc',currdir+'performance'+os.sep+name+'.py'])
    Test3.add_fn_test(fn=lp_test, name='lp_'+name, options=[currdir+'performance'+os.sep+name+'.py'])
    Test3.add_fn_test(fn=lp_test, name='lp_O_'+name, options=['--disable-gc',currdir+'performance'+os.sep+name+'.py'])

    # symbolic label output variants
    Test3.add_fn_test(fn=nl_test, name='nl_symbolic_labels_'+name, options=["--symbolic-solver-labels", currdir+'performance'+os.sep+name+'.py'])
    Test3.add_fn_test(fn=nl_test, name='nl_O_symbolic_labels_'+name, options=["--symbolic-solver-labels", '--disable-gc',currdir+'performance'+os.sep+name+'.py'])
    Test3.add_fn_test(fn=lp_test, name='lp_symbolic_labels_'+name, options=["--symbolic-solver-labels", currdir+'performance'+os.sep+name+'.py'])
    Test3.add_fn_test(fn=lp_test, name='lp_O_symbolic_labels_'+name, options=["--symbolic-solver-labels", '--disable-gc',currdir+'performance'+os.sep+name+'.py'])

# added with-solve tests to identify potential issues with loading solutions (which we have previously observed).
# using cplex to ensure that the solves themselves are not the bottleneck.
if 'cplex' in solvers:
    for i in [6,7,8]:
        name = 'test'+str(i)
        Test4.add_fn_test(fn=lp_with_cplex_solve_test, name='lp_with_cplex_solve_pmedian.'+name, options=[datadir+'pmedian.py'])

if __name__ == "__main__":
    try:
        unittest.main()
    except SystemExit:
        pass
