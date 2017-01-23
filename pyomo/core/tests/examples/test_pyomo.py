#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Test the Pyomo command-line interface
#

import re
import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.subprocess
import pyutilib.th as unittest
from pyutilib.misc import setup_redirect, reset_redirect

import pyomo.core
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers

from six import StringIO

try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

if os.path.exists(sys.exec_prefix+os.sep+'bin'+os.sep+'coverage'):
    executable=sys.exec_prefix+os.sep+'bin'+os.sep+'coverage -x '
else:
    executable=sys.executable

def filter_fn(line):
    tmp = line.strip()
    return tmp.startswith('Disjunct') or tmp.startswith('DEPRECATION') or tmp.startswith('DiffSet') or line.startswith('    ') or tmp.startswith("Differential") or tmp.startswith("DerivativeVar") or tmp.startswith("InputVar") or tmp.startswith('StateVar') or tmp.startswith('Complementarity')


_diff_tol = 1e-6

solvers = None
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = check_available_solvers('glpk')

    def pyomo(self, cmd, **kwds):
        if 'root' in kwds:
            OUTPUT=kwds['root']+'.out'
            results=kwds['root']+'.jsn'
            self.ofile = OUTPUT
        else:
            OUTPUT=StringIO()
            results='results.jsn'
        setup_redirect(OUTPUT)
        os.chdir(currdir)
        if type(cmd) is list:
            output = main.main(['solve', '--solver=glpk', '--results-format=json', '--save-results=%s' % results] + cmd, get_return=True)
        elif cmd.endswith('json') or cmd.endswith('yaml'):
            output = main.main(['solve', '--results-format=json', '--save-results=%s' % results] + [cmd], get_return=True)
        else:
            args=re.split('[ ]+',cmd)
            output = main.main(['solve', '--solver=glpk', '--results-format=json', '--save-results=%s' % results] + list(args), get_return=True)
        reset_redirect()
        if not 'root' in kwds:
            return OUTPUT.getvalue()
        return output

    def setUp(self):
        self.ofile = None
        if not 'glpk' in solvers:
            self.skipTest("GLPK is not installed")

    def tearDown(self):
        return
        if self.ofile and os.path.exists(self.ofile):
            return
            os.remove(self.ofile)
        if os.path.exists(currdir+'results.jsn'):
            return
            os.remove(currdir+'results.jsn')

    def run_pyomo(self, cmd, root=None):
        return pyutilib.subprocess.run('pyomo solve --solver=glpk --results-format=json --save-results=%s.jsn ' % (root) +cmd, outfile=root+'.out')

    def test1_simple_pyomo_execution(self):
        # Simple execution of 'pyomo'
        self.pyomo(currdir+'pmedian.py pmedian.dat', root=currdir+'test1')
        self.assertMatchesJsonBaseline(currdir+"test1.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test1.out')

    def test1a_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' in a subprocess
        self.run_pyomo(currdir+'pmedian.py pmedian.dat', root=currdir+'test1a')
        self.assertMatchesJsonBaseline(currdir+"test1a.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test1a.out')

    def test1b_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with a configuration file
        self.pyomo(currdir+'test1b.json', root=currdir+'test1')
        self.assertMatchesJsonBaseline(currdir+"test1.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test1.out')

    def test2_bad_model_name(self):
        # Run pyomo with bad --model-name option value
        self.pyomo('--model-name=dummy pmedian.py pmedian.dat', root=currdir+'test2')
        def filter2(line):
            return line.startswith('[') or line.startswith('DEPRECATION')
        self.assertFileEqualsBaseline(currdir+"test2.out", currdir+"test2.txt", filter=filter2)

    def test2b_bad_model_name(self):
        # Run pyomo with bad --model-name option value (configfile)
        self.pyomo(currdir+'test2b.json', root=currdir+'test2')
        def filter2(line):
            return line.startswith('[') or line.startswith('DEPRECATION')
        self.assertFileEqualsBaseline(currdir+"test2.out", currdir+"test2.txt", filter=filter2)

    def test3_missing_model_object(self):
        # Run pyomo with model that does not define model object
        self.pyomo('pmedian1.py pmedian.dat', root=currdir+'test3')
        def filter3(line):
            return line.startswith('[') or line.startswith('DEPRECATION')
        self.assertMatchesJsonBaseline(currdir+"test3.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test3.out')

    def test4_valid_modelname_option(self):
        # Run pyomo with good --model-name option value
        self.pyomo('--model-name=MODEL '+currdir+'pmedian1.py pmedian.dat', root=currdir+'test4')
        self.assertMatchesJsonBaseline(currdir+"test4.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test4.out')

    def test4b_valid_modelname_option(self):
        # Run pyomo with good 'object name' option value (configfile)
        self.pyomo(currdir+'test4b.json', root=currdir+'test4b')
        self.assertMatchesJsonBaseline(currdir+"test4b.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test4b.out')

    def test5_create_model_fcn(self):
        #"""Run pyomo with create_model function"""
        self.pyomo('pmedian2.py pmedian.dat', root=currdir+'test5')
        def filter5(line):
            return ("Writing model " in line) or ("Solver results file" in line) or \
                   line.startswith('[') or line.startswith('DEPRECATION')
        self.assertFileEqualsBaseline(currdir+"test5.out", currdir+"test5.txt",filter=filter5,tolerance=_diff_tol)
        os.remove(currdir+'test5.jsn')

    def test5b_create_model_fcn(self):
        # Run pyomo with create_model function (configfile)
        self.pyomo(currdir+'test5b.json', root=currdir+'test5')
        def filter5(line):
            return ("Writing model " in line) or ("Solver results file" in line) or \
                   line.startswith('[') or line.startswith('DEPRECATION')
        self.assertFileEqualsBaseline(currdir+"test5.out", currdir+"test5.txt",filter=filter5,tolerance=_diff_tol)
        os.remove(currdir+'test5.jsn')

    def test8_instanceonly_option(self):
        #"""Run pyomo with --instance-only option"""
        output = self.pyomo('--instance-only pmedian.py pmedian.dat', root=currdir+'test8')
        self.assertEqual(type(output.retval.instance), pyomo.core.AbstractModel)
        # Check that the results file was NOT created
        self.assertRaises(OSError, lambda: os.remove(currdir+'test8.jsn'))
        os.remove(currdir+'test8.out')

    def test8b_instanceonly_option(self):
        # Run pyomo with --instance-only option (configfile)
        output = self.pyomo(currdir+'test8b.json', root=currdir+'test8')
        self.assertEqual(type(output.retval.instance), pyomo.core.AbstractModel)
        # Check that the results file was NOT created
        self.assertRaises(OSError, lambda: os.remove(currdir+'test8.jsn'))
        os.remove(currdir+'test8.out')

    def test9_disablegc_option(self):
        #"""Run pyomo with --disable-gc option"""
        output = self.pyomo('--disable-gc pmedian.py pmedian.dat', root=currdir+'test9')
        self.assertEqual(type(output.retval.instance), pyomo.core.AbstractModel)
        os.remove(currdir+'test9.jsn')
        os.remove(currdir+'test9.out')

    def test9b_disablegc_option(self):
        # Run pyomo with --disable-gc option (configfile)
        output = self.pyomo(currdir+'test9b.json', root=currdir+'test9')
        self.assertEqual(type(output.retval.instance), pyomo.core.AbstractModel)
        os.remove(currdir+'test9.jsn')
        os.remove(currdir+'test9.out')

    def test12_output_option(self):
        #"""Run pyomo with --output option"""
        self.pyomo('--logfile=%s pmedian.py pmedian.dat' % (currdir+'test12.log'), root=currdir+'test12')
        self.assertMatchesJsonBaseline(currdir+"test12.jsn", currdir+"test12.txt",tolerance=_diff_tol)
        os.remove(currdir+'test12.log')
        os.remove(currdir+'test12.out')

    def test12b_output_option(self):
        # Run pyomo with --output option (configfile)
        self.pyomo(currdir+'test12b.json', root=currdir+'test12')
        self.assertMatchesJsonBaseline(currdir+"test12.jsn", currdir+"test12.txt",tolerance=_diff_tol)
        os.remove('test12b.log')
        os.remove(currdir+'test12.out')

    def test14_concrete_model_with_constraintlist(self):
        # Simple execution of 'pyomo' with a concrete model and constraint lists
        self.pyomo('pmedian4.py', root=currdir+'test14')
        self.assertMatchesJsonBaseline(currdir+"test14.jsn", currdir+"test14.txt",tolerance=_diff_tol)
        os.remove(currdir+'test14.out')

    def test14b_concrete_model_with_constraintlist(self):
        # Simple execution of 'pyomo' with a concrete model and constraint lists (configfile)
        self.pyomo('pmedian4.py', root=currdir+'test14')
        self.assertMatchesJsonBaseline(currdir+"test14.jsn", currdir+"test14.txt",tolerance=_diff_tol)
        os.remove(currdir+'test14.out')

    def test15_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(['--solver-options="mipgap=0.02 cuts="', currdir+'pmedian.py', 'pmedian.dat'], root=currdir+'test15')
        self.assertMatchesJsonBaseline(currdir+"test15.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test15.out')

    def test15b_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(currdir+'test15b.json', root=currdir+'test15b')
        self.assertMatchesJsonBaseline(currdir+"test15b.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test15b.out')

    def test15c_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(currdir+'test15c.json', root=currdir+'test15c')
        self.assertMatchesJsonBaseline(currdir+"test15c.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test15c.out')


@unittest.skipIf(not yaml_available, "YAML not available available")
class TestWithYaml(Test):

    def test15b_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(currdir+'test15b.yaml', root=currdir+'test15b')
        self.assertMatchesJsonBaseline(currdir+"test15b.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test15b.out')

    def test15c_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(currdir+'test15c.yaml', root=currdir+'test15c')
        self.assertMatchesJsonBaseline(currdir+"test15c.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test15c.out')


if __name__ == "__main__":
    unittest.main()
