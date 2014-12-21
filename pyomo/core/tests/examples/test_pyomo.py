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

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import re
from six import StringIO
import pyutilib.services
import pyutilib.subprocess
import pyutilib.th as unittest
from pyutilib.misc import setup_redirect, reset_redirect
import pyomo.environ
import pyomo.core
import pyomo.scripting.pyomo_command as main

if os.path.exists(sys.exec_prefix+os.sep+'bin'+os.sep+'coverage'):
    executable=sys.exec_prefix+os.sep+'bin'+os.sep+'coverage -x '
else:
    executable=sys.executable

def filter_fn(line):
    #print line
    tmp = line.strip()
    #print line.startswith('    ')
    return tmp.startswith('Disjunct') or tmp.startswith('DEPRECATION') or tmp.startswith('DiffSet') or line.startswith('    ') or tmp.startswith("Differential") or tmp.startswith("DerivativeVar") or tmp.startswith("InputVar") or tmp.startswith('StateVar') or tmp.startswith('Complementarity')


_diff_tol = 1e-6

@unittest.skipIf(pyutilib.services.registered_executable("glpsol") is None, "The 'glpsol' executable is not available")
class Test(unittest.TestCase):

    def pyomo(self, cmd, **kwds):
        args=re.split('[ ]+',cmd)
        if 'root' in kwds:
            OUTPUT=kwds['root']+'.out'
            results=kwds['root']+'.jsn'
            self.ofile = OUTPUT
        else:
            OUTPUT=StringIO()
            results='results.jsn'
        setup_redirect(OUTPUT)
        os.chdir(currdir)
        output = main.run(['--json', '--save-results=%s' % results] + list(args))
        reset_redirect()
        if not 'root' in kwds:
            return OUTPUT.getvalue()
        return output

    def setUp(self):
        self.ofile = None

    def tearDown(self):
        return
        if self.ofile and os.path.exists(self.ofile):
            return
            os.remove(self.ofile)
        if os.path.exists(currdir+'results.jsn'):
            return
            os.remove(currdir+'results.jsn')

    def run_pyomo(self, cmd, root=None):
        return pyutilib.subprocess.run('pyomo solve --json --save-results=%s.jsn ' % (root) +cmd, outfile=root+'.out')

    def test1_simple_pyomo_execution(self):
        #"""Simple execution of 'pyomo'"""
        self.pyomo(currdir+'pmedian.py pmedian.dat', root=currdir+'test1')
        self.assertMatchesJsonBaseline(currdir+"test1.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test1.out')

    def test1a_simple_pyomo_execution(self):
        #"""Simple execution of 'pyomo'"""
        self.run_pyomo(currdir+'pmedian.py pmedian.dat', root=currdir+'test1a')
        self.assertMatchesJsonBaseline(currdir+"test1a.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test1a.out')

    def test2_bad_model_name(self):
        #"""Run pyomo with bad --model-name option value"""
        self.pyomo('--model-name=dummy pmedian.py pmedian.dat', root=currdir+'test2')
        def filter2(line):
            return line.startswith('[') or line.startswith('DEPRECATION')
        self.assertFileEqualsBaseline(currdir+"test2.out", currdir+"test2.txt", filter=filter2)

    def test3_missing_model_object(self):
        #"""Run pyomo with model that does not define model object"""
        self.pyomo('pmedian1.py pmedian.dat', root=currdir+'test3')
        def filter3(line):
            return line.startswith('[') or line.startswith('DEPRECATION')
        self.assertFileEqualsBaseline(currdir+"test3.out", currdir+"test3.txt", filter=filter3)

    def test4_valid_modelname_option(self):
        #"""Run pyomo with good --model-name option value"""
        self.run_pyomo('-k -l --model-name=MODEL '+currdir+'pmedian1.py pmedian.dat', root=currdir+'test4')
        self.assertMatchesJsonBaseline(currdir+"test4.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test4.out')

    def test5_create_model_fcn(self):
        #"""Run pyomo with create_model function"""
        self.pyomo('pmedian2.py pmedian.dat', root=currdir+'test5')
        def filter5(line):
            return ("Writing model " in line) or ("Solver results file" in line) or \
                   line.startswith('[') or line.startswith('DEPRECATION')
        self.assertFileEqualsBaseline(currdir+"test5.out", currdir+"test5.txt",filter=filter5,tolerance=_diff_tol)
        os.remove(currdir+'test5.jsn')

    def test6_help_components_option(self):
        #"""Run pyomo with help-components option"""
        self.pyomo('--help-components', root=currdir+'test6')
        self.assertFileEqualsBaseline(currdir+"test6.out", currdir+"test6.txt", filter=filter_fn)

    def Xtest7_help_option(self):
        #"""Run pyomo with help option"""
        self.pyomo('--help', root=currdir+'test7')
        self.assertMatchesJsonBaseline(currdir+"test7.jsn", currdir+"test7.txt")

    def test8_instanceonly_option(self):
        #"""Run pyomo with --instance-only option"""
        output = self.pyomo('--instance-only pmedian.py pmedian.dat', root=currdir+'test8')
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

    def Xtest10_verbose_option(self):
        #"""Run pyomo with --verbose option"""
        def filter10(line):
            return ("Writing model " in line) or ("Solver results file" in line) or \
                   line.startswith('DEBUG:') or line.startswith('INFO:') or \
                   line.startswith('[') or line.endswith('cpxlp') or line.startswith('DEPRECATION') or \
                    line.startswith('WARNING')
        self.pyomo('-v pmedian.py pmedian.dat', root=currdir+'test10')
        self.assertFileEqualsBaseline(currdir+"test10.out", currdir+"test10.txt", filter10)
        os.remove(currdir+'test10.jsn')

    def Xtest11_debug_generate_option(self):
        #"""Run pyomo with --debug=generate option"""
        self.pyomo('--debug=generate pmedian.py pmedian.dat', root=currdir+'test11')
        self.assertFileEqualsBaseline(currdir+"test11.jsn", currdir+"test11.txt")

    def test12_output_option(self):
        #"""Run pyomo with --output option"""
        def Xfilter12(line):
            return ("Writing model " in line) or line.startswith('DEPRECATION')
        self.pyomo('--output=%s pmedian.py pmedian.dat' % (currdir+'test12.log'), root=currdir+'test12')
        self.assertMatchesJsonBaseline(currdir+"test12.jsn", currdir+"test12.txt",tolerance=_diff_tol)
        os.remove(currdir+'test12.log')
        os.remove(currdir+'test12.out')

    def Xtest13_pyomo_with_implicit_rules(self):
        #"""Simple execution of 'pyomo' with implicit rules"""
        self.pyomo('pmedian3.py pmedian.dat', root=currdir+'test13')
        self.assertMatchesJsonBaseline(currdir+"test13.jsn", currdir+"test1.txt",tolerance=_diff_tol)
        os.remove(currdir+'test13.out')

    def test14_concrete_model_with_constraintlist(self):
        #"""Simple execution of 'pyomo' with a concrete model and constraint lists"""
        self.pyomo('pmedian4.py', root=currdir+'test14')
        self.assertMatchesJsonBaseline(currdir+"test14.jsn", currdir+"test14.txt",tolerance=_diff_tol)
        os.remove(currdir+'test14.out')


if __name__ == "__main__":
    unittest.main()
