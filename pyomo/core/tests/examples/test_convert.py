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
import re
from os.path import abspath, dirname

currdir = dirname(abspath(__file__))+os.sep
scriptdir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))+os.sep

from pyutilib.misc import setup_redirect, reset_redirect
import pyutilib.subprocess
import pyutilib.th as unittest

import pyomo.scripting.convert as main

from six import StringIO

if os.path.exists(sys.exec_prefix+os.sep+'bin'+os.sep+'coverage'):
    executable=sys.exec_prefix+os.sep+'bin'+os.sep+'coverage -x '
else:
    executable=sys.executable

def filter_fn(line):
    return line.startswith('Disjunct')

_NL_diff_tol = 1e-9
_LP_diff_tol = 1e-9


class xTest(object):
#class Test(unittest.TestCase):

    def convert(self, cmd, type, **kwds):
        args=re.split('[ ]+',cmd)
        args.append("--symbolic-solver-labels") # for readability / quick inspections
        if 'file' in kwds:
            OUTPUT=kwds['file']
        else:
            OUTPUT=StringIO()
        setup_redirect(OUTPUT)
        os.chdir(currdir)
        if type == 'lp':
            output = main.pyomo2lp(list(args))
        else:
            output = main.pyomo2nl(list(args))
        reset_redirect()
        if not 'file' in kwds:
            return OUTPUT.getvalue()
        return output.retval, output.errorcode

    # in the convert tests, make sure everything is generating symbolic solver labels - aids debugging.
    def run_convert2nl(self, cmd, file=None):
        return pyutilib.subprocess.run('pyomo convert --format=nl --symbolic-solver-labels '+cmd, outfile=file)

    def run_convert2lp(self, cmd, file=None):
        return pyutilib.subprocess.run('pyomo convert --format=lp --symbolic-solver-labels '+cmd, outfile=file)

    def test1a(self):
        """Simple execution of 'convert2nl'"""
        ans,errorcode = self.convert('pmedian.py pmedian.dat', type='nl', file=currdir+'test1a.out')
        self.assertEquals(errorcode, 0)
        self.assertFileEqualsBaseline(currdir+'unknown.nl', currdir+'_pmedian.nl', tolerance=_NL_diff_tol)
        os.remove(currdir+'test1a.out')

    def test1b(self):
        """Simple execution of 'pyomo2nl'"""
        self.run_convert2nl('pmedian.py pmedian.dat', file=currdir+'test1b.out')
        self.assertFileEqualsBaseline(currdir+'unknown.nl', currdir+'_pmedian.nl', tolerance=_NL_diff_tol)
        os.remove(currdir+'test1b.out')

    def test2a(self):
        """Simple execution of 'convert2lp'"""
        ans,errorcode = self.convert('pmedian.py pmedian.dat', type='lp', file=currdir+'test2a.out')
        self.assertEquals(errorcode, 0)
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_pmedian.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'test2a.out')

    def test2b(self):
        """Simple execution of 'pyomo2lp'"""
        self.run_convert2lp('pmedian.py pmedian.dat', file=currdir+'test2b.out')
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_pmedian.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'test2b.out')

    def test3a(self):
        """Simple execution of 'convert2nl'"""
        ans,errorcode = self.convert('pmedian4.py', type='nl', file=currdir+'test3a.out')
        self.assertEquals(errorcode, 0)
        self.assertFileEqualsBaseline(currdir+'unknown.nl', currdir+'_pmedian4.nl', tolerance=_NL_diff_tol)
        os.remove(currdir+'test3a.out')

    def test3b(self):
        """Simple execution of 'pyomo2nl'"""
        self.run_convert2nl('pmedian4.py', file=currdir+'test3b.out')
        self.assertFileEqualsBaseline(currdir+'unknown.nl', currdir+'_pmedian4.nl', tolerance=_NL_diff_tol)
        os.remove(currdir+'test3b.out')

    def test4a(self):
        """Simple execution of 'convert2lp' with a concrete model"""
        ans,errorcode = self.convert('pmedian4.py', type='lp', file=currdir+'test4a.out')
        self.assertEquals(errorcode, 0)
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_pmedian4.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'test4a.out')

    def test4b(self):
        """Simple execution of 'pyomo2lp' with a concrete model"""
        self.run_convert2lp('pmedian4.py', file=currdir+'test4b.out')
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_pmedian4.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'test4b.out')

    def test_quadratic1(self):
        """Test examples/pyomo/quadratic/example1.py"""
        self.run_convert2lp(os.path.join(scriptdir,'examples','pyomo','quadratic','example1.py'), file=currdir+'quadratic1.out')
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_quadratic1.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'quadratic1.out')

    def test_quadratic2(self):
        """Test examples/pyomo/quadratic/example2.py"""
        self.run_convert2lp(os.path.join(scriptdir,'examples','pyomo','quadratic','example2.py'), file=currdir+'quadratic2.out')
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_quadratic2.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'quadratic2.out')

    def test_quadratic3(self):
        """Test examples/pyomo/quadratic/example3.py"""
        self.run_convert2lp(os.path.join(scriptdir,'examples','pyomo','quadratic','example3.py'), file=currdir+'quadratic3.out')
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_quadratic3.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'quadratic3.out')

    def test_quadratic4(self):
        """Test examples/pyomo/quadratic/example4.py"""
        self.run_convert2lp(os.path.join(scriptdir,'examples','pyomo','quadratic','example4.py'), file=currdir+'quadratic4.out')
        self.assertFileEqualsBaseline(currdir+'unknown.lp', currdir+'_quadratic4.lp', tolerance=_LP_diff_tol)
        os.remove(currdir+'quadratic4.out')


if __name__ == "__main__":
    unittest.main()
