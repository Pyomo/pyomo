#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Test Pyomo models with OpenOpt solvers
#

import re
import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.subprocess
import pyutilib.th as unittest
from pyutilib.misc import setup_redirect, reset_redirect

import pyomo.scripting.pyomo_main as main

from six import StringIO

try:
    import FuncDesigner
    FD_available=True
except:
    FD_available=False
try:
    import openopt
    OO_available=True
except:
    OO_available=False


def filter_fn(line):
    #print line
    tmp = line.strip()
    #print line.startswith('    ')
    return tmp.startswith('Disjunct') or tmp.startswith('DEPRECATION') or tmp.startswith('DiffSet') or line.startswith('    ') or tmp.startswith("Differential") or tmp.startswith("DerivativeVar") or tmp.startswith("InputVar") or tmp.startswith('StateVar') or tmp.startswith('Complementarity')


_diff_tol = 1e-6


@unittest.skipUnless(FD_available, "FuncDesigner module required")
@unittest.skipUnless(OO_available, "OpenOpt module required")
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

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
        output = main.main(['--results-format=json', '-c', '--stream-solver', '--save-results=%s' % results, '--solver=%s' % kwds['solver']] + list(args))
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
        return pyutilib.subprocess.run('pyomo solve --results-format=json --save-results=%s.jsn ' % (root) +cmd, outfile=root+'.out')

    def test1(self):
        self.pyomo(currdir+'test1.py', root=currdir+'test1', solver='openopt:ralg')
        self.assertMatchesJsonBaseline(currdir+"test1.jsn", currdir+"test1.txt", tolerance=_diff_tol)
        os.remove(currdir+'test1.out')


if __name__ == "__main__":
    unittest.main()
