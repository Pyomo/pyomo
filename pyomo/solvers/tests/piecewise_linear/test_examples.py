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
# Test the Pyomo command-line interface
#

from filecmp import cmp
import os
from os.path import abspath, dirname, join
currdir = dirname(abspath(__file__))+os.sep
scriptdir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))+os.sep
scriptdir = join(scriptdir,'examples','pyomo','piecewise')

import pyomo.common.unittest as unittest

import pyomo.scripting.convert as convert

_NL_diff_tol = 1e-9
_LP_diff_tol = 1e-9

class Test(unittest.TestCase):

    def run_convert2nl(self, name):
        os.chdir(currdir)
        return convert.pyomo2nl(['--symbolic-solver-labels'
                                 ,join(scriptdir,name)])

    def run_convert2lp(self, name):
        os.chdir(currdir)
        return convert.pyomo2lp(['--symbolic-solver-labels',join(scriptdir,name)])

    def test_step_lp(self):
        """Test examples/pyomo/piecewise/step.py"""
        self.run_convert2lp('step.py')
        self.assertTrue(cmp(join(currdir,'unknown.lp'), currdir+'step.lp'))

    def test_step_nl(self):
        """Test examples/pyomo/piecewise/step.py"""
        self.run_convert2nl('step.py')
        self.assertTrue(cmp(join(currdir,'unknown.nl'), currdir+'step.nl'))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_nonconvex_lp(self):
        """Test examples/pyomo/piecewise/nonconvex.py"""
        self.run_convert2lp('nonconvex.py')
        self.assertTrue(cmp(join(currdir,'unknown.lp'), currdir+'nonconvex.lp'))

    def test_nonconvex_nl(self):
        """Test examples/pyomo/piecewise/nonconvex.py"""
        self.run_convert2nl('nonconvex.py')
        self.assertTrue(cmp(join(currdir,'unknown.nl'), currdir+'nonconvex.nl'))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_convex_lp(self):
        """Test examples/pyomo/piecewise/convex.py"""
        self.run_convert2lp('convex.py')
        self.assertTrue(cmp(join(currdir,'unknown.lp'), currdir+'convex.lp'))

    def test_convex_nl(self):
        """Test examples/pyomo/piecewise/convex.py"""
        self.run_convert2nl('convex.py')
        self.assertTrue(cmp(join(currdir,'unknown.nl'), currdir+'convex.nl'))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_indexed_lp(self):
        """Test examples/pyomo/piecewise/indexed.py"""
        self.run_convert2lp('indexed.py')
        with open(join(currdir,'unknown.lp'), 'r') as f1, \
            open(currdir+'indexed.lp', 'r') as f2:
                f1_contents = list(filter(None, f1.read().split()))
                f2_contents = list(filter(None, f2.read().split()))
                for item1, item2 in zip(f1_contents, f2_contents):
                    try:
                        self.assertEqual(item1, item2)
                    except:
                        self.assertAlmostEqual(float(item1), float(item2))

    def test_indexed_nl(self):
        """Test examples/pyomo/piecewise/indexed.py"""
        self.run_convert2nl('indexed.py')
        with open(join(currdir,'unknown.nl'), 'r') as f1, \
            open(currdir+'indexed.nl', 'r') as f2:
                f1_contents = list(filter(None, f1.read().split()))
                f2_contents = list(filter(None, f2.read().split()))
                for item1, item2 in zip(f1_contents, f2_contents):
                    try:
                        self.assertEqual(item1, item2)
                    except:
                        self.assertAlmostEqual(float(item1), float(item2))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_indexed_nonlinear_nl(self):
        """Test examples/pyomo/piecewise/indexed_nonlinear.py"""
        self.run_convert2nl('indexed_nonlinear.py')
        self.assertTrue(cmp(join(currdir,'unknown.nl'), currdir+'indexed_nonlinear.nl'))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))


if __name__ == "__main__":
    unittest.main()
