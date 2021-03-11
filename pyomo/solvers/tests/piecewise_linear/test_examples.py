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

from itertools import zip_longest
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
        _out, _log = join(currdir,'unknown.lp'), join(currdir, 'step.lp')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))

    def test_step_nl(self):
        """Test examples/pyomo/piecewise/step.py"""
        self.run_convert2nl('step.py')
        _out, _log = join(currdir,'unknown.nl'), join(currdir, 'step.nl')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_nonconvex_lp(self):
        """Test examples/pyomo/piecewise/nonconvex.py"""
        self.run_convert2lp('nonconvex.py')
        _out, _log = join(currdir,'unknown.lp'), join(currdir, 'nonconvex.lp')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))

    def test_nonconvex_nl(self):
        """Test examples/pyomo/piecewise/nonconvex.py"""
        self.run_convert2nl('nonconvex.py')
        _out, _log = join(currdir,'unknown.nl'), join(currdir, 'nonconvex.nl')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_convex_lp(self):
        """Test examples/pyomo/piecewise/convex.py"""
        self.run_convert2lp('convex.py')
        _out, _log = join(currdir,'unknown.lp'), join(currdir, 'convex.lp')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))

    def test_convex_nl(self):
        """Test examples/pyomo/piecewise/convex.py"""
        self.run_convert2nl('convex.py')
        _out, _log = join(currdir,'unknown.nl'), join(currdir, 'convex.nl')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_indexed_lp(self):
        """Test examples/pyomo/piecewise/indexed.py"""
        self.run_convert2lp('indexed.py')
        with open(join(currdir,'unknown.lp'), 'r') as f1, \
            open(join(currdir, 'indexed.lp'), 'r') as f2:
                f1_contents = list(filter(None, f1.read().split()))
                f2_contents = list(filter(None, f2.read().split()))
                for item1, item2 in zip_longest(f1_contents, f2_contents):
                    try:
                        self.assertAlmostEqual(float(item1), float(item2))
                    except:
                        self.assertEqual(item1, item2)

    def test_indexed_nl(self):
        """Test examples/pyomo/piecewise/indexed.py"""
        self.run_convert2nl('indexed.py')
        with open(join(currdir,'unknown.nl'), 'r') as f1, \
            open(join(currdir, 'indexed.nl'), 'r') as f2:
                f1_contents = list(filter(None, f1.read().split()))
                f2_contents = list(filter(None, f2.read().split()))
                for item1, item2 in zip_longest(f1_contents, f2_contents):
                    try:
                        self.assertAlmostEqual(float(item1), float(item2))
                    except:
                        self.assertEqual(item1, item2)
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))

    def test_indexed_nonlinear_nl(self):
        """Test examples/pyomo/piecewise/indexed_nonlinear.py"""
        self.run_convert2nl('indexed_nonlinear.py')
        _out, _log = join(currdir,'unknown.nl'), join(currdir, 'indexed_nonlinear.nl')
        self.assertTrue(cmp(_out, _log),
                        msg="Files %s and %s differ" % (_out, _log))
        os.remove(join(currdir,'unknown.row'))
        os.remove(join(currdir,'unknown.col'))


if __name__ == "__main__":
    unittest.main()
