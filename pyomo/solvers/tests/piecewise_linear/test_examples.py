#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Test the Pyomo command-line interface
#

from itertools import zip_longest
import os
from os.path import abspath, dirname, join

import pyomo.common.unittest as unittest
import pyomo.scripting.convert as convert

from pyomo.common.fileutils import this_file_dir, PYOMO_ROOT_DIR
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline

currdir = this_file_dir()
scriptdir = join(PYOMO_ROOT_DIR, 'examples', 'pyomo', 'piecewise')

_NL_diff_tol = 1e-9
_LP_diff_tol = 1e-9


class Test(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        TempfileManager.push()
        self.tmpdir = TempfileManager.create_tempdir()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.cwd)
        TempfileManager.pop()

    def run_convert2nl(self, name):
        return convert.pyomo2nl(
            [
                '--symbolic-solver-labels',
                '--file-determinism',
                '1',
                join(scriptdir, name),
            ]
        )

    def run_convert2lp(self, name):
        return convert.pyomo2lp(['--symbolic-solver-labels', join(scriptdir, name)])

    def test_step_lp(self):
        """Test examples/pyomo/piecewise/step.py"""
        self.run_convert2lp('step.py')
        _test, _base = join(self.tmpdir, 'unknown.lp'), join(currdir, 'step.lp')
        self.assertEqual(*load_and_compare_lp_baseline(_base, _test))

    def test_step_nl(self):
        """Test examples/pyomo/piecewise/step.py"""
        self.run_convert2nl('step.py')
        _test, _base = join(self.tmpdir, 'unknown.nl'), join(currdir, 'step.nl')
        self.assertEqual(*load_and_compare_nl_baseline(_base, _test))

    def test_nonconvex_lp(self):
        """Test examples/pyomo/piecewise/nonconvex.py"""
        self.run_convert2lp('nonconvex.py')
        _test, _base = join(self.tmpdir, 'unknown.lp'), join(currdir, 'nonconvex.lp')
        self.assertEqual(*load_and_compare_lp_baseline(_base, _test))

    def test_nonconvex_nl(self):
        """Test examples/pyomo/piecewise/nonconvex.py"""
        self.run_convert2nl('nonconvex.py')
        _test, _base = join(self.tmpdir, 'unknown.nl'), join(currdir, 'nonconvex.nl')
        self.assertEqual(*load_and_compare_nl_baseline(_base, _test))

    def test_convex_lp(self):
        """Test examples/pyomo/piecewise/convex.py"""
        self.run_convert2lp('convex.py')
        _test, _base = join(self.tmpdir, 'unknown.lp'), join(currdir, 'convex.lp')
        self.assertEqual(*load_and_compare_lp_baseline(_base, _test))

    def test_convex_nl(self):
        """Test examples/pyomo/piecewise/convex.py"""
        self.run_convert2nl('convex.py')
        _test, _base = join(self.tmpdir, 'unknown.nl'), join(currdir, 'convex.nl')
        self.assertEqual(*load_and_compare_nl_baseline(_base, _test))

    def test_indexed_lp(self):
        """Test examples/pyomo/piecewise/indexed.py"""
        self.run_convert2lp('indexed.py')
        _test, _base = join(self.tmpdir, 'unknown.lp'), join(currdir, 'indexed.lp')
        self.assertEqual(*load_and_compare_lp_baseline(_base, _test))

    def test_indexed_nl(self):
        """Test examples/pyomo/piecewise/indexed.py"""
        self.run_convert2nl('indexed.py')
        _base = join(currdir, 'indexed.nl')
        _test = join(self.tmpdir, 'unknown.nl')
        self.assertEqual(*load_and_compare_nl_baseline(_base, _test))

    def test_indexed_nonlinear_nl(self):
        """Test examples/pyomo/piecewise/indexed_nonlinear.py"""
        self.run_convert2nl('indexed_nonlinear.py')
        _test = join(self.tmpdir, 'unknown.nl')
        _base = join(currdir, 'indexed_nonlinear.nl')
        self.assertEqual(*load_and_compare_nl_baseline(_base, _test))


if __name__ == "__main__":
    unittest.main()
