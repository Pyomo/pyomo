#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Tests for Pyomo kernel examples
#

import glob
import os
import platform
import subprocess
import sys

import pyomo.common.unittest as unittest
import pyomo.environ

from pyomo.common.dependencies import numpy_available, scipy_available
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases

if platform.python_implementation() == "PyPy":
    # The scipy is importable into PyPy, but ODE integrators don't work. (2/ 18)
    scipy_available = False

testing_solvers = {}
testing_solvers['ipopt', 'nl'] = False
testing_solvers['glpk', 'lp'] = False
testing_solvers['mosek_direct', 'python'] = False


def setUpModule():
    for _solver, _io in _test_solver_cases():
        if (_solver, _io) in testing_solvers and _test_solver_cases(
            _solver, _io
        ).available:
            testing_solvers[_solver, _io] = True


def create_method(example):
    # It is important that this inner function has a name that
    # starts with 'test' in order for pytest to discover it
    # after we assign it to the class. I have _no_ idea why
    # this is the case since we are returning the function object
    # and placing it on the class with a different name.
    def testmethod(self):
        result = subprocess.run(
            [sys.executable, example],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)

    tm = testmethod
    if os.path.basename(example) == "piecewise_nd_functions.py":
        tm = unittest.skipUnless(numpy_available, "Test requires numpy")(tm)
        tm = unittest.skipUnless(scipy_available, "Test requires scipy")(tm)
        tm = unittest.skipUnless(testing_solvers['ipopt', 'nl'], "Test requires ipopt")(
            tm
        )
        tm = unittest.skipUnless(testing_solvers['glpk', 'lp'], "Test requires glpk")(
            tm
        )
    elif "mosek" in example:
        tm = unittest.skipUnless(testing_solvers['ipopt', 'nl'], "Test requires ipopt")(
            tm
        )
        tm = unittest.skipUnless(
            testing_solvers['mosek_direct', 'python'], "Test requires mosek"
        )(tm)
    return tm


class TestKernelExamples(unittest.TestCase):
    pass


examplesdir = os.path.join(PYOMO_ROOT_DIR, "examples", "kernel")
examples = glob.glob(os.path.join(examplesdir, "*.py"))
examples.extend(glob.glob(os.path.join(examplesdir, "mosek", "*.py")))
for filename in examples:
    testname = os.path.basename(filename)
    assert testname.endswith(".py")
    testname = "test_" + testname[:-3] + "_example"
    setattr(TestKernelExamples, testname, create_method(filename))


if __name__ == "__main__":
    unittest.main()
