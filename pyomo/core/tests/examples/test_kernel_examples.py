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
# Tests for Pyomo kernel examples
#

import glob
import sys
from os.path import basename, dirname, abspath, join

import pyutilib.subprocess
import pyutilib.th as unittest

from pyomo.common.dependencies import numpy_available, scipy_available

import platform
if platform.python_implementation() == "PyPy":
    # The scipy is importable into PyPy, but ODE integrators don't work. (2/ 18)
    scipy_available = False

currdir = dirname(abspath(__file__))
topdir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
examplesdir = join(topdir, "examples", "kernel")

examples = glob.glob(join(examplesdir,"*.py"))
examples.extend(glob.glob(join(examplesdir,"mosek","*.py")))

testing_solvers = {}
testing_solvers['ipopt','nl'] = False
testing_solvers['glpk','lp'] = False
testing_solvers['mosek_direct','python'] = False
def setUpModule():
    global testing_solvers
    import pyomo.environ
    from pyomo.solvers.tests.solvers import test_solver_cases
    for _solver, _io in test_solver_cases():
        if (_solver, _io) in testing_solvers and \
            test_solver_cases(_solver, _io).available:
            testing_solvers[_solver, _io] = True

@unittest.nottest
def create_test_method(example):
    # It is important that this inner function has a name that
    # starts with 'test' in order for nose to discover it
    # after we assign it to the class. I have _no_ idea why
    # this is the case since we are returing the function object
    # and placing it on the class with a different name.
    def testmethod(self):
        if basename(example) == "piecewise_nd_functions.py":
            if (not numpy_available) or \
               (not scipy_available) or \
               (not testing_solvers['ipopt','nl']) or \
               (not testing_solvers['glpk','lp']):
                self.skipTest("Numpy or Scipy or Ipopt or Glpk is not available")
        elif "mosek" in example:
            if (not testing_solvers['ipopt','nl']) or \
               (not testing_solvers['mosek_direct','python']):
                self.skipTest("Ipopt or Mosek is not available")
        rc, log = pyutilib.subprocess.run([sys.executable,example])
        self.assertEqual(rc, 0, msg=log)
    return testmethod

class TestKernelExamples(unittest.TestCase):
    pass
for filename in examples:
    testname = basename(filename)
    assert testname.endswith(".py")
    testname = "test_"+testname[:-3]+"_example"
    setattr(TestKernelExamples,
            testname,
            create_test_method(filename))

if __name__ == "__main__":
    unittest.main()
