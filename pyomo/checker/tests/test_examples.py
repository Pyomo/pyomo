#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import os
try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

import pyutilib.th as unittest

from pyomo.checker import *
from pyomo.checker.plugins.checker import PyomoModelChecker


currdir = os.path.dirname(os.path.abspath(__file__))
exdir = os.path.join(currdir, "examples")

def createTestMethod(defs, package, checkerName, key):
    def testMethod(obj, name):
        import pyomo.environ

        runner = ModelCheckRunner()
        path = os.path.join(exdir, package, "{0}_{1}.py".format(checkerName, key))
        runner.run(script = path, checkers = {package:[checkerName]})
        
        checker = runner._checkers()[0]
        pc = checker.problemCount
        lns = checker.linenos
        checker.resetProblemCount()
        obj.assertEqual(defs[package][checkerName][key]['problems'], pc)
        if 'lines' in defs[package][checkerName][key]:
            obj.assertEqual(sorted(lns), sorted(defs[package][checkerName][key]['lines']))
    return testMethod


def assignTests(cls):
    defs = yaml.load(open(os.path.join(currdir, 'examples.yml'), 'r'))
    
    for package in defs:
        for checkerName in defs[package]:
            for key in defs[package][checkerName]:
                attrName = "{0}_{1}_{2}".format(package, checkerName, key)
                cls.add_fn_test(name=attrName, fn=createTestMethod(defs, package, checkerName, key))
                #setattr(cls, attrName, createTestMethod(defs, package, checkerName, key))


class ExampleTest(unittest.TestCase):
    """
    Test an example script, provided in the 'scripts' directory.
    """

    def setUp(self):
        def mockProblem(self, message = "Error", runner = None, script = None, lineno = None):
            self.problemCount += 1
            if lineno is not None:
                self.linenos.append(lineno)
        def resetProblemCount(self):
            self.problemCount = 0
            self.linenos = []
        PyomoModelChecker.problem_ = PyomoModelChecker.problem
        PyomoModelChecker.problem = mockProblem
        PyomoModelChecker.problemCount = 0
        PyomoModelChecker.linenos = []
        PyomoModelChecker.resetProblemCount = resetProblemCount

    def tearDown(self):
        PyomoModelChecker.problem = PyomoModelChecker.problem_
        del PyomoModelChecker.problemCount
        del PyomoModelChecker.resetProblemCount

if yaml_available:
    # Disable test for py3k.  For some reason, this messes up nose
    if not (sys.version_info[0:2] >= (3,0)):
        assignTests(ExampleTest)


if __name__ == "__main__":
    unittest.main()

