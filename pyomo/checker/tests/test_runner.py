#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os

import pyutilib.th as unittest

from pyomo.checker import  ModelCheckRunner, ModelScript
from pyomo.checker.plugins.checker import IModelChecker, ImmediateDataChecker, ImmediateTreeChecker, IterativeDataChecker, IterativeTreeChecker


currdir = os.path.dirname(os.path.abspath(__file__))

class MockChecker(object):
    checkCount = 0
    def check(self, runner, script, info):
        self.checkCount += 1
    def _checkerPackage(self):
        return 'mock'
    def resetCount(self):
        self.checkCount = 0

# Multiple inheritance, because otherwise something weird happens with the plugin system
# MockChecker MUST BE specified first so that its overridden methods take precedence
# See http://www.python.org/download/releases/2.3/mro/

class MockImmediateDataChecker(MockChecker, ImmediateDataChecker): pass
class MockImmediateTreeChecker(MockChecker, ImmediateTreeChecker): pass
class MockIterativeDataChecker(MockChecker, IterativeDataChecker): pass
class MockIterativeTreeChecker(MockChecker, IterativeTreeChecker): pass

class RunnerTest(unittest.TestCase):
    """
    Test the ModelCheckRunner class.
    """

    testScripts = [
        "print('Hello, world!')\n",
        "import sys\nsys.stdout.write('Hello, world!\\n')\n"
        "for i in range(10):\n\tprint(i)\n"
    ]

    def test_init(self):
        "Check that a ModelCheckRunner instantiates properly"

        runner = ModelCheckRunner()

        self.assertEqual([], runner.scripts)
        self.assertTrue(len(runner._checkers(all=True)) > 0)
        for c in runner._checkers:
            self.assertTrue(IModelChecker in c._implements)

    def test_addScript(self):
        "Check that a runner handles its script list properly"

        runner = ModelCheckRunner()
        expectedScriptCount = 0

        for text in self.testScripts:
            self.assertEquals(expectedScriptCount, len(runner.scripts))

            script = ModelScript(text = text)
            runner.addScript(script)
            expectedScriptCount += 1

            self.assertEquals(expectedScriptCount, len(runner.scripts))
            self.assertTrue(script in runner.scripts)

    def test_run_immediate(self):
        "Check that a runner calls check() on an immediate checker"

        for text in self.testScripts:
            
            runner = ModelCheckRunner()
            script = ModelScript(text = text)
            runner.addScript(script)

            runner.run(checkers = {'mock':['MockImmediateDataChecker', 'MockImmediateTreeChecker']})

            for klass in [MockImmediateDataChecker, MockImmediateTreeChecker]:
                mockChecker = list(filter((lambda c : c.__class__ == klass), runner._checkers()))[0]
                self.assertEqual(1, mockChecker.checkCount)
                mockChecker.resetCount()

    def test_run_iterative(self):
        "Check that a runner calls check() on an iterative checker"

        for text in self.testScripts:
            
            runner = ModelCheckRunner()
            script = ModelScript(text = text)
            runner.addScript(script)

            runner.run(checkers = {'mock':['MockIterativeDataChecker', 'MockIterativeTreeChecker']})

            for klass in [MockIterativeDataChecker, MockIterativeTreeChecker]:
                mockChecker = list(filter((lambda c : c.__class__ == klass), runner._checkers()))[0]
                self.assertTrue(mockChecker.checkCount >= 1)
                mockChecker.resetCount()


if __name__ == "__main__":
    unittest.main()

