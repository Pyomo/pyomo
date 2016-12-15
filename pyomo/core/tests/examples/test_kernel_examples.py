#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Tests for Pyomo kernel examples
#

import os
import glob
from os.path import basename, dirname, abspath, join

import pyutilib.subprocess
import pyutilib.th as unittest

currdir = dirname(abspath(__file__))
topdir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
examplesdir = join(topdir, "examples", "kernel")

examples = glob.glob(join(examplesdir,"*.py"))

@unittest.nottest
def create_test_method(example):
    def _method(self):
        rc, log = pyutilib.subprocess.run(['python',example])
        self.assertEqual(rc, 0, msg=log)
    return _method

@unittest.category("smoke", "nightly", "expensive")
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
