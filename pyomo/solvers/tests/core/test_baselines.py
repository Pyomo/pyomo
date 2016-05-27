#
# Tests driven by test_baselines.yml
#

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services
import pyomo.environ

import pyutilib.autotest
pyutilib.autotest.create_test_suites(filename=currdir+'test_baselines.yml', _globals=globals())

# GAH: The pyutilib.autotest.create_test_suites function
#      does some strange things to the TempfileManager state
#      and I have no idea where these configurations get
#      updated (I grep'ed for TempfileManager inside pyomo.data
#      and pyutilib.autotest - nothing). Is this a bug?
def tearDownModule():
    pyutilib.services.TempfileManager.clear_tempfiles()
    pyutilib.services.TempfileManager.tempdir = None
    pyutilib.services.TempfileManager.unique_files()

if __name__ == "__main__":
    unittest.main()
