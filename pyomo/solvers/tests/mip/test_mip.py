#
# Tests driven by test_mip.yml
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.autotest

#pyutilib.autotest.create_test_suites(filename=currdir+'test_mip.yml', _globals=globals())

if __name__ == "__main__":
    unittest.main()
