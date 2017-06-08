#
# Tests driven by test_solvers.yml
#
# NOTE: This is configured to run NO tests unless run on the
#   command-line.
#

if __name__ == "__main__":
    import os
    import sys
    from os.path import abspath, dirname
    currdir = dirname(abspath(__file__))+os.sep

    import pyutilib.th as unittest
    import pyutilib.autotest
    #import pyomo.data.core

    pyutilib.autotest.create_test_suites(filename=currdir+'test_solvers.yml', _globals=globals())

    unittest.main()
