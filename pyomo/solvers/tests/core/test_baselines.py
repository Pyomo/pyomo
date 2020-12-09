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
# Tests driven by test_baselines.yml
#

#
# Tests driven by test_solvers.yml
#
# NOTE: This is configured to run NO tests unless run on the
#   command-line.
#

if __name__ == "__main__":
    import os
    from os.path import abspath, dirname
    currdir = dirname(abspath(__file__))+os.sep

    from pyomo.common.tempfiles import TempfileManager

    from pyutilib.autotest import create_test_suites
    create_test_suites(filename=currdir+'test_baselines.yml', _globals=globals())

    # GAH: The pyutilib.autotest.create_test_suites function
    #      does some strange things to the TempfileManager state
    #      and I have no idea where these configurations get
    #      updated (I grep'ed for TempfileManager inside pyomo.data
    #      and pyutilib.autotest - nothing). Is this a bug?
    def tearDownModule():
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = None
        TempfileManager.unique_files()

