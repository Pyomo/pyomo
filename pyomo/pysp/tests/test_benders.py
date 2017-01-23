#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Get the directory where this script is defined, and where the baseline
# files are located.
#

import os
import sys
from os.path import abspath, dirname

this_test_directory = dirname(abspath(__file__))+os.sep

benders_example_dir = \
    dirname(dirname(dirname(dirname(abspath(__file__))))) + \
    os.sep+"examples"+os.sep+"pyomo"+os.sep+"benders"+os.sep

#
# Import the testing packages
#
import pyutilib.th as unittest

import pyomo.opt

def filter_fn(line):
    tmp = line.strip()
    return tmp.startswith('WARNING') and 'CBC' in tmp

solver = None
class TestBenders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = pyomo.opt.check_available_solvers('cplex')

    def setUp(self):
        if os.path.exists(this_test_directory+'benders_cplex.out'):
            os.remove(this_test_directory+'benders_cplex.out')
        # IMPT: This step is key, as Python keys off the name of the
        #       module, not the location.  So, different reference
        #       models in different directories won't be detected.  If
        #       you don't do this, the symptom is a model that doesn't
        #       have the attributes that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]

    def test_benders_cplex(self):
        import subprocess
        if not 'cplex' in solvers:
            self.skipTest("The 'cplex' executable is not available")
        out_file = open(this_test_directory+"benders_cplex.out",'w')
        os.chdir(benders_example_dir)
        subprocess.Popen(["lbin",
                          "python",
                          benders_example_dir+"runbenders"],
                         stdout=out_file).wait()
        os.chdir(this_test_directory)
        self.assertFileEqualsBaseline(
            this_test_directory+"benders_cplex.out",
            this_test_directory+"benders_cplex.baseline",
            tolerance=1e-2,
            filter=filter_fn)

if __name__ == "__main__":
    unittest.main()
