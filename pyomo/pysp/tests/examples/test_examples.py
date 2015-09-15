#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import os
from os.path import join, dirname, abspath
import time
import subprocess
import difflib
import filecmp
import shutil

from pyutilib.pyro import using_pyro3, using_pyro4
import pyutilib.services
import pyutilib.th as unittest
from pyutilib.misc.config import ConfigBlock
from pyomo.environ import *
from pyomo.pysp.scenariotree.scenariotreemanager import (ScenarioTreeManagerSerial,
                                                         ScenarioTreeManagerSPPyro)
import pyomo.environ
from pyomo.opt import load_solvers

from six import StringIO

thisDir = dirname(abspath(__file__))
baselineDir = join(thisDir, "baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisDir)))), "examples", "pysp")
examples_dir = join(pysp_examples_dir, "scripting")

solvers = load_solvers('cplex', 'glpk')

pyutilib.services.register_executable("mpirun")
mpirun_executable = pyutilib.services.registered_executable('mpirun')
mpirun_available = not mpirun_executable is None

_pyomo_ns_options = ""
if using_pyro3:
    _pyomo_ns_options = "-r -k -n localhost"
elif using_pyro4:
    _pyomo_ns_options = "-n localhost"

@unittest.category('smoke','nightly','expensive')
class TestExamples(unittest.TestCase):

    @unittest.skipIf(solvers['cplex'] is None, 'cplex not available')
    def test_ef_duals(self):
        cmd = 'python '+join(examples_dir, 'ef_duals.py')
        print("Testing command: "+cmd)
        rc = os.system(cmd)
        self.assertEqual(rc, False)

    @unittest.skipIf((solvers['glpk'] is None) or \
                     (not mpirun_available) or \
                     (not (using_pyro3 or using_pyro4)),
                     'glpk or mpirun or Pyro / Pyro4 is not available')
    def test_solve_distributed(self):
        ns_process = \
            subprocess.Popen(["pyomo_ns"]+(_pyomo_ns_options.split()))
        try:
            cmd = 'mpirun -np 1 dispatch_srvr -n localhost : '
            cmd += '-np 3 scenariotreeserver --pyro-hostname=localhost --traceback : '
            cmd += '-np 1 python '+join(examples_dir, 'solve_distributed.py')
            print("Testing command: "+cmd)
            rc = os.system(cmd)
            self.assertEqual(rc, False)
        finally:
            ns_process.kill()

if __name__ == "__main__":
    unittest.main()
