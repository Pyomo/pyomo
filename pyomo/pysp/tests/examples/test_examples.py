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
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
import pyomo.environ
from pyomo.opt import load_solvers

from six import StringIO

thisDir = dirname(abspath(__file__))
baselineDir = join(thisDir, "baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisDir)))),
         "examples", "pysp")
examples_dir = join(pysp_examples_dir, "scripting")

solvers = load_solvers('cplex', 'glpk')

@unittest.category('nightly','expensive','parallel')
class TestExamples(unittest.TestCase):

    @unittest.skipIf(solvers['cplex'] is None, 'cplex not available')
    def test_ef_duals(self):
        cmd = 'python '+join(examples_dir, 'ef_duals.py')
        print("Testing command: "+cmd)
        rc = os.system(cmd)
        self.assertEqual(rc, False)

    @unittest.skipIf((solvers['glpk'] is None) or \
                     (not (using_pyro3 or using_pyro4)),
                     'glpk or Pyro / Pyro4 is not available')
    def test_solve_distributed(self):
        ns_host = '127.0.0.1'
        ns_process = None
        dispatcher_process = None
        scenariotreeserver_processes = []
        try:
            ns_process, ns_port = \
                _get_test_nameserver(ns_host=ns_host)
            self.assertNotEqual(ns_process, None)
            dispatcher_process, dispatcher_port = \
                _get_test_dispatcher(ns_host=ns_host,
                                     ns_port=ns_port)
            self.assertNotEqual(dispatcher_process, None)
            scenariotreeserver_processes = []
            for i in range(3):
                scenariotreeserver_processes.append(\
                    subprocess.Popen(["scenariotreeserver", "--traceback"] + \
                                     ["--pyro-host="+str(ns_host)] + \
                                     ["--pyro-port="+str(ns_port)]))
            cmd = ('python ' + \
                   join(examples_dir, 'solve_distributed.py') + \
                   ' %d') % (ns_port)
            print("Testing command: "+cmd)
            time.sleep(2)
            [_poll(proc) for proc in scenariotreeserver_processes]
            rc = os.system(cmd)
            self.assertEqual(rc, False)
        finally:
            _kill(ns_process)
            _kill(dispatcher_process)
            [_kill(proc) for proc in scenariotreeserver_processes]

if __name__ == "__main__":
    unittest.main()
