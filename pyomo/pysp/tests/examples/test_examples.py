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
from pyomo.opt import check_available_solvers

from six import StringIO

have_dot = True
try:
    if subprocess.call(["dot", "-?"],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT):
        have_dot = False
except:
    have_dot = False

have_networkx = False
try:
    import networkx
    have_networkx = True
except ImportError:
    have_networkx = False

thisdir = dirname(abspath(__file__))
baselineDir = join(thisdir, "baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisdir)))),
         "examples", "pysp")
examples_dir = join(pysp_examples_dir, "scripting")

solvers = check_available_solvers('cplex', 'glpk')

@unittest.category('nightly','expensive')
class TestExamples(unittest.TestCase):

    def setUp(self):
        self._tempfiles = []

    def _run_cmd(self, cmd):
        class_name, test_name = self.id().split('.')[-2:]
        print("%s.%s: Testing command: %s" % (class_name,
                                              test_name,
                                              str(' '.join(cmd))))
        outname = os.path.join(thisdir,
                               class_name+"."+test_name+".out")
        self._tempfiles.append(outname)
        with open(outname, "w") as f:
            subprocess.check_call(cmd,
                                  stdout=f,
                                  stderr=subprocess.STDOUT)

    def _cleanup(self):
        for fname in self._tempfiles:
            try:
                os.remove(fname)
            except OSError:
                pass
        self._tempfiles = []

    @unittest.skipIf(not 'cplex' in solvers,
                     'cplex not available')
    def test_ef_duals(self):
        cmd = ['python', join(examples_dir, 'ef_duals.py')]
        self._run_cmd(cmd)
        self._cleanup()

    @unittest.skipIf(not 'cplex' in solvers,
                     'cplex not available')
    def test_benders_scripting(self):
        cmd = ['python', join(examples_dir, 'benders_scripting.py')]
        self._run_cmd(cmd)
        self._cleanup()

    @unittest.skipIf(not 'cplex' in solvers,
                     'cplex not available')
    def test_admm(self):
        cmd = ['python', join(examples_dir, 'apps', 'admm.py')]
        cmd.extend(["-m", join(pysp_examples_dir, "farmer", "models")])
        cmd.extend(["-s", join(pysp_examples_dir, "farmer", "scenariodata")])
        self._run_cmd(cmd)
        self._cleanup()

    @unittest.skipIf(not have_networkx,
                     "networkx module not installed")
    def test_compile_scenario_tree(self):
        class_name, test_name = self.id().split('.')[-2:]
        tmpdir = os.path.join(thisdir, class_name+"_"+test_name)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertEqual(os.path.exists(tmpdir), False)
        cmd = ['python', join(examples_dir, 'apps', 'compile_scenario_tree.py')]
        cmd.extend(["-m", join(pysp_examples_dir,
                               "networkx_scenariotree",
                               "ReferenceModel.py")])
        cmd.extend(["--output-directory", tmpdir])
        self._run_cmd(cmd)
        self.assertEqual(os.path.exists(tmpdir), True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self._cleanup()

    @unittest.skipIf(not have_networkx,
                     "networkx module not installed")
    def test_generate_distributed_NL(self):
        class_name, test_name = self.id().split('.')[-2:]
        tmpdir = os.path.join(thisdir, class_name+"_"+test_name)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertEqual(os.path.exists(tmpdir), False)
        cmd = ['python', join(examples_dir, 'apps', 'generate_distributed_NL.py')]
        cmd.extend(["-m", join(pysp_examples_dir,
                               "networkx_scenariotree",
                               "ReferenceModel.py")])
        cmd.extend(["--output-directory", tmpdir])
        self._run_cmd(cmd)
        self.assertEqual(os.path.exists(tmpdir), True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self._cleanup()

    @unittest.skipIf((not have_dot) or (not have_networkx),
                     "dot command not available or networkx module not installed")
    def test_scenario_tree_image(self):
        class_name, test_name = self.id().split('.')[-2:]
        tmpfname = os.path.join(thisdir, class_name+"_"+test_name)+".pdf"
        try:
            os.remove(tmpfname)
        except OSError:
            pass
        self.assertEqual(os.path.exists(tmpfname), False)
        cmd = ['python', join(examples_dir, 'apps', 'scenario_tree_image.py')]
        cmd.extend(["-m", join(pysp_examples_dir,
                               "networkx_scenariotree",
                               "ReferenceModel.py")])
        cmd.extend(["--output-file", tmpfname])
        self._run_cmd(cmd)
        self.assertEqual(os.path.exists(tmpfname), True)
        os.remove(tmpfname)
        self._cleanup()

@unittest.category('parallel')
@unittest.skipIf(not (using_pyro3 or using_pyro4),
                 "Pyro / Pyro4 not available")
class TestParallelExamples(unittest.TestCase):

    def setUp(self):
        self._tempfiles = []

    def _run_cmd(self, cmd):
        class_name, test_name = self.id().split('.')[-2:]
        outname = os.path.join(thisdir,
                               class_name+"."+test_name+".out")
        self._tempfiles.append(outname)
        with open(outname, "w") as f:
            subprocess.check_call(cmd,
                                  stdout=f,
                                  stderr=subprocess.STDOUT)

    @unittest.nottest
    def _run_cmd_with_pyro(self, cmd, num_servers):
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
            class_name, test_name = self.id().split('.')[-2:]
            for i in range(num_servers):
                outname = os.path.join(thisdir,
                                       class_name+"."+test_name+".scenariotreeserver_"+str(i+1)+".out")
                self._tempfiles.append(outname)
                with open(outname, "w") as f:
                    scenariotreeserver_processes.append(
                        subprocess.Popen(["scenariotreeserver", "--traceback"] + \
                                         ["--pyro-host="+str(ns_host)] + \
                                         ["--pyro-port="+str(ns_port)],
                                         stdout=f,
                                         stderr=subprocess.STDOUT))
            cmd.append("--scenario-tree-manager=pyro")
            cmd.append("--pyro-host="+str(ns_host))
            cmd.append("--pyro-port="+str(ns_port))
            time.sleep(2)
            [_poll(proc) for proc in scenariotreeserver_processes]
            self._run_cmd(cmd)
        finally:
            _kill(ns_process)
            _kill(dispatcher_process)
            [_kill(proc) for proc in scenariotreeserver_processes]
            if os.path.exists(os.path.join(thisdir,'Pyro_NS_URI')):
                try:
                    os.remove(os.path.join(thisdir,'Pyro_NS_URI'))
                except OSError:
                    pass

    def _cleanup(self):
        for fname in self._tempfiles:
            try:
                os.remove(fname)
            except OSError:
                pass
        self._tempfiles = []


    @unittest.skipIf(not 'glpk' in solvers,
                     'glpk not available')
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
            class_name, test_name = self.id().split('.')[-2:]
            for i in range(3):
                outname = os.path.join(thisdir,
                                       class_name+"."+test_name+".scenariotreeserver_"+str(i+1)+".out")
                self._tempfiles.append(outname)
                with open(outname, "w") as f:
                    scenariotreeserver_processes.append(
                        subprocess.Popen(["scenariotreeserver", "--traceback"] + \
                                         ["--pyro-host="+str(ns_host)] + \
                                         ["--pyro-port="+str(ns_port)],
                                         stdout=f,
                                         stderr=subprocess.STDOUT))
            cmd = ['python', join(examples_dir, 'solve_distributed.py'), str(ns_port)]
            time.sleep(2)
            [_poll(proc) for proc in scenariotreeserver_processes]
            self._run_cmd(cmd)
        finally:
            _kill(ns_process)
            _kill(dispatcher_process)
            [_kill(proc) for proc in scenariotreeserver_processes]
            if os.path.exists(os.path.join(thisdir,'Pyro_NS_URI')):
                try:
                    os.remove(os.path.join(thisdir,'Pyro_NS_URI'))
                except OSError:
                    pass
        self._cleanup()

    @unittest.skipIf(not 'cplex' in solvers,
                     'cplex not available')
    def test_admm(self):
        cmd = ['python',
               join(examples_dir, 'apps', 'admm.py'),
               '-m', join(pysp_examples_dir, "farmer", "models"),
               '-s', join(pysp_examples_dir, "farmer", "scenariodata")]
        self._run_cmd_with_pyro(cmd, 3)
        self._cleanup()

    @unittest.skipIf(not have_networkx,
                     "networkx module not installed")
    def test_compile_scenario_tree(self):
        class_name, test_name = self.id().split('.')[-2:]
        tmpdir = os.path.join(thisdir, class_name+"_"+test_name)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertEqual(os.path.exists(tmpdir), False)
        cmd = ['python', join(examples_dir, 'apps', 'compile_scenario_tree.py')]
        cmd.extend(["-m", join(pysp_examples_dir,
                               "networkx_scenariotree",
                               "ReferenceModel.py")])
        cmd.extend(["--output-directory", tmpdir])
        self._run_cmd_with_pyro(cmd, 5)
        self.assertEqual(os.path.exists(tmpdir), True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self._cleanup()

    @unittest.skipIf(not have_networkx,
                     "networkx module not installed")
    def test_generate_distributed_NL(self):
        class_name, test_name = self.id().split('.')[-2:]
        tmpdir = os.path.join(thisdir, class_name+"_"+test_name)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self.assertEqual(os.path.exists(tmpdir), False)
        cmd = ['python', join(examples_dir, 'apps', 'generate_distributed_NL.py')]
        cmd.extend(["-m", join(pysp_examples_dir,
                               "networkx_scenariotree",
                               "ReferenceModel.py")])
        cmd.extend(["--output-directory", tmpdir])
        self._run_cmd_with_pyro(cmd, 5)
        self.assertEqual(os.path.exists(tmpdir), True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        self._cleanup()

if __name__ == "__main__":
    unittest.main()
