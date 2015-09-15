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
from six import StringIO

thisDir = dirname(abspath(__file__))
baselineDir = join(thisDir, "baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisDir)))), "examples", "pysp")
pyomo_bin_dir = \
    join(dirname(dirname(dirname(dirname(dirname(dirname(thisDir)))))), "bin")

farmer_examples_dir = join(pysp_examples_dir, "farmer")
farmer_model_dir = join(farmer_examples_dir, "smps_model")
farmer_data_dir = join(farmer_examples_dir, "scenariodata")

class _SMPSTesterBase(object):

    def _setup(self, options):
        options['--model-location'] = farmer_model_dir
        options['--scenario-tree-location'] = farmer_data_dir
        options['--verbose'] = ''
        options['--output-times'] = ''
        options['--explicit'] = ''
        options['--keep-scenario-files'] = ''
        self.options['--basename'] = 'farmer'
        class_name, test_name = self.id().split('.')[-2:]
        self.options['--output-directory'] = \
            join(thisDir, class_name+"."+test_name)

    def _get_cmd(self):
        cmd = 'pysp2smps '
        for name, val in self.options.items():
            cmd += name
            if val != '':
                cmd += "="+str(self.options[name])
            cmd += ' '
        print("Command: "+str(cmd))
        return cmd

    def _diff(self, baselinedir, outputdir):
        dc = filecmp.dircmp(baselinedir, outputdir, ['.svn'])
        if dc.left_only:
            self.fail("Files or subdirectories missing from output: "
                      +str(dc.left_only))
        if dc.right_only:
            self.fail("Files or subdirectories missing from baseline: "
                      +str(dc.right_only))
        for name in dc.diff_files:
            fromfile = join(dc.left, name)
            tofile = join(dc.right, name)
            fromlines = open(fromfile, 'U').readlines()
            tolines = open(tofile, 'U').readlines()
            diff = difflib.context_diff(fromlines, tolines,
                                        fromfile+" (baseline)",
                                        tofile+" (output)")
            out = StringIO()
            out.write("Output file does not match baseline:\n")
            for line in diff:
                out.write(line)
            self.fail(out.getvalue())
        shutil.rmtree(outputdir, ignore_errors=True)

    def test_scenarios(self):
        self._setup(self.options)
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, 'farmer_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_symbolic_names(self):
        self._setup(self.options)
        self.options['--symbolic-solver-labels'] = ''
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, 'farmer_symbolic_names_baseline'),
                   self.options['--output-directory'])

#
# create the actual testing classes
#

@unittest.category('nightly','expensive')
class TestPySP2SMPS_Serial(unittest.TestCase, _SMPSTesterBase):

    def setUp(self):
        self.options = {}
        self.options['--scenario-tree-manager'] = 'serial'

_pyro_ns_process = None
_pyomo_ns_options = ""
if using_pyro3:
    _pyomo_ns_options = "-r -k -n localhost"
elif using_pyro4:
    _pyomo_ns_options = "-n localhost"
_dispatch_srvr_process = None
_dispatch_srvr_options = "-n localhost"
_scenariotreeserver_options = "--verbose --pyro-hostname=localhost --traceback"
_scenariotreeserver_processes = []

def tearDownModule():
    global _pyro_ns_process
    global _dispatch_srvr_process
    global _scenariotreeserver_processes
    if _pyro_ns_process is not None:
        try:
            _pyro_ns_process.kill()
        except:
            pass
        _pyro_ns_process = None
    if _dispatch_srvr_process is not None:
        try:
            _dispatch_srvr_process.kill()
        except:
            pass
        _dispatch_srvr_process = None
    if len(_scenariotreeserver_processes):
        try:
            for _p in _scenariotreeserver_processes:
                _p.kill()
        except:
            pass
    _scenariotreeserver_processes = []

class _SMPSSPPyroTesterBase(_SMPSTesterBase):

    @classmethod
    def setUpClass(cls):
        global _pyro_ns_process
        global _dispatch_srvr_process
        global _scenariotreeserver_processes
        if _pyro_ns_process is None:
            _pyro_ns_process = \
                subprocess.Popen(["pyomo_ns"]+(_pyomo_ns_options.split()))
        if _dispatch_srvr_process is None:
            cmd = ['dispatch_srvr'] + _dispatch_srvr_options.split()
            print("Launching cmd: %s" % (' '.join(cmd)))
            _dispatch_srvr_process = subprocess.Popen(cmd)
            time.sleep(2)
        if len(_scenariotreeserver_processes) == 0:
            cmd = ["scenariotreeserver"] + _scenariotreeserver_options.split()
            for i in range(3):
                print("Launching cmd: %s" % (' '.join(cmd)))
                _scenariotreeserver_processes.append(
                    subprocess.Popen(cmd))

    def setUp(self):
        self.options = {}
        self.options['--scenario-tree-manager'] = 'sppyro'
        self.options['--pyro-hostname'] = 'localhost'
        self.options['--sppyro-required-servers'] = 3

    def _setup(self, options, servers=None):
        _SMPSTesterBase._setup(self, options)
        if servers is not None:
            options['--sppyro-required-servers'] = servers

    def test_scenarios_1server(self):
        self._setup(self.options, servers=1)
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, 'farmer_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_symbolic_names_1server(self):
        self._setup(self.options, servers=1)
        self.options['--symbolic-solver-labels'] = ''
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, 'farmer_symbolic_names_baseline'),
                   self.options['--output-directory'])

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestPySP2SMPS_SPPyro(unittest.TestCase, _SMPSSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _SMPSSPPyroTesterBase.setUpClass()
    def setUp(self):
        _SMPSSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _SMPSSPPyroTesterBase._setup(self, options, servers=servers)

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestPySP2SMPS_SPPyro_MultipleWorkers(unittest.TestCase,
                                           _SMPSSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _SMPSSPPyroTesterBase.setUpClass()
    def setUp(self):
        _SMPSSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _SMPSSPPyroTesterBase._setup(self, options, servers=servers)
        options['--sppyro-multiple-server-workers'] = ''

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestPySP2SMPS_SPPyro_HandshakeAtStartup(unittest.TestCase,
                                              _SMPSSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _SMPSSPPyroTesterBase.setUpClass()
    def setUp(self):
        _SMPSSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _SMPSSPPyroTesterBase._setup(self, options, servers=servers)
        options['--sppyro-handshake-at-startup'] = ''

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestPySP2SMPS_SPPyro_HandshakeAtStartup_MultipleWorkers(unittest.TestCase,
                                                              _SMPSSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _SMPSSPPyroTesterBase.setUpClass()
    def setUp(self):
        _SMPSSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _SMPSSPPyroTesterBase._setup(self, options, servers=servers)
        options['--sppyro-handshake-at-startup'] = ''
        options['--sppyro-multiple-server-workers'] = ''

if __name__ == "__main__":
    unittest.main()
