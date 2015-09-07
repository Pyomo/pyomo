#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import os
import time
import subprocess

from pyutilib.pyro import using_pyro3, using_pyro4
import pyutilib.services
import pyutilib.th as unittest
from pyutilib.misc.config import ConfigBlock
from pyomo.environ import *
from pyomo.pysp.scenariotree.scenariotreemanager import (ScenarioTreeManagerSerial,
                                                         ScenarioTreeManagerSPPyro)
thisdir = os.path.dirname(os.path.abspath(__file__))

class _ScenarioTreeManagerTesterBase(object):

    def _setup(self, options):
        options.model_location = os.path.join(thisdir, 'dummy_model.py')
        options.scenario_tree_location = None
        options.aggregategetter_callback_location = \
            [os.path.join(thisdir, 'aggregate_callback1.py'),
             os.path.join(thisdir, 'aggregate_callback2.py')]
        options.postinit_callback_location = \
            [os.path.join(thisdir, 'postinit_callback1.py'),
             os.path.join(thisdir, 'postinit_callback2.py')]
        options.objective_sense_stage_based = 'min'
        options.verbose = True
        options.output_times = True
        options.compile_scenario_instances = True
        options.output_instance_construction_time = True
        options.profile_memory = True
        options.scenario_tree_random_seed = 1

    def test_scenarios(self):
        self._setup(self.options)
        with self.cls(self.options) as manager:
            ahs = manager.initialize()
            if ahs is not None:
                manager.complete_actions(ahs)
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_bundles1(self):
        options = ConfigBlock()
        self._setup(self.options)
        self.options.create_random_bundles = 1
        with self.cls(self.options) as manager:
            ahs = manager.initialize()
            if ahs is not None:
                manager.complete_actions(ahs)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_bundles2(self):
        options = ConfigBlock()
        self._setup(self.options)
        self.options.create_random_bundles = 2
        with self.cls(self.options) as manager:
            ahs = manager.initialize()
            if ahs is not None:
                manager.complete_actions(ahs)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_bundles3(self):
        options = ConfigBlock()
        self._setup(self.options)
        self.options.create_random_bundles = 3
        with self.cls(self.options) as manager:
            ahs = manager.initialize()
            if ahs is not None:
                manager.complete_actions(ahs)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

#
# create the actual testing classes
#

@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSerial(unittest.TestCase, _ScenarioTreeManagerTesterBase):

    cls = ScenarioTreeManagerSerial

    def setUp(self):
        self.options = ConfigBlock()
        ScenarioTreeManagerSerial.register_options(self.options)

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

class _ScenarioTreeManagerSPPyroTesterBase(_ScenarioTreeManagerTesterBase):

    cls = ScenarioTreeManagerSPPyro

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
        self.options = ConfigBlock()
        ScenarioTreeManagerSPPyro.register_options(self.options)

    def _setup(self, options, servers=None):
        _ScenarioTreeManagerTesterBase._setup(self, options)
        options.pyro_hostname = 'localhost'
        if servers is not None:
            options.sppyro_required_servers = servers

    def test_scenarios_1server(self):
        self._setup(self.options, servers=1)
        with self.cls(self.options) as manager:
            manager.complete_actions(manager.initialize())
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_bundles1_1server(self):
        self._setup(self.options, servers=1)
        self.options.create_random_bundles = 1
        with self.cls(self.options) as manager:
            manager.complete_actions(manager.initialize())
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_bundles2_1server(self):
        self._setup(self.options, servers=1)
        self.options.create_random_bundles = 2
        with self.cls(self.options) as manager:
            manager.complete_actions(manager.initialize())
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_bundles3_1server(self):
        self._setup(self.options, servers=1)
        self.options.create_random_bundles = 3
        with self.cls(self.options) as manager:
            manager.complete_actions(manager.initialize())
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyro(unittest.TestCase, _ScenarioTreeManagerSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _ScenarioTreeManagerSPPyroTesterBase.setUpClass()
    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.handshake_with_sppyro = False
        options.sppyro_serial_workers = False

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyroManyWorkers(unittest.TestCase, _ScenarioTreeManagerSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _ScenarioTreeManagerSPPyroTesterBase.setUpClass()
    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.handshake_with_sppyro = False
        options.sppyro_serial_workers = True

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyroHandshakePyro(unittest.TestCase, _ScenarioTreeManagerSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _ScenarioTreeManagerSPPyroTesterBase.setUpClass()
    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.handshake_with_sppyro = True
        options.sppyro_serial_workers = False

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyroHandshakePyroManyWorkers(unittest.TestCase, _ScenarioTreeManagerSPPyroTesterBase):

    @classmethod
    def setUpClass(cls):
        _ScenarioTreeManagerSPPyroTesterBase.setUpClass()
    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.handshake_with_sppyro = True
        options.sppyro_serial_workers = True

if __name__ == "__main__":
    unittest.main()
