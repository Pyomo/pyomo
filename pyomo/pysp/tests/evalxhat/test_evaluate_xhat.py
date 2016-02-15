#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import socket
import os
from os.path import join, dirname, abspath
import time
import subprocess
import difflib
import filecmp
import shutil

try:
    from subprocess import check_output as _run_cmd
except:
    # python 2.6
    from subprocess import check_call as _run_cmd

import pyutilib.services
import pyutilib.th as unittest
from pyutilib.pyro import using_pyro3, using_pyro4
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
from pyomo.environ import *

from six import StringIO

thisDir = dirname(abspath(__file__))
baselineDir = join(thisDir, "baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisDir)))), "examples", "pysp")

_run_verbose = True
_json_exact_comparison = True
_diff_tolerance = 1e-4

class _EvalXHATTesterBase(object):

    basename = None
    model_location = None
    scenario_tree_location = None

    def _setup(self, options):
        assert self.basename is not None
        assert self.model_location is not None
        options['--model-location'] = self.model_location
        if self.scenario_tree_location is not None:
            options['--scenario-tree-location'] = self.scenario_tree_location
        if _run_verbose:
            options['--verbose'] = ''
        options['--output-times'] = ''
        options['--traceback'] = ''
        options['--solution-loader-extension'] = 'pyomo.pysp.plugins.jsonio'
        options['--jsonloader-input-name'] = \
            join(thisDir, self.basename+'_ef_solution.json')
        options['--solution-saver-extension'] = 'pyomo.pysp.plugins.jsonio'

        class_name, test_name = self.id().split('.')[-2:]
        options['--jsonsaver-output-name'] = \
            join(thisDir, class_name+"."+test_name+'_solution.json')
        options['--output-scenario-costs'] = \
            join(thisDir, class_name+"."+test_name+'_costs.json')
        if os.path.exists(options['--jsonsaver-output-name']):
            shutil.rmtree(options['--jsonsaver-output-name'],
                          ignore_errors=True)
        if os.path.exists(options['--output-scenario-costs']):
            shutil.rmtree(options['--output-scenario-costs'],
                          ignore_errors=True)

    def _get_cmd(self):
        cmd = 'evaluate_xhat '
        for name, val in self.options.items():
            cmd += name
            if val != '':
                cmd += "="+str(self.options[name])
            cmd += ' '
        print("Command: "+str(cmd))
        return cmd

    def test_scenarios(self):
        self._setup(self.options)
        cmd = self._get_cmd()
        _run_cmd(cmd, shell=True)
        self.assertMatchesJsonBaseline(
            self.options['--jsonsaver-output-name'],
            join(thisDir, self.basename+'_ef_solution.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)
        self.assertMatchesJsonBaseline(
            self.options['--output-scenario-costs'],
            join(thisDir, self.basename+'_ef_costs.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)

_pyomo_ns_host = '127.0.0.1'
_pyomo_ns_port = None
_pyomo_ns_process = None
_dispatch_srvr_port = None
_dispatch_srvr_process = None
_dispatch_srvr_options = "--host localhost --daemon-host localhost"
_taskworker_processes = []
def _setUpModule():
    global _pyomo_ns_port
    global _pyomo_ns_process
    global _dispatch_srvr_port
    global _dispatch_srvr_process
    global _taskworker_processes
    if _pyomo_ns_process is None:
        _pyomo_ns_process, _pyomo_ns_port = \
            _get_test_nameserver(ns_host=_pyomo_ns_host)
    assert _pyomo_ns_process is not None
    if _dispatch_srvr_process is None:
        _dispatch_srvr_process, _dispatch_srvr_port = \
            _get_test_dispatcher(ns_host=_pyomo_ns_host,
                                 ns_port=_pyomo_ns_port)
    assert _dispatch_srvr_process is not None
    if len(_taskworker_processes) == 0:
        for i in range(3):
            _taskworker_processes.append(\
                subprocess.Popen(["scenariotreeserver", "--traceback"] + \
                                 (["--verbose"] if _run_verbose else []) + \
                                 ["--pyro-host="+str(_pyomo_ns_host)] + \
                                 ["--pyro-port="+str(_pyomo_ns_port)]))

        time.sleep(2)
        [_poll(proc) for proc in _taskworker_processes]

def tearDownModule():
    global _pyomo_ns_port
    global _pyomo_ns_process
    global _dispatch_srvr_port
    global _dispatch_srvr_process
    global _taskworker_processes
    _kill(_pyomo_ns_process)
    _pyomo_ns_port = None
    _pyomo_ns_process = None
    _kill(_dispatch_srvr_process)
    _dispatch_srvr_port = None
    _dispatch_srvr_process = None
    [_kill(proc) for proc in _taskworker_processes]
    _taskworker_processes = []
    if os.path.exists(join(thisDir, "Pyro_NS_URI")):
        try:
            os.remove(join(thisDir, "Pyro_NS_URI"))
        except OSError:
            pass

class _EvalXHATPyroTesterBase(_EvalXHATTesterBase):

    def setUp(self):
        _setUpModule()
        [_poll(proc) for proc in _taskworker_processes]
        self.options = {}
        self.options['--scenario-tree-manager'] = 'pyro'
        self.options['--pyro-host'] = 'localhost'
        self.options['--pyro-port'] = _pyomo_ns_port
        self.options['--pyro-required-scenariotreeservers'] = 3

    def _setup(self, options, servers=None):
        _EvalXHATTesterBase._setup(self, options)
        if servers is not None:
            options['--pyro-required-scenariotreeservers'] = servers

    def test_scenarios_1server(self):
        self._setup(self.options, servers=1)
        cmd = self._get_cmd()
        _run_cmd(cmd, shell=True)
        self.assertMatchesJsonBaseline(
            self.options['--jsonsaver-output-name'],
            join(thisDir, self.basename+'_ef_solution.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)
        self.assertMatchesJsonBaseline(
            self.options['--output-scenario-costs'],
            join(thisDir, self.basename+'_ef_costs.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)

@unittest.nottest
def create_test_classes(basename,
                        model_location,
                        scenario_tree_location,
                        categories):
    assert basename is not None

    class _base(object):
        pass
    _base.basename = basename
    _base.model_location = model_location
    _base.scenario_tree_location = scenario_tree_location

    class_names = []

    @unittest.category(*categories)
    class TestEvalXHAT_Serial(_base,
                              _EvalXHATTesterBase):
        def setUp(self):
            self.options = {}
            self.options['--scenario-tree-manager'] = 'serial'
    class_names.append(TestEvalXHAT_Serial.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Serial, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro(_base,
                            unittest.TestCase,
                            _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
    class_names.append(TestEvalXHAT_Pyro.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Pyro, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro_MultipleWorkers(_base,
                                            unittest.TestCase,
                                            _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-multiple-scenariotreeserver-workers'] = ''
    class_names.append(TestEvalXHAT_Pyro_MultipleWorkers.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Pyro_MultipleWorkers, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro_HandshakeAtStartup(_base,
                                               unittest.TestCase,
                                               _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-handshake-at-startup'] = ''
    class_names.append(TestEvalXHAT_Pyro_HandshakeAtStartup.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Pyro_HandshakeAtStartup, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro_HandshakeAtStartup_MultipleWorkers(_base,
                                                               unittest.TestCase,
                                                               _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-handshake-at-startup'] = ''
            options['--pyro-multiple-scenariotreeserver-workers'] = ''
    class_names.append(TestEvalXHAT_Pyro_HandshakeAtStartup_MultipleWorkers.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1],
        (TestEvalXHAT_Pyro_HandshakeAtStartup_MultipleWorkers, unittest.TestCase),
        {})

    return tuple(globals()[name] for name in class_names)

#
# create the actual testing classes
#

farmer_examples_dir = join(pysp_examples_dir, "farmer")
farmer_model_dir = join(farmer_examples_dir, "models")
farmer_data_dir = join(farmer_examples_dir, "scenariodata")
create_test_classes('farmer',
                    farmer_model_dir,
                    farmer_data_dir,
                    ('nightly','expensive'))

finance_examples_dir = join(pysp_examples_dir, "finance")
finance_model_dir = join(finance_examples_dir, "models")
finance_data_dir = join(finance_examples_dir, "scenariodata")
create_test_classes('finance',
                    finance_model_dir,
                    finance_data_dir,
                    ('nightly','expensive'))

hydro_examples_dir = join(pysp_examples_dir, "hydro")
hydro_model_dir = join(hydro_examples_dir, "models")
hydro_data_dir = join(hydro_examples_dir, "scenariodata")
create_test_classes('hydro',
                    hydro_model_dir,
                    hydro_data_dir,
                    ('nightly','expensive'))

baa99_examples_dir = join(pysp_examples_dir, "baa99")
baa99_model_dir = join(baa99_examples_dir, "baa99.py")
baa99_data_dir = None
create_test_classes('baa99',
                    baa99_model_dir,
                    baa99_data_dir,
                    ('nightly','expensive'))

if __name__ == "__main__":
    unittest.main()
