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

class _SMPSTesterBase(object):

    basename = None
    model_location = None
    scenario_tree_location = None

    def _setup(self, options):
        assert self.basename is not None
        assert self.model_location is not None
        options['--basename'] = self.basename
        options['--model-location'] = self.model_location
        if self.scenario_tree_location is not None:
            options['--scenario-tree-location'] = self.scenario_tree_location
        if _run_verbose:
            options['--verbose'] = ''
        options['--output-times'] = ''
        options['--explicit'] = ''
        options['--traceback'] = ''
        options['--keep-scenario-files'] = ''
        options['--keep-auxiliary-files'] = ''
        class_name, test_name = self.id().split('.')[-2:]
        options['--output-directory'] = \
            join(thisDir, class_name+"."+test_name)
        if os.path.exists(options['--output-directory']):
            shutil.rmtree(options['--output-directory'], ignore_errors=True)

    def _get_cmd(self):
        cmd = 'pysp2smps '
        for name, val in self.options.items():
            cmd += name
            if val != '':
                cmd += "="+str(self.options[name])
            cmd += ' '
        print("Command: "+str(cmd))
        return cmd

    def _diff(self, baselinedir, outputdir, dc=None):
        if dc is None:
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
            with open(fromfile, 'r') as f_from:
                fromlines = f_from.readlines()
                with open(tofile, 'r') as f_to:
                    tolines = f_to.readlines()
                    diff = difflib.context_diff(fromlines, tolines,
                                                fromfile+" (baseline)",
                                                tofile+" (output)")
                    out = StringIO()
                    out.write("Output file does not match baseline:\n")
                    for line in diff:
                        out.write(line)
                    self.fail(out.getvalue())
        for subdir in dc.subdirs:
            self._diff(join(baselinedir, subdir),
                       join(outputdir, subdir),
                       dc=dc.subdirs[subdir])
        shutil.rmtree(outputdir, ignore_errors=True)

    def test_scenarios_LP(self):
        self._setup(self.options)
        self.options['--core-format'] = 'lp'
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_LP_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_MPS(self):
        self._setup(self.options)
        self.options['--core-format'] = 'mps'
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_MPS_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_LP_symbolic_names(self):
        self._setup(self.options)
        self.options['--core-format'] = 'lp'
        self.options['--symbolic-solver-labels'] = ''
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_LP_symbolic_names_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_MPS_symbolic_names(self):
        self._setup(self.options)
        self.options['--core-format'] = 'mps'
        self.options['--symbolic-solver-labels'] = ''
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_MPS_symbolic_names_baseline'),
                   self.options['--output-directory'])

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

class _SMPSPyroTesterBase(_SMPSTesterBase):

    def setUp(self):
        _setUpModule()
        [_poll(proc) for proc in _taskworker_processes]
        self.options = {}
        self.options['--scenario-tree-manager'] = 'pyro'
        self.options['--pyro-host'] = 'localhost'
        self.options['--pyro-port'] = _pyomo_ns_port
        self.options['--pyro-required-scenariotreeservers'] = 3

    def _setup(self, options, servers=None):
        _SMPSTesterBase._setup(self, options)
        if servers is not None:
            options['--pyro-required-scenariotreeservers'] = servers

    def test_scenarios_LP_1server(self):
        self._setup(self.options, servers=1)
        self.options['--core-format'] = 'lp'
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_LP_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_MPS_1server(self):
        self._setup(self.options, servers=1)
        self.options['--core-format'] = 'mps'
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_MPS_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_LP_symbolic_names_1server(self):
        self._setup(self.options, servers=1)
        self.options['--core-format'] = 'lp'
        self.options['--symbolic-solver-labels'] = ''
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_LP_symbolic_names_baseline'),
                   self.options['--output-directory'])

    def test_scenarios_MPS_symbolic_names_1server(self):
        self._setup(self.options, servers=1)
        self.options['--core-format'] = 'mps'
        self.options['--symbolic-solver-labels'] = ''
        cmd = self._get_cmd()
        rc = os.system(cmd)
        self.assertEqual(rc, False)
        self._diff(os.path.join(thisDir, self.basename+'_MPS_symbolic_names_baseline'),
                   self.options['--output-directory'])

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
    class TestPySP2SMPS_Serial(_base,
                               _SMPSTesterBase):
        def setUp(self):
            self.options = {}
            self.options['--scenario-tree-manager'] = 'serial'
    class_names.append(TestPySP2SMPS_Serial.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestPySP2SMPS_Serial, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category(*categories)
    class TestPySP2SMPS_Pyro(_base,
                             unittest.TestCase,
                             _SMPSPyroTesterBase):
        def setUp(self):
            _SMPSPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _SMPSPyroTesterBase._setup(self, options, servers=servers)
    class_names.append(TestPySP2SMPS_Pyro.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestPySP2SMPS_Pyro, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category(*categories)
    class TestPySP2SMPS_Pyro_MultipleWorkers(_base,
                                             unittest.TestCase,
                                             _SMPSPyroTesterBase):
        def setUp(self):
            _SMPSPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _SMPSPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-multiple-scenariotreeserver-workers'] = ''
    class_names.append(TestPySP2SMPS_Pyro_MultipleWorkers.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestPySP2SMPS_Pyro_MultipleWorkers, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category(*categories)
    class TestPySP2SMPS_Pyro_HandshakeAtStartup(_base,
                                                unittest.TestCase,
                                                _SMPSPyroTesterBase):
        def setUp(self):
            _SMPSPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _SMPSPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-handshake-at-startup'] = ''
    class_names.append(TestPySP2SMPS_Pyro_HandshakeAtStartup.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestPySP2SMPS_Pyro_HandshakeAtStartup, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category(*categories)
    class TestPySP2SMPS_Pyro_HandshakeAtStartup_MultipleWorkers(_base,
                                                                unittest.TestCase,
                                                                _SMPSPyroTesterBase):
        def setUp(self):
            _SMPSPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _SMPSPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-handshake-at-startup'] = ''
            options['--pyro-multiple-scenariotreeserver-workers'] = ''
    class_names.append(TestPySP2SMPS_Pyro_HandshakeAtStartup_MultipleWorkers.__name__ + "_"+basename)
    globals()[class_names[-1]] = type(
        class_names[-1],
        (TestPySP2SMPS_Pyro_HandshakeAtStartup_MultipleWorkers, unittest.TestCase),
        {})

    return tuple(globals()[name] for name in class_names)

#
# create the actual testing classes
#

farmer_examples_dir = join(pysp_examples_dir, "farmer")
farmer_model_dir = join(farmer_examples_dir, "smps_model")
farmer_data_dir = join(farmer_examples_dir, "scenariodata")

create_test_classes('farmer',
                    farmer_model_dir,
                    farmer_data_dir,
                    ('nightly','expensive'))

piecewise_model_dir = join(thisDir, "piecewise_model.py")
create_test_classes('piecewise',
                    piecewise_model_dir,
                    None,
                    ('nightly','expensive'))

if __name__ == "__main__":
    unittest.main()
