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

networkx_available = False
try:
    import networkx
    networkx_available = True
except:
    networkx_available = False

thisdir = dirname(abspath(__file__))
baselineDir = join(thisdir, "baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisdir)))), "examples", "pysp")

_run_verbose = True
_json_exact_comparison = True
_diff_tolerance = 1e-4

testing_solvers = {}
testing_solvers['cplex','nl'] = False
testing_solvers['cplex','lp'] = False
testing_solvers['cplex','mps'] = False
testing_solvers['cplex','python'] = False
testing_solvers['_cplex_persistent','python'] = False
testing_solvers['ipopt','nl'] = False
def setUpModule():
    global testing_solvers
    import pyomo.environ
    from pyomo.solvers.tests.solvers import test_solver_cases
    for _solver, _io in test_solver_cases():
        if (_solver, _io) in testing_solvers and \
            test_solver_cases(_solver, _io).available:
            testing_solvers[_solver, _io] = True


class _EvalXHATTesterBase(object):

    basename = None
    model_location = None
    scenario_tree_location = None
    solver_name = None
    solver_io = None

    def setUp(self):
        self._tempfiles = []
        self.options = {}
        self.options['--scenario-tree-manager'] = 'serial'

    def _run_cmd(self, cmd):
        class_name, test_name = self.id().split('.')[-2:]
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

    def _setup(self, options):
        assert self.basename is not None
        assert self.model_location is not None
        assert self.solver_name is not None
        assert self.solver_io is not None
        if not testing_solvers[self.solver_name, self.solver_io]:
            self.skipTest("%s (interface=%s) is not available"
                          % (self.solver_name, self.solver_io))
        options['--solver'] = self.solver_name
        options['--solver-io'] = self.solver_io
        options['--model-location'] = self.model_location
        if self.scenario_tree_location is not None:
            options['--scenario-tree-location'] = self.scenario_tree_location
        if _run_verbose:
            options['--verbose'] = None
        options['--output-times'] = None
        options['--traceback'] = None
        options['--solution-loader-extension'] = 'pyomo.pysp.plugins.jsonio'
        options['--jsonloader-input-name'] = \
            join(thisdir, self.basename+'_ef_solution.json')
        options['--solution-saver-extension'] = 'pyomo.pysp.plugins.jsonio'

        class_name, test_name = self.id().split('.')[-2:]
        options['--jsonsaver-output-name'] = \
            join(thisdir, class_name+"."+test_name+'_solution.json')
        options['--output-scenario-costs'] = \
            join(thisdir, class_name+"."+test_name+'_costs.json')
        if os.path.exists(options['--jsonsaver-output-name']):
            shutil.rmtree(options['--jsonsaver-output-name'],
                          ignore_errors=True)
        if os.path.exists(options['--output-scenario-costs']):
            shutil.rmtree(options['--output-scenario-costs'],
                          ignore_errors=True)

    def _get_cmd(self):
        cmd = ['evaluate_xhat']
        for name, val in self.options.items():
            cmd.append(name)
            if val is not None:
                cmd.append(str(val))
        class_name, test_name = self.id().split('.')[-2:]
        print("%s.%s: Testing command: %s" % (class_name,
                                              test_name,
                                              str(' '.join(cmd))))
        return cmd

    def test_scenarios(self):
        self._setup(self.options)
        cmd = self._get_cmd()
        self._run_cmd(cmd)
        self.assertMatchesJsonBaseline(
            self.options['--jsonsaver-output-name'],
            join(thisdir, self.basename+'_ef_solution.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)
        self.assertMatchesJsonBaseline(
            self.options['--output-scenario-costs'],
            join(thisdir, self.basename+'_ef_costs.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)
        self._cleanup()

_pyomo_ns_host = '127.0.0.1'
_pyomo_ns_port = None
_pyomo_ns_process = None
_dispatch_srvr_port = None
_dispatch_srvr_process = None
_taskworker_processes = []
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
    if os.path.exists(join(thisdir, "Pyro_NS_URI")):
        try:
            os.remove(join(thisdir, "Pyro_NS_URI"))
        except OSError:
            pass

class _EvalXHATPyroTesterBase(_EvalXHATTesterBase):

    def _setUpPyro(self):
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
        class_name, test_name = self.id().split('.')[-2:]
        if len(_taskworker_processes) == 0:
            for i in range(3):
                outname = os.path.join(thisdir,
                                       class_name+"."+test_name+".scenariotreeserver_"+str(i+1)+".out")
                self._tempfiles.append(outname)
                with open(outname, "w") as f:
                    _taskworker_processes.append(
                        subprocess.Popen(["scenariotreeserver", "--traceback"] + \
                                         (["--verbose"] if _run_verbose else []) + \
                                         ["--pyro-host="+str(_pyomo_ns_host)] + \
                                         ["--pyro-port="+str(_pyomo_ns_port)],
                                         stdout=f,
                                         stderr=subprocess.STDOUT))

            time.sleep(2)
            [_poll(proc) for proc in _taskworker_processes]

    def setUp(self):
        self._tempfiles = []
        self._setUpPyro()
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
        self._run_cmd(cmd)
        self.assertMatchesJsonBaseline(
            self.options['--jsonsaver-output-name'],
            join(thisdir, self.basename+'_ef_solution.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)
        self.assertMatchesJsonBaseline(
            self.options['--output-scenario-costs'],
            join(thisdir, self.basename+'_ef_costs.json'),
            tolerance=_diff_tolerance,
            delete=True,
            exact=_json_exact_comparison)
        self._cleanup()

@unittest.nottest
def create_test_classes(basename,
                        model_location,
                        scenario_tree_location,
                        solver_name,
                        solver_io,
                        categories):
    assert basename is not None

    class _base(object):
        pass
    _base.basename = basename
    _base.model_location = model_location
    _base.scenario_tree_location = scenario_tree_location
    _base.solver_name = solver_name
    _base.solver_io = solver_io

    class_append_name = basename + "_" + solver_name + "_" + solver_io
    class_names = []

    @unittest.category(*categories)
    class TestEvalXHAT_Serial(_base,
                              _EvalXHATTesterBase):
        pass
    class_names.append(TestEvalXHAT_Serial.__name__ + "_"+class_append_name)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Serial, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro(_base,
                            _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
    class_names.append(TestEvalXHAT_Pyro.__name__ + "_"+class_append_name)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Pyro, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro_MultipleWorkers(_base,
                                            _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-multiple-scenariotreeserver-workers'] = None
    class_names.append(TestEvalXHAT_Pyro_MultipleWorkers.__name__ + "_"+class_append_name)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Pyro_MultipleWorkers, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro_HandshakeAtStartup(_base,
                                               _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-handshake-at-startup'] = None
    class_names.append(TestEvalXHAT_Pyro_HandshakeAtStartup.__name__ + "_"+class_append_name)
    globals()[class_names[-1]] = type(
        class_names[-1], (TestEvalXHAT_Pyro_HandshakeAtStartup, unittest.TestCase), {})

    @unittest.skipIf(not (using_pyro3 or using_pyro4),
                     "Pyro or Pyro4 is not available")
    @unittest.category('parallel')
    class TestEvalXHAT_Pyro_HandshakeAtStartup_MultipleWorkers(_base,
                                                               _EvalXHATPyroTesterBase):
        def setUp(self):
            _EvalXHATPyroTesterBase.setUp(self)
        def _setup(self, options, servers=None):
            _EvalXHATPyroTesterBase._setup(self, options, servers=servers)
            options['--pyro-handshake-at-startup'] = None
            options['--pyro-multiple-scenariotreeserver-workers'] = None
    class_names.append(TestEvalXHAT_Pyro_HandshakeAtStartup_MultipleWorkers.__name__ + "_"+class_append_name)
    globals()[class_names[-1]] = type(
        class_names[-1],
        (TestEvalXHAT_Pyro_HandshakeAtStartup_MultipleWorkers, unittest.TestCase),
        {})

    return class_names

#
# create the actual testing classes
#

for solver_name, solver_io in [('cplex','lp'),
                               ('cplex','mps'),
                               ('cplex','nl'),
                               ('cplex','python'),
                               ('_cplex_persistent','python')]:

    simple_quadratic_examples_dir = join(pysp_examples_dir, "simple_quadratic")
    simple_quadratic_model_dir = join(simple_quadratic_examples_dir, "ReferenceModel.py")
    simple_quadratic_data_dir = None
    create_test_classes('simple_quadratic',
                        simple_quadratic_model_dir,
                        simple_quadratic_data_dir,
                        solver_name,
                        solver_io,
                        ('nightly','expensive'))

    farmer_examples_dir = join(pysp_examples_dir, "farmer")
    farmer_model_dir = join(farmer_examples_dir, "models")
    farmer_data_dir = join(farmer_examples_dir, "scenariodata")
    create_test_classes('farmer',
                        farmer_model_dir,
                        farmer_data_dir,
                        solver_name,
                        solver_io,
                        ('nightly','expensive'))

    finance_examples_dir = join(pysp_examples_dir, "finance")
    finance_model_dir = join(finance_examples_dir, "models")
    finance_data_dir = join(finance_examples_dir, "scenariodata")
    create_test_classes('finance',
                        finance_model_dir,
                        finance_data_dir,
                        solver_name,
                        solver_io,
                        ('nightly','expensive'))

    hydro_examples_dir = join(pysp_examples_dir, "hydro")
    hydro_model_dir = join(hydro_examples_dir, "models")
    hydro_data_dir = join(hydro_examples_dir, "scenariodata")
    create_test_classes('hydro',
                        hydro_model_dir,
                        hydro_data_dir,
                        solver_name,
                        solver_io,
                        ('nightly','expensive'))

for solver_name, solver_io in [('ipopt','nl')]:

    networkx_examples_dir = join(pysp_examples_dir, "networkx_scenariotree")
    networkx_model_dir = join(networkx_examples_dir, "ReferenceModel.py")
    networkx_data_dir = None
    class_names = create_test_classes('networkx',
                                      networkx_model_dir,
                                      networkx_data_dir,
                                      solver_name,
                                      solver_io,
                                      ('nightly','expensive'))
    for name in class_names:
        globals()[name] = unittest.skipIf(not networkx_available,
                                          "networkx module is not available")(globals()[name])

# this example is big
for solver_name, solver_io in [('cplex','lp')]:

    baa99_examples_dir = join(pysp_examples_dir, "baa99")
    baa99_model_dir = join(baa99_examples_dir, "ReferenceModel.py")
    baa99_data_dir = None
    create_test_classes('baa99',
                        baa99_model_dir,
                        baa99_data_dir,
                        solver_name,
                        solver_io,
                        ('nightly','expensive'))

if __name__ == "__main__":
    unittest.main()
