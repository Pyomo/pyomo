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
try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

from pyutilib.pyro import using_pyro3, using_pyro4
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
import pyutilib.services
import pyutilib.th as unittest
from pyomo.pysp.util.config import PySPConfigBlock
from pyomo.pysp.scenariotree.scenariotreemanager import (ScenarioTreeManagerSerial,
                                                         ScenarioTreeManagerSPPyro)
from pyomo.pysp.scenariotree.scenariotreeworkerbasic import \
    ScenarioTreeWorkerBasic
from pyomo.pysp.scenariotree.scenariotreeserver import (RegisterScenarioTreeWorker,
                                                        SPPyroScenarioTreeServer)
from pyomo.pysp.scenariotree.scenariotreeserverutils import \
    InvocationType
from pyomo.environ import *

thisfile = os.path.abspath(__file__)
thisdir = os.path.dirname(thisfile)

_run_verbose = True

class _ScenarioTreeWorkerTest(ScenarioTreeWorkerBasic):

    def junk(self, *args, **kwds):
        return (args, kwds)

class _ScenarioTreeManagerTestSerial(ScenarioTreeManagerSerial):

    def __init__(self, *args, **kwds):
        assert kwds.pop('registered_worker_name', None) == 'ScenarioTreeWorkerTest'
        super(_ScenarioTreeManagerTestSerial, self).__init__(*args, **kwds)

    def junk(self, *args, **kwds):
        return (args, kwds)

if "ScenarioTreeWorkerTest" not in SPPyroScenarioTreeServer._registered_workers:
    RegisterScenarioTreeWorker("ScenarioTreeWorkerTest", _ScenarioTreeWorkerTest)

_init_kwds = {'registered_worker_name': 'ScenarioTreeWorkerTest'}

def _Single(manager, scenario_tree):
    return {'scenarios': [scenario.name for scenario in scenario_tree.scenarios],
            'bundles': [bundle.name for bundle in scenario_tree.bundles]}

def _byScenario(manager, scenario_tree, scenario):
    return scenario.name

def _byBundle(manager, scenario_tree, bundle):
    return bundle.name

def _byScenarioChained(manager, scenario_tree, scenario, data):
    return ((scenario.name, data),)

def _byBundleChained(manager, scenario_tree, bundle, data):
    return ((bundle.name, data),)

class _ScenarioTreeManagerTesterBase(object):

    _bundle_dict3 = OrderedDict()
    _bundle_dict3['Bundle1'] = ['Scenario1']
    _bundle_dict3['Bundle2'] = ['Scenario2']
    _bundle_dict3['Bundle3'] = ['Scenario3']

    _bundle_dict2 = OrderedDict()
    _bundle_dict2['Bundle1'] = ['Scenario1', 'Scenario2']
    _bundle_dict2['Bundle2'] = ['Scenario3']

    _bundle_dict1 = OrderedDict()
    _bundle_dict1['Bundle1'] = ['Scenario1','Scenario2','Scenario3']

    @unittest.nottest
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
        options.verbose = _run_verbose
        options.output_times = True
        options.compile_scenario_instances = True
        options.output_instance_construction_time = True
        options.profile_memory = True
        options.scenario_tree_random_seed = 1

    @unittest.nottest
    def _run_function_tests(self, manager, async=False, oneway=False, delay=False):
        class_name, test_name = self.id().split('.')[-2:]
        print("Running function tests on %s.%s" % (class_name, test_name))
        data = {}
        init = manager.initialize(async=async)
        if async:
            init = init.complete()
        self.assertEqual(all(_v is True for _v in init.values()), True)
        self.assertEqual(sorted(init.keys()), sorted(manager.worker_names()))
        self.assertEqual(len(manager.scenario_tree.scenarios) > 0, True)
        if manager.scenario_tree.contains_bundles():
            self.assertEqual(len(manager.scenario_tree.bundles) > 0, True)
        else:
            self.assertEqual(len(manager.scenario_tree.bundles), 0)

        #
        # test_invoke_external_function
        #

        print("")
        print("Running InvocationType.Single...")
        results = manager.invoke_external_function(
            thisfile,
            "_Single",
            invocation_type=InvocationType.SingleInvocation,
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_Single'] = results
        print("Running InvocationType.byScenario...")
        results = manager.invoke_external_function(
            thisfile,
            "_byScenario",
            invocation_type=InvocationType.PerScenarioInvocation,
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_byScenario'] = results
        print("Running InvocationType.byScenarioChained...")
        results = manager.invoke_external_function(
            thisfile,
            "_byScenarioChained",
            function_args=(None,),
            invocation_type=InvocationType.PerScenarioChainedInvocation,
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_byScenarioChained'] = results
        if manager.scenario_tree.contains_bundles():
            print("Running InvocationType.byBundle...")
            results = manager.invoke_external_function(
                thisfile,
                "_byBundle",
                invocation_type=InvocationType.PerBundleInvocation,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_byBundle'] = results
            print("Running InvocationType.byBundleChained...")
            results = manager.invoke_external_function(
                thisfile,
                "_byBundleChained",
                function_args=(None,),
                invocation_type=InvocationType.PerBundleChainedInvocation,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_byBundleChained'] = results
        for name in data:
            results = data[name]
            if delay:
                if async:
                    results = results.complete()
            if oneway:
                self.assertEqual(id(results), id(None))
            else:
                if name == "_Single":
                    self.assertEqual(sorted(results.keys()),
                                     sorted(manager.worker_names()))
                    scenarios = []
                    bundles = []
                    for worker_name in results:
                        self.assertEqual(len(results[worker_name]['scenarios']) > 0,
                                         True)
                        scenarios.extend(results[worker_name]['scenarios'])
                        if manager.scenario_tree.contains_bundles():
                            self.assertEqual(len(results[worker_name]['bundles']) > 0,
                                             True)
                        bundles.extend(results[worker_name]['bundles'])
                    self.assertEqual(sorted(scenarios),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                    self.assertEqual(sorted(bundles),
                                     sorted([_bundle.name for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_byScenario":
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                elif name == "_byScenarioChained":
                    self.assertEqual(
                        results, (('Scenario3', ('Scenario2', ('Scenario1', None))),))
                elif name == "_byBundle":
                    self.assertEqual(manager.scenario_tree.contains_bundles(), True)
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_bundle.name for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_byBundleChained":
                    self.assertEqual(manager.scenario_tree.contains_bundles(), True)
                    if len(manager.scenario_tree.bundles) == 3:
                        self.assertEqual(
                            results, (('Bundle3', ('Bundle2', ('Bundle1', None))),))
                    elif len(manager.scenario_tree.bundles) == 2:
                        self.assertEqual(results, (('Bundle2', ('Bundle1', None)),))
                    elif len(manager.scenario_tree.bundles) == 1:
                        self.assertEqual(results, (('Bundle1', None),))
                    else:
                        assert False
                else:
                    assert False

        if isinstance(manager, ScenarioTreeManagerSerial):
            self.assertEqual(manager._aggregate_user_data['leaf_node'],
                             [scenario.leaf_node.name for scenario
                              in manager.scenario_tree.scenarios])
            self.assertEqual(len(manager._aggregate_user_data['names']), 0)
        elif isinstance(manager, ScenarioTreeManagerSPPyro):
            self.assertEqual(manager._aggregate_user_data['leaf_node'],
                             [scenario.leaf_node.name for scenario
                              in manager.scenario_tree.scenarios])
            self.assertEqual(manager._aggregate_user_data['names'],
                             [scenario.name for scenario
                              in manager.scenario_tree.scenarios])
        else:
            assert False

        #
        # Test invoke_external_function_on_worker
        #

        data = {}
        print("")
        print("Running InvocationType.Single on individual workers...")
        data['_Single'] = {}
        for worker_name in manager.worker_names():
            results = manager.invoke_external_function_on_worker(
                worker_name,
                thisfile,
                "_Single",
                invocation_type=InvocationType.Single,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_Single'][worker_name] = results
        print("Running InvocationType.byScenario on individual workers...")
        data['_byScenario'] = {}
        for worker_name in manager.worker_names():
            results = manager.invoke_external_function_on_worker(
                worker_name,
                thisfile,
                "_byScenario",
                invocation_type=InvocationType.PerScenario,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_byScenario'][worker_name] = results
        print("Running InvocationType.byScenarioChained on individual workers...")
        data['_byScenarioChained'] = {}
        results = (None,)
        for worker_name in manager.worker_names():
            results = manager.invoke_external_function_on_worker(
                worker_name,
                thisfile,
                "_byScenarioChained",
                function_args=results,
                invocation_type=InvocationType.PerScenarioChained,
                async=async)
            if async:
                results = results.complete()
        data['_byScenarioChained'] = results
        if manager.scenario_tree.contains_bundles():
            print("Running InvocationType.byBundle on individual workers...")
            data['_byBundle'] = {}
            for worker_name in manager.worker_names():
                results = manager.invoke_external_function_on_worker(
                    worker_name,
                    thisfile,
                    "_byBundle",
                    invocation_type=InvocationType.PerBundle,
                    oneway=oneway,
                    async=async)
                if not delay:
                    if async:
                        results = results.complete()
                data['_byBundle'][worker_name] = results
            print("Running InvocationType.byBundleChained on individual workers...")
            data['_byBundleChained'] = {}
            results = (None,)
            for worker_name in manager.worker_names():
                results = manager.invoke_external_function_on_worker(
                    worker_name,
                    thisfile,
                    "_byBundleChained",
                    function_args=results,
                    invocation_type=InvocationType.PerBundleChained,
                    async=async)
                if async:
                    results = results.complete()
            data['_byBundleChained'] = results
        print("")
        for name in data:
            results = data[name]
            if (name != '_byScenarioChained') and (name != '_byBundleChained'):
                if delay:
                    if async:
                        for worker_name in results:
                            results[worker_name] = results[worker_name].complete()
            if oneway:
                if (name != '_byScenarioChained') and (name != '_byBundleChained'):
                    for worker_name in results:
                        self.assertEqual(id(results[worker_name]), id(None))
            else:
                if name == "_Single":
                    self.assertEqual(sorted(results.keys()),
                                     sorted(manager.worker_names()))
                    scenarios = []
                    bundles = []
                    for worker_name in results:
                        self.assertEqual(len(results[worker_name]['scenarios']) > 0,
                                         True)
                        scenarios.extend(results[worker_name]['scenarios'])
                        if manager.scenario_tree.contains_bundles():
                            self.assertEqual(len(results[worker_name]['bundles']) > 0,
                                             True)
                        bundles.extend(results[worker_name]['bundles'])
                    self.assertEqual(sorted(scenarios),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                    self.assertEqual(sorted(bundles),
                                     sorted([_bundle.name for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_byScenario":
                    _results = {}
                    for worker_name in results:
                        _results.update(results[worker_name])
                    results = _results
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                elif name == "_byScenarioChained":
                    self.assertEqual(
                        results, (('Scenario3', ('Scenario2', ('Scenario1', None))),))
                elif name == "_byBundle":
                    _results = {}
                    for worker_name in results:
                        _results.update(results[worker_name])
                    results = _results
                    self.assertEqual(manager.scenario_tree.contains_bundles(), True)
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_bundle.name for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_byBundleChained":
                    self.assertEqual(manager.scenario_tree.contains_bundles(), True)
                    if len(manager.scenario_tree.bundles) == 3:
                        self.assertEqual(
                            results, (('Bundle3', ('Bundle2', ('Bundle1', None))),))
                    elif len(manager.scenario_tree.bundles) == 2:
                        self.assertEqual(results, (('Bundle2', ('Bundle1', None)),))
                    elif len(manager.scenario_tree.bundles) == 1:
                        self.assertEqual(results, (('Bundle1', None),))
                    else:
                        assert False
                else:
                    assert False

        #
        # Test invoke_method
        #

        results = []
        results.append(manager.invoke_method("junk",
                                             method_args=(None,),
                                             method_kwds={'a': None},
                                             oneway=oneway,
                                             async=async))
        if not delay:
            if async:
                results[-1] = results[-1].complete()
        results.append(manager.invoke_method("junk",
                                             method_args=(None,),
                                             method_kwds={'a': None},
                                             worker_names=[manager.worker_names()[-1]],
                                             oneway=oneway,
                                             async=async))
        if not delay:
            if async:
                results[-1] = results[-1].complete()

        if delay:
            if async:
                results = [_result.complete() for _result in results]
        if oneway:
            for _result in results:
                self.assertEqual(id(_result), id(None))
        else:
            self.assertEqual(sorted(results[0].keys()),
                             sorted(manager.worker_names()))
            for worker_name in results[0]:
                self.assertEqual(results[0][worker_name], ((None,), {'a': None}))
            self.assertEqual(len(results[1].keys()), 1)
            self.assertEqual(list(results[1].keys())[0],
                             manager.worker_names()[-1])
            for worker_name in results[1]:
                self.assertEqual(results[1][worker_name], ((None,), {'a': None}))

        #
        # Test invoke_method_on_worker
        #

        results = dict((worker_name,
                        manager.invoke_method_on_worker(worker_name,
                                                        "junk",
                                                        method_args=(None,),
                                                        method_kwds={'a': None},
                                                        oneway=oneway,
                                                        async=async))
                       for worker_name in manager.worker_names())
        if async:
            results = dict((worker_name, results[worker_name].complete())
                           for worker_name in results)

        if oneway:
            for worker_name in results:
                self.assertEqual(id(results[worker_name]), id(None))
        else:
            self.assertEqual(sorted(results.keys()),
                             sorted(manager.worker_names()))
            for worker_name in results:
                self.assertEqual(results[worker_name], ((None,), {'a': None}))

    @unittest.nottest
    def _scenarios_test(self, async=False, oneway=False, delay=False):
        self._setup(self.options)
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    @unittest.nottest
    def _bundles1_test(self, async=False, oneway=False, delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict1
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    @unittest.nottest
    def _bundles2_test(self, async=False, oneway=False, delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict2
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    @unittest.nottest
    def _bundles3_test(self, async=False, oneway=False, delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict3
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_scenarios(self):
        self._scenarios_test(async=False, oneway=False, delay=False)
    def test_scenarios_async(self):
        self._scenarios_test(async=True, oneway=False, delay=False)
    def test_scenarios_async_oneway(self):
        self._scenarios_test(async=True, oneway=True, delay=False)
    def test_scenarios_async_delay(self):
        self._scenarios_test(async=True, oneway=False, delay=True)
    def test_scenarios_async_oneway_delay(self):
        self._scenarios_test(async=True, oneway=True, delay=True)

    def test_bundles1(self):
        self._bundles1_test(async=False, oneway=False, delay=False)
    def test_bundles1_async(self):
        self._bundles1_test(async=True, oneway=False, delay=False)
    def test_bundles1_async_oneway(self):
        self._bundles1_test(async=True, oneway=True, delay=False)
    def test_bundles1_async_delay(self):
        self._bundles1_test(async=True, oneway=False, delay=True)
    def test_bundles1_async_oneway_delay(self):
        self._bundles1_test(async=True, oneway=True, delay=True)

    def test_bundles2(self):
        self._bundles2_test(async=False, oneway=False, delay=False)
    def test_bundles2_async(self):
        self._bundles2_test(async=True, oneway=False, delay=False)
    def test_bundles2_async_oneway(self):
        self._bundles2_test(async=True, oneway=True, delay=False)
    def test_bundles2_async_delay(self):
        self._bundles2_test(async=True, oneway=False, delay=True)
    def test_bundles2_async_oneway_delay(self):
        self._bundles2_test(async=True, oneway=True, delay=True)

    def test_bundles3(self):
        self._bundles3_test(async=False, oneway=False, delay=False)
    def test_bundles3_async(self):
        self._bundles3_test(async=True, oneway=False, delay=False)
    def test_bundles3_async_oneway(self):
        self._bundles3_test(async=True, oneway=True, delay=False)
    def test_bundles3_async_delay(self):
        self._bundles3_test(async=True, oneway=False, delay=True)
    def test_bundles3_async_oneway_delay(self):
        self._bundles3_test(async=True, oneway=True, delay=True)

    def test_random_bundles(self):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.create_random_bundles = 2
        with self.cls(self.options, **_init_kwds) as manager:
            manager.initialize()
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

#
# create the actual testing classes
#

@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSerial(unittest.TestCase, _ScenarioTreeManagerTesterBase):

    cls = _ScenarioTreeManagerTestSerial

    def setUp(self):
        self.options = PySPConfigBlock()
        self.async = False
        self.oneway = False
        self.delay = False
        ScenarioTreeManagerSerial.register_options(self.options)

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
                                 ["--import-module="+thisfile] + \
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
    if os.path.exists(os.path.join(thisdir, "Pyro_NS_URI")):
        try:
            os.remove(os.path.join(thisdir, "Pyro_NS_URI"))
        except OSError:
            pass

class _ScenarioTreeManagerSPPyroTesterBase(_ScenarioTreeManagerTesterBase):

    cls = ScenarioTreeManagerSPPyro

    def setUp(self):
        _setUpModule()
        [_poll(proc) for proc in _taskworker_processes]
        self.options = PySPConfigBlock()
        ScenarioTreeManagerSPPyro.register_options(
            self.options,
            registered_worker_name='ScenarioTreeWorkerTest')

    @unittest.nottest
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerTesterBase._setup(self, options)
        options.pyro_host = 'localhost'
        options.pyro_port = _pyomo_ns_port
        if servers is not None:
            options.sppyro_required_servers = servers

    @unittest.nottest
    def _scenarios_1server_test(self, async=False, oneway=False, delay=False):
        self._setup(self.options, servers=1)
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    @unittest.nottest
    def _bundles1_1server_test(self, async=False, oneway=False, delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict1
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    @unittest.nottest
    def _bundles2_1server_test(self, async=False, oneway=False, delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict2
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    @unittest.nottest
    def _bundles3_1server_test(self, async=False, oneway=False, delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict3
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

    def test_scenarios_1server(self):
        self._scenarios_1server_test(async=False, oneway=False, delay=False)
    def test_scenarios_1server_async(self):
        self._scenarios_1server_test(async=True, oneway=False, delay=False)
    def test_scenarios_1server_async_oneway(self):
        self._scenarios_1server_test(async=True, oneway=True, delay=False)
    def test_scenarios_1server_async_delay(self):
        self._scenarios_1server_test(async=True, oneway=False, delay=True)
    def test_scenarios_1server_async_oneway_delay(self):
        self._scenarios_1server_test(async=True, oneway=True, delay=True)

    def test_bundles1_1server(self):
        self._bundles1_1server_test(async=False, oneway=False, delay=False)
    def test_bundles1_1server_async(self):
        self._bundles1_1server_test(async=True, oneway=False, delay=False)
    def test_bundles1_1server_async_oneway(self):
        self._bundles1_1server_test(async=True, oneway=True, delay=False)
    def test_bundles1_1server_async_delay(self):
        self._bundles1_1server_test(async=True, oneway=False, delay=True)
    def test_bundles1_1server_async_oneway_delay(self):
        self._bundles1_test(async=True, oneway=True, delay=True)

    def test_bundles2_1server(self):
        self._bundles2_1server_test(async=False, oneway=False, delay=False)
    def test_bundles2_1server_async(self):
        self._bundles2_1server_test(async=True, oneway=False, delay=False)
    def test_bundles2_1server_async_oneway(self):
        self._bundles2_1server_test(async=True, oneway=True, delay=False)
    def test_bundles2_1server_async_delay(self):
        self._bundles2_1server_test(async=True, oneway=False, delay=True)
    def test_bundles2_1server_async_oneway_delay(self):
        self._bundles2_1server_test(async=True, oneway=True, delay=True)

    def test_bundles3_1server(self):
        self._bundles3_1server_test(async=False, oneway=False, delay=False)
    def test_bundles3_1server_async(self):
        self._bundles3_1server_test(async=True, oneway=False, delay=False)
    def test_bundles3_1server_async_oneway(self):
        self._bundles3_1server_test(async=True, oneway=True, delay=False)
    def test_bundles3_1server_async_delay(self):
        self._bundles3_1server_test(async=True, oneway=False, delay=True)
    def test_bundles3_1server_async_oneway_delay(self):
        self._bundles3_1server_test(async=True, oneway=True, delay=True)

    def test_random_bundles_1server(self):
        options = PySPConfigBlock()
        self._setup(self.options, servers=1)
        self.options.create_random_bundles = 2
        with self.cls(self.options, **_init_kwds) as manager:
            manager.initialize()
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(len(list(self.options.unused_user_values())), 0)

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyro(unittest.TestCase,
                                    _ScenarioTreeManagerSPPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.sppyro_handshake_at_startup = False
        options.sppyro_multiple_server_workers = False

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyro_MultipleWorkers(
        unittest.TestCase,
        _ScenarioTreeManagerSPPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.sppyro_handshake_at_startup = False
        options.sppyro_multiple_server_workers = True

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyro_HandshakeAtStartup(
        unittest.TestCase,
        _ScenarioTreeManagerSPPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.sppyro_handshake_at_startup = True
        options.sppyro_multiple_server_workers = False

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('nightly','expensive')
class TestScenarioTreeManagerSPPyro_HandshakeAtStartup_MultipleWorkers(
        unittest.TestCase,
        _ScenarioTreeManagerSPPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerSPPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerSPPyroTesterBase._setup(self, options, servers=servers)
        options.sppyro_handshake_at_startup = True
        options.sppyro_multiple_server_workers = True

if __name__ == "__main__":
    unittest.main()
