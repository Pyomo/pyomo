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
from pyomo.pysp.scenariotree.manager import (ScenarioTreeManagerClientSerial,
                                             ScenarioTreeManagerClientPyro,
                                             InvocationType)
from pyomo.pysp.scenariotree.manager_worker_pyro import ScenarioTreeManagerWorkerPyro
from pyomo.pysp.scenariotree.server_pyro import (RegisterWorker,
                                                 ScenarioTreeServerPyro)

from pyomo.environ import *

thisfile = os.path.abspath(__file__)
thisdir = os.path.dirname(thisfile)

_run_verbose = True
_run_profile_memory = False

class _ScenarioTreeManagerWorkerTest(ScenarioTreeManagerWorkerPyro):

    def junk(self, *args, **kwds):
        return (args, kwds)

class _ScenarioTreeManagerClientTestSerial(ScenarioTreeManagerClientSerial):

    def __init__(self, *args, **kwds):
        assert kwds.pop('registered_worker_name', None) == 'ScenarioTreeManagerWorkerTest'
        super(_ScenarioTreeManagerClientTestSerial, self).__init__(*args, **kwds)

    def junk(self, *args, **kwds):
        return (args, kwds)

if "ScenarioTreeManagerWorkerTest" not in ScenarioTreeServerPyro._registered_workers:
    RegisterWorker("ScenarioTreeManagerWorkerTest", _ScenarioTreeManagerWorkerTest)

_init_kwds = {'registered_worker_name': 'ScenarioTreeManagerWorkerTest'}

def _Single(worker):
    return {'scenarios': [(scenario.name, scenario.probability) for scenario in worker.scenario_tree.scenarios],
            'bundles': [(bundle.name, bundle.probability) for bundle in worker.scenario_tree.bundles]}

def _PerScenario(worker, scenario):
    return scenario.name

def _PerBundle(worker, bundle):
    return bundle.name

def _PerScenarioChained(worker, scenario, data):
    return ((scenario.name, data),)

def _PerScenarioChained_noargs(worker, scenario):
    return scenario.name

def _PerBundleChained(worker, bundle, data):
    return ((bundle.name, data),)

def _PerBundleChained_noargs(worker, bundle):
    return bundle.name

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
        options.profile_memory = _run_profile_memory
        options.scenario_tree_random_seed = 1

    @unittest.nottest
    def _run_function_tests(self, manager, async=False, oneway=False, delay=False):
        assert not (async and oneway)
        class_name, test_name = self.id().split('.')[-2:]
        print("Running function tests on %s.%s" % (class_name, test_name))
        data = {}
        init = manager.initialize(async=async)
        if async:
            init = init.complete()
        self.assertEqual(all(_v is True for _v in init.values()), True)
        self.assertEqual(sorted(init.keys()), sorted(manager.worker_names))
        self.assertEqual(len(manager.scenario_tree.scenarios) > 0, True)
        if manager.scenario_tree.contains_bundles():
            self.assertEqual(len(manager.scenario_tree.bundles) > 0, True)
        else:
            self.assertEqual(len(manager.scenario_tree.bundles), 0)

        #
        # test invoke_function
        #

        print("")
        print("Running InvocationType.Single...")

        # make sure deprecated invocation types are converted
        results = manager.invoke_function(
            "_Single",
            thisfile,
            invocation_type=InvocationType.Single,
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_Single'] = results
        print("Running InvocationType.PerScenario...")
        results = manager.invoke_function(
            "_PerScenario",
            thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_PerScenario'] = results
        print("Running InvocationType.PerScenarioChained...")
        results = manager.invoke_function(
            "_PerScenarioChained",
            thisfile,
            function_args=(None,),
            invocation_type=InvocationType.PerScenarioChained,
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_PerScenarioChained'] = results
        print("Running InvocationType.PerScenarioChained (no args)...")
        results = manager.invoke_function(
            "_PerScenarioChained_noargs",
            thisfile,
            invocation_type=InvocationType.PerScenarioChained,
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_PerScenarioChained_noargs'] = results

        print("Running InvocationType.OnScenario...")
        results = manager.invoke_function(
            "_PerScenario",
            thisfile,
            invocation_type=InvocationType.OnScenario('Scenario1'),
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_OnScenario'] = results
        print("Running InvocationType.OnScenarios...")
        results = manager.invoke_function(
            "_PerScenario",
            thisfile,
            invocation_type=InvocationType.OnScenarios(['Scenario1', 'Scenario3']),
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_OnScenarios'] = results
        print("Running InvocationType.OnScenariosChained...")
        results = manager.invoke_function(
            "_PerScenarioChained",
            thisfile,
            function_args=(None,),
            invocation_type=InvocationType.OnScenariosChained(['Scenario1', 'Scenario3']),
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_OnScenariosChained'] = results
        print("Running InvocationType.OnScenariosChained (no args)...")
        results = manager.invoke_function(
            "_PerScenarioChained_noargs",
            thisfile,
            invocation_type=InvocationType.OnScenariosChained(['Scenario3', 'Scenario2']),
            oneway=oneway,
            async=async)
        if not delay:
            if async:
                results = results.complete()
        data['_OnScenariosChained_noargs'] = results

        if manager.scenario_tree.contains_bundles():
            print("Running InvocationType.PerBundle...")
            results = manager.invoke_function(
                "_PerBundle",
                thisfile,
                invocation_type=InvocationType.PerBundle,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_PerBundle'] = results
            print("Running InvocationType.PerBundleChained...")
            results = manager.invoke_function(
                "_PerBundleChained",
                thisfile,
                function_args=(None,),
                invocation_type=InvocationType.PerBundleChained,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_PerBundleChained'] = results
            print("Running InvocationType.PerBundleChained (no args)...")
            results = manager.invoke_function(
                "_PerBundleChained_noargs",
                thisfile,
                invocation_type=InvocationType.PerBundleChained,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_PerBundleChained_noargs'] = results

            print("Running InvocationType.OnBundle...")
            results = manager.invoke_function(
                "_PerBundle",
                thisfile,
                invocation_type=InvocationType.OnBundle('Bundle1'),
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_OnBundle'] = results
            print("Running InvocationType.OnBundles...")
            if len(manager.scenario_tree.bundles) == 1:
                _bundle_names = [manager.scenario_tree.bundles[0].name]
            else:
                _bundle_names = [b.name for b in manager.scenario_tree.bundles[:-1]]
            results = manager.invoke_function(
                "_PerBundle",
                thisfile,
                invocation_type=InvocationType.OnBundles([b.name for b in manager.scenario_tree.bundles]),
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_OnBundles'] = results
            print("Running InvocationType.OnBundlesChained...")
            results = manager.invoke_function(
                "_PerBundleChained",
                thisfile,
                function_args=(None,),
                invocation_type=InvocationType.OnBundlesChained(_bundle_names),
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_OnBundlesChained'] = results
            print("Running InvocationType.OnBundlesChained (no args)...")
            results = manager.invoke_function(
                "_PerBundleChained_noargs",
                thisfile,
                invocation_type=InvocationType.OnBundlesChained(_bundle_names),
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_OnBundlesChained_noargs'] = results

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
                                     sorted(manager.worker_names))
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
                                     sorted([(_scenario.name, _scenario.probability) for _scenario
                                             in manager.scenario_tree.scenarios]))
                    self.assertEqual(sorted(bundles),
                                     sorted([(_bundle.name, _bundle.probability) for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_PerScenario":
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                    self.assertEqual(sorted(results.values()),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                elif name == "_PerScenarioChained":
                    self.assertEqual(
                        results, (('Scenario3', ('Scenario2', ('Scenario1', None))),))
                elif name == "_PerScenarioChained_noargs":
                    self.assertEqual(results, 'Scenario3')
                elif name == "_OnScenario":
                    self.assertEqual(results, 'Scenario1')
                elif name == "_OnScenarios":
                    self.assertEqual(sorted(results.keys()),
                                     ['Scenario1', 'Scenario3'])
                    self.assertEqual(sorted(results.values()),
                                     ['Scenario1', 'Scenario3'])
                elif name == "_OnScenariosChained":
                    self.assertEqual(
                        results, (('Scenario3', ('Scenario1', None)),))
                elif name == "_OnScenariosChained_noargs":
                    self.assertEqual(results, 'Scenario2')
                elif name == "_PerBundle":
                    self.assertEqual(manager.scenario_tree.contains_bundles(), True)
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_bundle.name for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_PerBundleChained":
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
                elif name == "_PerBundleChained_noargs":
                    self.assertEqual(manager.scenario_tree.contains_bundles(), True)
                    if len(manager.scenario_tree.bundles) == 3:
                        self.assertEqual(results, 'Bundle3')
                    elif len(manager.scenario_tree.bundles) == 2:
                        self.assertEqual(results, 'Bundle2')
                    elif len(manager.scenario_tree.bundles) == 1:
                        self.assertEqual(results, 'Bundle1')
                    else:
                        assert False
                elif name == "_OnBundle":
                    self.assertEqual(results, 'Bundle1')
                elif name == "_OnBundles":
                    self.assertEqual(
                        sorted(results.keys()),
                        sorted([b.name for b in manager.scenario_tree.bundles]))
                    self.assertEqual(
                        sorted(results.values()),
                        sorted([b.name for b in manager.scenario_tree.bundles]))
                elif name == "_OnBundlesChained":
                    test_results = (None,)
                    for bundle_name in _bundle_names:
                        test_results = ((bundle_name, test_results[0]),)
                    self.assertEqual(results, test_results)
                elif name == "_OnBundlesChained_noargs":
                    self.assertEqual(results, _bundle_names[-1])
                else:
                    assert False

        if isinstance(manager, ScenarioTreeManagerClientSerial):
            self.assertEqual(manager._aggregate_user_data['leaf_node'],
                             [scenario.leaf_node.name for scenario
                              in manager.scenario_tree.scenarios])
            self.assertEqual(len(manager._aggregate_user_data['names']), 0)
        elif isinstance(manager, ScenarioTreeManagerClientPyro):
            self.assertEqual(manager._aggregate_user_data['leaf_node'],
                             [scenario.leaf_node.name for scenario
                              in manager.scenario_tree.scenarios])
            self.assertEqual(manager._aggregate_user_data['names'],
                             [scenario.name for scenario
                              in manager.scenario_tree.scenarios])
        else:
            assert False

        #
        # Test invoke_function_on_worker
        #

        data = {}
        print("")
        print("Running InvocationType.Single on individual workers...")
        data['_Single'] = {}
        for worker_name in manager.worker_names:
            results = manager.invoke_function_on_worker(
                worker_name,
                "_Single",
                thisfile,
                invocation_type=InvocationType.Single,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_Single'][worker_name] = results
        print("Running InvocationType.PerScenario on individual workers...")
        data['_PerScenario'] = {}
        for worker_name in manager.worker_names:
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenario",
                thisfile,
                invocation_type=InvocationType.PerScenario,
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_PerScenario'][worker_name] = results
        print("Running InvocationType.PerScenarioChained on individual workers...")
        data['_PerScenarioChained'] = {}
        results = (None,)
        for worker_name in manager.worker_names:
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenarioChained",
                thisfile,
                function_args=results,
                invocation_type=InvocationType.PerScenarioChained,
                async=async)
            if async:
                results = results.complete()
        data['_PerScenarioChained'] = results
        print("Running InvocationType.PerScenarioChained (no args) on individual workers...")
        data['_PerScenarioChained_noargs'] = {}
        for worker_name in manager.worker_names:
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenarioChained_noargs",
                thisfile,
                invocation_type=InvocationType.PerScenarioChained,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_PerScenarioChained_noargs'][worker_name] = results

        print("Running InvocationType.OnScenario on individual workers...")
        data['_OnScenario'] = {}
        for worker_name in manager.worker_names:
            assert len(manager.get_scenarios_for_worker(worker_name)) > 0
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenario",
                thisfile,
                invocation_type=InvocationType.OnScenario(manager.get_scenarios_for_worker(worker_name)[0]),
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_OnScenario'][worker_name] = results
        print("Running InvocationType.OnScenarios on individual workers...")
        data['_OnScenarios'] = {}
        for worker_name in manager.worker_names:
            assert len(manager.get_scenarios_for_worker(worker_name)) > 0
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenario",
                thisfile,
                invocation_type=InvocationType.OnScenarios(manager.get_scenarios_for_worker(worker_name)),
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_OnScenarios'][worker_name] = results
        print("Running InvocationType.OnScenariosChained on individual workers...")
        data['_OnScenariosChained'] = {}
        results = (None,)
        for worker_name in manager.worker_names:
            assert len(manager.get_scenarios_for_worker(worker_name)) > 0
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenarioChained",
                thisfile,
                function_args=results,
                invocation_type=InvocationType.OnScenariosChained(manager.get_scenarios_for_worker(worker_name)),
                oneway=oneway,
                async=async)
            if async:
                results = results.complete()
        data['_OnScenariosChained'] = results
        print("Running InvocationType.OnScenariosChained (no args) on individual workers...")
        data['_OnScenariosChained_noargs'] = {}
        for worker_name in manager.worker_names:
            assert len(manager.get_scenarios_for_worker(worker_name)) > 0
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenarioChained_noargs",
                thisfile,
                invocation_type=InvocationType.OnScenariosChained(manager.get_scenarios_for_worker(worker_name)),
                oneway=oneway,
                async=async)
            if not delay:
                if async:
                    results = results.complete()
            data['_OnScenariosChained_noargs'][worker_name] = results

        if manager.scenario_tree.contains_bundles():
            print("Running InvocationType.PerBundle on individual workers...")
            data['_PerBundle'] = {}
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundle",
                    thisfile,
                    invocation_type=InvocationType.PerBundle,
                    oneway=oneway,
                    async=async)
                if not delay:
                    if async:
                        results = results.complete()
                data['_PerBundle'][worker_name] = results
            print("Running InvocationType.PerBundleChained on individual workers...")
            data['_PerBundleChained'] = {}
            results = (None,)
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained",
                    thisfile,
                    function_args=results,
                    invocation_type=InvocationType.PerBundleChained,
                    async=async)
                if async:
                    results = results.complete()
            data['_PerBundleChained'] = results
            print("Running InvocationType.PerBundleChained (no args) on individual workers...")
            data['_PerBundleChained_noargs'] = {}
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained_noargs",
                    thisfile,
                    invocation_type=InvocationType.PerBundleChained,
                    async=async)
                if not delay:
                    if async:
                        results = results.complete()
                data['_PerBundleChained_noargs'][worker_name] = results

            print("Running InvocationType.OnBundle on individual workers...")
            data['_OnBundle'] = {}
            for worker_name in manager.worker_names:
                assert len(manager.get_scenarios_for_worker(worker_name)) > 0
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundle",
                    thisfile,
                    invocation_type=InvocationType.OnBundle(manager.get_bundles_for_worker(worker_name)[0]),
                    oneway=oneway,
                    async=async)
                if not delay:
                    if async:
                        results = results.complete()
                data['_OnBundle'][worker_name] = results
            print("Running InvocationType.OnBundles on individual workers...")
            data['_OnBundles'] = {}
            for worker_name in manager.worker_names:
                assert len(manager.get_scenarios_for_worker(worker_name)) > 0
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundle",
                    thisfile,
                    invocation_type=InvocationType.OnBundles(manager.get_bundles_for_worker(worker_name)),
                    oneway=oneway,
                    async=async)
                if not delay:
                    if async:
                        results = results.complete()
                data['_OnBundles'][worker_name] = results
            print("Running InvocationType.OnBundlesChained on individual workers...")
            data['_OnBundlesChained'] = {}
            results = (None,)
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained",
                    thisfile,
                    function_args=results,
                    invocation_type=InvocationType.OnBundlesChained(manager.get_bundles_for_worker(worker_name)),
                    async=async)
                if async:
                    results = results.complete()
            data['_OnBundlesChained'] = results
            print("Running InvocationType.OnBundlesChained (no args) on individual workers...")
            data['_OnBundlesChained_noargs'] = {}
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained_noargs",
                    thisfile,
                    invocation_type=InvocationType.OnBundlesChained(manager.get_bundles_for_worker(worker_name)),
                    async=async)
                if not delay:
                    if async:
                        results = results.complete()
                data['_OnBundlesChained_noargs'][worker_name] = results

        print("")
        for name in data:
            results = data[name]
            if (name != '_PerScenarioChained') and \
               (name != '_PerBundleChained') and \
               (name != '_OnScenariosChained') and \
               (name != '_OnBundlesChained'):
                if delay:
                    if async:
                        for worker_name in results:
                            results[worker_name] = results[worker_name].complete()
            if oneway:
                if (name != '_PerScenarioChained') and \
                   (name != '_PerBundleChained') and \
                   (name != '_OnScenariosChained') and \
                   (name != '_OnBundlesChained'):
                    for worker_name in results:
                        self.assertEqual(id(results[worker_name]), id(None))
            else:
                if name == "_Single":
                    self.assertEqual(sorted(results.keys()),
                                     sorted(manager.worker_names))
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
                                     sorted([(_scenario.name, _scenario.probability) for _scenario
                                             in manager.scenario_tree.scenarios]))
                    self.assertEqual(sorted(bundles),
                                     sorted([(_bundle.name, _bundle.probability) for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_PerScenario":
                    _results = {}
                    for worker_name in results:
                        _results.update(results[worker_name])
                    results = _results
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                    self.assertEqual(sorted(results.values()),
                                     sorted([_scenario.name for _scenario
                                             in manager.scenario_tree.scenarios]))
                elif name == "_PerScenarioChained":
                    self.assertEqual(
                        results, (('Scenario3', ('Scenario2', ('Scenario1', None))),))
                elif name == "_PerScenarioChained_noargs":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(
                            results[worker_name],
                            manager.get_scenarios_for_worker(worker_name)[-1])
                elif name == "_OnScenario":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(results[worker_name],
                                         manager.get_scenarios_for_worker(worker_name)[0])
                elif name == "_OnScenarios":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(
                            sorted(results[worker_name].keys()),
                            sorted(manager.get_scenarios_for_worker(worker_name)))
                        self.assertEqual(
                            sorted(results[worker_name].values()),
                            sorted(manager.get_scenarios_for_worker(worker_name)))
                elif name == "_OnScenariosChained":
                    self.assertEqual(
                        results, (('Scenario3', ('Scenario2', ('Scenario1', None))),))
                elif name == "_OnScenariosChained_noargs":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(
                            results[worker_name],
                            manager.get_scenarios_for_worker(worker_name)[-1])
                elif name == "_PerBundle":
                    _results = {}
                    for worker_name in results:
                        _results.update(results[worker_name])
                    results = _results
                    self.assertEqual(manager.scenario_tree.contains_bundles(), True)
                    self.assertEqual(sorted(results.keys()),
                                     sorted([_bundle.name for _bundle
                                             in manager.scenario_tree.bundles]))
                elif name == "_PerBundleChained":
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
                elif name == "_PerBundleChained_noargs":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(
                            results[worker_name],
                            manager.get_bundles_for_worker(worker_name)[-1])
                elif name == "_OnBundle":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(
                            results[worker_name],
                            manager.get_bundles_for_worker(worker_name)[0])
                elif name == "_OnBundles":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(
                            sorted(results[worker_name].keys()),
                            sorted(manager.get_bundles_for_worker(worker_name)))
                        self.assertEqual(
                            sorted(results[worker_name].values()),
                            sorted(manager.get_bundles_for_worker(worker_name)))
                elif name == "_OnBundlesChained":
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
                elif name == "_OnBundlesChained_noargs":
                    assert len(results) > 0
                    for worker_name in results:
                        self.assertEqual(
                            results[worker_name],
                            manager.get_bundles_for_worker(worker_name)[-1])
                else:
                    assert False

        #
        # Test invoke_method
        #

        result = manager.invoke_method(
            "junk",
            method_args=(None,),
            method_kwds={'a': None},
            oneway=oneway,
            async=async)
        if async:
            result = result.complete()
        if oneway:
            self.assertEqual(id(result), id(None))
        else:
            self.assertEqual(sorted(result.keys()),
                             sorted(manager.worker_names))
            for worker_name in result:
                self.assertEqual(result[worker_name],
                                 ((None,), {'a': None}))


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
                       for worker_name in manager.worker_names)
        if async:
            results = dict((worker_name, results[worker_name].complete())
                           for worker_name in results)

        if oneway:
            for worker_name in results:
                self.assertEqual(id(results[worker_name]), id(None))
        else:
            self.assertEqual(sorted(results.keys()),
                             sorted(manager.worker_names))
            for worker_name in results:
                self.assertEqual(results[worker_name], ((None,), {'a': None}))

    @unittest.nottest
    def _scenarios_test(self,
                        async=False,
                        oneway=False,
                        delay=False):
        self._setup(self.options)
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async=async,
                                     oneway=oneway,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles1_test(self,
                       async=False,
                       oneway=False,
                       delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict1
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async=async,
                                     oneway=oneway,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles2_test(self,
                       async=False,
                       oneway=False,
                       delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict2
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async=async,
                                     oneway=oneway,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles3_test(self,
                       async=False,
                       oneway=False,
                       delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict3
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async=async,
                                     oneway=oneway,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    def test_scenarios(self):
        self._scenarios_test(async=False,
                             oneway=False,
                             delay=False)
    def test_scenarios_async(self):
        self._scenarios_test(async=True,
                             oneway=False,
                             delay=False)
    def test_scenarios_async_delay(self):
        self._scenarios_test(async=True,
                             oneway=False,
                             delay=True)

    def test_bundles1(self):
        self._bundles1_test(async=False,
                            oneway=False,
                            delay=False)
    def test_bundles1_async(self):
        self._bundles1_test(async=True,
                            oneway=False,
                            delay=False)
    def test_bundles1_async_delay(self):
        self._bundles1_test(async=True,
                            oneway=False,
                            delay=True)

    def test_bundles2(self):
        self._bundles2_test(async=False,
                            oneway=False,
                            delay=False)
    def test_bundles2_async(self):
        self._bundles2_test(async=True,
                            oneway=False,
                            delay=False)
    def test_bundles2_async_delay(self):
        self._bundles2_test(async=True,
                            oneway=False,
                            delay=True)

    def test_bundles3(self):
        self._bundles3_test(async=False,
                            oneway=False,
                            delay=False)
    def test_bundles3_async(self):
        self._bundles3_test(async=True,
                            oneway=False,
                            delay=False)
    def test_bundles3_async_delay(self):
        self._bundles3_test(async=True,
                            oneway=False,
                            delay=True)

    def test_random_bundles(self):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.create_random_bundles = 2
        with self.cls(self.options, **_init_kwds) as manager:
            manager.initialize()
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

#
# create the actual testing classes
#

@unittest.category('smoke','nightly','expensive')
class TestScenarioTreeManagerClientSerial(unittest.TestCase, _ScenarioTreeManagerTesterBase):

    cls = _ScenarioTreeManagerClientTestSerial

    def setUp(self):
        self.options = PySPConfigBlock()
        self.async = False
        self.oneway = False
        self.delay = False
        ScenarioTreeManagerClientSerial.register_options(self.options)

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
    if os.path.exists(os.path.join(thisdir, "Pyro_NS_URI")):
        try:
            os.remove(os.path.join(thisdir, "Pyro_NS_URI"))
        except OSError:
            pass

class _ScenarioTreeManagerClientPyroTesterBase(_ScenarioTreeManagerTesterBase):

    cls = ScenarioTreeManagerClientPyro

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
                                         ["--import-module="+thisfile] + \
                                         (["--verbose"] if _run_verbose else []) + \
                                         ["--pyro-host="+str(_pyomo_ns_host)] + \
                                         ["--pyro-port="+str(_pyomo_ns_port)],
                                         stdout=f,
                                         stderr=subprocess.STDOUT))

            time.sleep(2)
            [_poll(proc) for proc in _taskworker_processes]

    def _cleanup(self):
        for fname in self._tempfiles:
            try:
                os.remove(fname)
            except OSError:
                pass
        self._tempfiles = []

    def setUp(self):
        self._tempfiles = []
        self._setUpPyro()
        [_poll(proc) for proc in _taskworker_processes]
        self.options = PySPConfigBlock()
        ScenarioTreeManagerClientPyro.register_options(
            self.options,
            registered_worker_name='ScenarioTreeManagerWorkerTest')

    @unittest.nottest
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerTesterBase._setup(self, options)
        options.pyro_host = 'localhost'
        options.pyro_port = _pyomo_ns_port
        if servers is not None:
            options.pyro_required_scenariotreeservers = servers

    @unittest.nottest
    def _scenarios_1server_test(self,
                                async=False,
                                oneway=False,
                                delay=False):
        self._setup(self.options, servers=1)
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async=async,
                                     oneway=oneway,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(list(self.options.unused_user_values()), [])
        self._cleanup()

    @unittest.nottest
    def _bundles1_1server_test(self,
                               async=False,
                               oneway=False,
                               delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict1
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async=async,
                                     oneway=oneway,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])
        self._cleanup()

    @unittest.nottest
    def _bundles2_1server_test(self,
                               async=False,
                               oneway=False,
                               delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict2
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])
        self._cleanup()

    @unittest.nottest
    def _bundles3_1server_test(self,
                               async=False,
                               oneway=False,
                               delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict3
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async=async, oneway=oneway, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])
        self._cleanup()

    def test_scenarios_1server(self):
        self._scenarios_1server_test(async=False,
                                     oneway=False,
                                     delay=False)
    def test_scenarios_1server_async(self):
        self._scenarios_1server_test(async=True,
                                     oneway=False,
                                     delay=False)
    def test_scenarios_1server_async_delay(self):
        self._scenarios_1server_test(async=True,
                                     oneway=False,
                                     delay=True)

    def test_bundles1_1server(self):
        self._bundles1_1server_test(async=False,
                                    oneway=False,
                                    delay=False)
    def test_bundles1_1server_async(self):
        self._bundles1_1server_test(async=True,
                                    oneway=False,
                                    delay=False)
    def test_bundles1_1server_async_delay(self):
        self._bundles1_1server_test(async=True,
                                    oneway=False,
                                    delay=True)

    def test_bundles2_1server(self):
        self._bundles2_1server_test(async=False,
                                    oneway=False,
                                    delay=False)
    def test_bundles2_1server_async(self):
        self._bundles2_1server_test(async=True,
                                    oneway=False,
                                    delay=False)
    def test_bundles2_1server_async_delay(self):
        self._bundles2_1server_test(async=True,
                                    oneway=False,
                                    delay=True)

    def test_bundles3_1server(self):
        self._bundles3_1server_test(async=False,
                                    oneway=False,
                                    delay=False)
    def test_bundles3_1server_async(self):
        self._bundles3_1server_test(async=True,
                                    oneway=False,
                                    delay=False)
    def test_bundles3_1server_async_delay(self):
        self._bundles3_1server_test(async=True,
                                    oneway=False,
                                    delay=True)

    def test_random_bundles_1server(self):
        options = PySPConfigBlock()
        self._setup(self.options, servers=1)
        self.options.create_random_bundles = 2
        with self.cls(self.options, **_init_kwds) as manager:
            manager.initialize()
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('parallel')
class TestScenarioTreeManagerClientPyro(unittest.TestCase,
                                        _ScenarioTreeManagerClientPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerClientPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerClientPyroTesterBase._setup(self, options, servers=servers)
        options.pyro_handshake_at_startup = False
        options.pyro_multiple_scenariotreeserver_workers = False

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('parallel')
class TestScenarioTreeManagerClientPyro_MultipleWorkers(
        unittest.TestCase,
        _ScenarioTreeManagerClientPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerClientPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerClientPyroTesterBase._setup(self, options, servers=servers)
        options.pyro_handshake_at_startup = False
        options.pyro_multiple_scenariotreeserver_workers = True

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('parallel')
class TestScenarioTreeManagerClientPyro_HandshakeAtStartup(
        unittest.TestCase,
        _ScenarioTreeManagerClientPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerClientPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerClientPyroTesterBase._setup(self, options, servers=servers)
        options.pyro_handshake_at_startup = True
        options.pyro_multiple_scenariotreeserver_workers = False

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('parallel')
class TestScenarioTreeManagerClientPyro_HandshakeAtStartup_MultipleWorkers(
        unittest.TestCase,
        _ScenarioTreeManagerClientPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerClientPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerClientPyroTesterBase._setup(self, options, servers=servers)
        options.pyro_handshake_at_startup = True
        options.pyro_multiple_scenariotreeserver_workers = True

if __name__ == "__main__":
    unittest.main()
