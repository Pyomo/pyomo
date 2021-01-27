#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import os
import time
import subprocess
import sys

from pyomo.common.collections import OrderedDict
if sys.version_info[:2] >= (3,7):
    # dict became ordered in CPython 3.6 and added to the standard in 3.7
    _ordered_dict_ = dict
else:
    _ordered_dict_ = OrderedDict

from pyutilib.pyro import using_pyro3, using_pyro4
import pyutilib.th as unittest

from pyomo.common.dependencies import dill, dill_available
from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
from pyomo.pysp.util.config import PySPConfigBlock
from pyomo.pysp.scenariotree.manager import (ScenarioTreeManager,
                                             ScenarioTreeManagerClient,
                                             _ScenarioTreeManagerWorker,
                                             ScenarioTreeManagerClientSerial,
                                             ScenarioTreeManagerClientPyro,
                                             ScenarioTreeManagerFactory,
                                             InvocationType)
from pyomo.pysp.scenariotree.manager_worker_pyro import \
    ScenarioTreeManagerWorkerPyro
from pyomo.pysp.scenariotree.server_pyro import (RegisterWorker,
                                                 ScenarioTreeServerPyro)
from pyomo.pysp.scenariotree.tree_structure_model import \
    CreateConcreteTwoStageScenarioTreeModel
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory

from pyomo.environ import ConcreteModel, Var, Expression, Constraint, Objective, sum_product

thisfile = os.path.abspath(__file__)
thisdir = os.path.dirname(thisfile)

_run_verbose = True
_run_profile_memory = False

class TestScenarioTreeManagerMisc(unittest.TestCase):

    def test_factory(self):
        options = ScenarioTreeManagerFactory.register_options()
        tmp = ScenarioTreeManagerFactory.register_options(options)
        self.assertIs(tmp, options)

        with self.assertRaises(TypeError):
            ScenarioTreeManagerFactory.register_options(options, options)
        with self.assertRaises(TypeError):
            ScenarioTreeManagerFactory.register_options('a')

        options.scenario_tree_manager = 'junk'
        with self.assertRaises(ValueError):
            ScenarioTreeManagerFactory(options)

        options.scenario_tree_manager = 'serial'

        model = ConcreteModel()
        model.x = Var()
        model.y = Var(bounds=(1,None))
        model.stage_cost = Expression([1,2])
        model.stage_cost[1].expr = model.x
        model.stage_cost[2].expr = 0.0
        model.o = Objective(expr=sum_product(model.stage_cost))
        model.c = Constraint(expr=model.x >= model.y)

        scenario_tree_model = CreateConcreteTwoStageScenarioTreeModel(3)
        scenario_tree_model.StageCost['Stage1'] = 'stage_cost[1]'
        scenario_tree_model.StageCost['Stage2'] = 'stage_cost[2]'
        scenario_tree_model.NodeVariables['RootNode'].add('x')
        scenario_tree_model.StageVariables['Stage1'].add('y')

        instance_factory = ScenarioTreeInstanceFactory(
            model=model,
            scenario_tree=scenario_tree_model)

        manager = ScenarioTreeManagerFactory(options,
                                             factory=instance_factory)
        self.assertTrue(isinstance(manager, ScenarioTreeManagerClientSerial))
        self.assertEqual(manager.initialized, False)
        manager.initialize()
        self.assertEqual(manager.initialized, True)
        self.assertIs(manager.scenario_tree,
                      manager.uncompressed_scenario_tree)
        manager.close()

    def test_bad_init(self):
        with self.assertRaises(NotImplementedError):
            ScenarioTreeManager()
        with self.assertRaises(NotImplementedError):
            ScenarioTreeManagerClient()
        with self.assertRaises(NotImplementedError):
            _ScenarioTreeManagerWorker()

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

    _bundle_dict3 = _ordered_dict_()
    _bundle_dict3['Bundle1'] = ['Scenario1']
    _bundle_dict3['Bundle2'] = ['Scenario2']
    _bundle_dict3['Bundle3'] = ['Scenario3']

    _bundle_dict2 = _ordered_dict_()
    _bundle_dict2['Bundle1'] = ['Scenario1', 'Scenario2']
    _bundle_dict2['Bundle2'] = ['Scenario3']

    _bundle_dict1 = _ordered_dict_()
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
    def _run_function_tests(self, manager, async_call=False, oneway_call=False, delay=False):
        assert not (async_call and oneway_call)
        class_name, test_name = self.id().split('.')[-2:]
        print("Running function tests on %s.%s" % (class_name, test_name))
        data = []
        init = manager.initialize(async_call=async_call)
        if async_call:
            init = init.complete()
        self.assertEqual(all(_v is True for _v in init.values()), True)
        self.assertEqual(sorted(init.keys()), sorted(manager.worker_names))
        self.assertEqual(len(manager.scenario_tree.scenarios) > 0, True)
        if manager.scenario_tree.contains_bundles():
            self.assertEqual(len(manager.scenario_tree.bundles) > 0, True)
        else:
            self.assertEqual(len(manager.scenario_tree.bundles), 0)

        with self.assertRaises(KeyError):
            manager.get_worker_for_scenario("_not_a_scenario_name")
        with self.assertRaises(KeyError):
            manager.get_worker_for_bundle("_not_a_bundles_name")
        with self.assertRaises(KeyError):
            manager.get_scenarios_for_worker("_not_a_worker_name")
        with self.assertRaises(KeyError):
            manager.get_bundles_for_worker("_not_a_worker_name")

        #
        # test invoke_function
        #

        # no module name
        with self.assertRaises(ValueError):
            manager.invoke_function(
                "_Single",
                module_name=None,
                invocation_type=InvocationType.Single,
                oneway_call=oneway_call,
                async_call=async_call)
        # bad invocation type
        with self.assertRaises(ValueError):
            manager.invoke_function(
                "_Single",
                module_name=thisfile,
                invocation_type="_not_an_invocation_type_",
                oneway_call=oneway_call,
                async_call=async_call)
        # bad oneway_call and async_call combo
        with self.assertRaises(ValueError):
            manager.invoke_function(
                "_Single",
                module_name=thisfile,
                invocation_type=InvocationType.Single,
                oneway_call=True,
                async_call=True)
        # bad paused state
        if isinstance(manager, ScenarioTreeManagerClientPyro) and \
           (not async_call) and (not oneway_call) and (not delay):
            self.assertEqual(manager._transmission_paused, False)
            manager.pause_transmit()
            self.assertEqual(manager._transmission_paused, True)
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_Single",
                    module_name=thisfile,
                    invocation_type=InvocationType.Single,
                    oneway_call=oneway_call,
                    async_call=async_call)
            manager.unpause_transmit()
            self.assertEqual(manager._transmission_paused, False)
        if dill_available or \
           isinstance(manager, ScenarioTreeManagerClientPyro):
            print("")
            print("Running InvocationType.Single... (using dill)")
            results = manager.invoke_function(
                _Single,
                invocation_type=InvocationType.Single,
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_Single', results))
            if isinstance(manager, ScenarioTreeManagerClientPyro):
                # module name must be None
                with self.assertRaises(ValueError):
                    manager.invoke_function(
                        _Single,
                        module_name=thisfile,
                        invocation_type=InvocationType.Single,
                        oneway_call=oneway_call,
                        async_call=async_call)

        elif isinstance(manager, ScenarioTreeManagerClientPyro):
            # requires dill
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    _Single,
                    invocation_type=InvocationType.Single,
                    oneway_call=oneway_call,
                    async_call=async_call)
        print("")
        print("Running InvocationType.Single...")
        results = manager.invoke_function(
            "_Single",
            module_name=thisfile,
            invocation_type=InvocationType.Single,
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_Single', results))
        print("Running InvocationType.PerScenario...")
        results = manager.invoke_function(
            "_PerScenario",
            module_name=thisfile,
            invocation_type=InvocationType.PerScenario,
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_PerScenario', results))
        print("Running InvocationType.PerScenarioChained...")
        results = manager.invoke_function(
            "_PerScenarioChained",
            module_name=thisfile,
            function_args=(None,),
            invocation_type=InvocationType.PerScenarioChained,
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_PerScenarioChained', results))
        print("Running InvocationType.PerScenarioChained (no args)...")
        results = manager.invoke_function(
            "_PerScenarioChained_noargs",
            module_name=thisfile,
            invocation_type=InvocationType.PerScenarioChained,
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_PerScenarioChained_noargs', results))

        # bad paused state
        if isinstance(manager, ScenarioTreeManagerClientPyro) and \
           (not delay):
            self.assertEqual(manager._transmission_paused, False)
            manager.pause_transmit()
            self.assertEqual(manager._transmission_paused, True)
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_PerScenarioChained",
                    module_name=thisfile,
                    function_args=(None,),
                    invocation_type=InvocationType.PerScenarioChained,
                    oneway_call=oneway_call,
                    async_call=async_call)
            manager.unpause_transmit()
            self.assertEqual(manager._transmission_paused, False)

        print("Running InvocationType.OnScenario...")
        results = manager.invoke_function(
            "_PerScenario",
            module_name=thisfile,
            invocation_type=InvocationType.OnScenario('Scenario1'),
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_OnScenario', results))
        print("Running InvocationType.OnScenarios...")
        results = manager.invoke_function(
            "_PerScenario",
            module_name=thisfile,
            invocation_type=InvocationType.OnScenarios(['Scenario1', 'Scenario3']),
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_OnScenarios', results))
        print("Running InvocationType.OnScenariosChained...")
        results = manager.invoke_function(
            "_PerScenarioChained",
            module_name=thisfile,
            function_args=(None,),
            invocation_type=InvocationType.OnScenariosChained(['Scenario1', 'Scenario3']),
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_OnScenariosChained', results))
        print("Running InvocationType.OnScenariosChained (no args)...")
        results = manager.invoke_function(
            "_PerScenarioChained_noargs",
            module_name=thisfile,
            invocation_type=InvocationType.OnScenariosChained(['Scenario3', 'Scenario2']),
            oneway_call=oneway_call,
            async_call=async_call)
        if not delay:
            if async_call:
                results = results.complete()
        data.append(('_OnScenariosChained_noargs', results))

        # bad paused state
        if isinstance(manager, ScenarioTreeManagerClientPyro) and \
           (not delay):
            self.assertEqual(manager._transmission_paused, False)
            manager.pause_transmit()
            self.assertEqual(manager._transmission_paused, True)
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_PerScenarioChained",
                    module_name=thisfile,
                    function_args=(None,),
                    invocation_type=InvocationType.OnScenariosChained(['Scenario1', 'Scenario3']),
                    oneway_call=oneway_call,
                    async_call=async_call)
            manager.unpause_transmit()
            self.assertEqual(manager._transmission_paused, False)

        if manager.scenario_tree.contains_bundles():
            print("Running InvocationType.PerBundle...")
            results = manager.invoke_function(
                "_PerBundle",
                module_name=thisfile,
                invocation_type=InvocationType.PerBundle,
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_PerBundle', results))
            print("Running InvocationType.PerBundleChained...")
            results = manager.invoke_function(
                "_PerBundleChained",
                module_name=thisfile,
                function_args=(None,),
                invocation_type=InvocationType.PerBundleChained,
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_PerBundleChained', results))
            print("Running InvocationType.PerBundleChained (no args)...")
            results = manager.invoke_function(
                "_PerBundleChained_noargs",
                module_name=thisfile,
                invocation_type=InvocationType.PerBundleChained,
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_PerBundleChained_noargs', results))

            print("Running InvocationType.OnBundle...")
            results = manager.invoke_function(
                "_PerBundle",
                module_name=thisfile,
                invocation_type=InvocationType.OnBundle('Bundle1'),
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_OnBundle', results))
            print("Running InvocationType.OnBundles...")
            if len(manager.scenario_tree.bundles) == 1:
                _bundle_names = [manager.scenario_tree.bundles[0].name]
            else:
                _bundle_names = [b.name for b in manager.scenario_tree.bundles[:-1]]
            results = manager.invoke_function(
                "_PerBundle",
                module_name=thisfile,
                invocation_type=InvocationType.OnBundles([b.name for b in manager.scenario_tree.bundles]),
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_OnBundles', results))
            print("Running InvocationType.OnBundlesChained...")
            results = manager.invoke_function(
                "_PerBundleChained",
                module_name=thisfile,
                function_args=(None,),
                invocation_type=InvocationType.OnBundlesChained(_bundle_names),
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_OnBundlesChained', results))
            print("Running InvocationType.OnBundlesChained (no args)...")
            results = manager.invoke_function(
                "_PerBundleChained_noargs",
                module_name=thisfile,
                invocation_type=InvocationType.OnBundlesChained(_bundle_names),
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data.append(('_OnBundlesChained_noargs', results))
        else:
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_PerBundle",
                    module_name=thisfile,
                    invocation_type=InvocationType.PerBundle,
                    oneway_call=oneway_call,
                    async_call=async_call)
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_PerBundleChained",
                    module_name=thisfile,
                    function_args=(None,),
                    invocation_type=InvocationType.PerBundleChained,
                    oneway_call=oneway_call,
                    async_call=async_call)
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_PerBundle",
                    module_name=thisfile,
                    invocation_type=InvocationType.OnBundle('Bundle1'),
                    oneway_call=oneway_call,
                    async_call=async_call)
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_PerBundle",
                    module_name=thisfile,
                    invocation_type=InvocationType.OnBundles(['B1','B2']),
                    oneway_call=oneway_call,
                    async_call=async_call)
            with self.assertRaises(ValueError):
                manager.invoke_function(
                    "_PerBundleChained",
                    module_name=thisfile,
                    function_args=(None,),
                    invocation_type=InvocationType.OnBundlesChained(['B1','B2']),
                    oneway_call=oneway_call,
                    async_call=async_call)

        for name, results in data:
            if delay:
                if async_call:
                    results = results.complete()
            if oneway_call:
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

            # bad oneway_call and async_call combo
            with self.assertRaises(ValueError):
                manager.invoke_function_on_worker(
                    worker_name,
                    "_Single",
                    module_name=thisfile,
                    invocation_type=InvocationType.Single,
                    oneway_call=True,
                    async_call=True)

            if dill_available or \
               isinstance(manager, ScenarioTreeManagerClientPyro):
                print("")
                print("Running InvocationType.Single... (using dill)")
                results = manager.invoke_function_on_worker(
                    worker_name,
                    _Single,
                    invocation_type=InvocationType.Single,
                    oneway_call=oneway_call,
                    async_call=async_call)
                if isinstance(manager, ScenarioTreeManagerClientPyro):
                    # module name must be None
                    with self.assertRaises(ValueError):
                        manager.invoke_function_on_worker(
                            worker_name,
                            "_Single",
                            module_name=None,
                            invocation_type=InvocationType.Single,
                            oneway_call=oneway_call,
                            async_call=async_call)

            else:
                if isinstance(manager, ScenarioTreeManagerClientPyro):
                    # requires dill
                    with self.assertRaises(ValueError):
                        manager.invoke_function_on_worker(
                            worker_name,
                            _Single,
                            invocation_type=InvocationType.Single,
                            oneway_call=oneway_call,
                            async_call=async_call)
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_Single",
                    module_name=thisfile,
                    invocation_type=InvocationType.Single,
                    oneway_call=oneway_call,
                    async_call=async_call)

            if not delay:
                if async_call:
                    results = results.complete()
            data['_Single'][worker_name] = results

        print("Running InvocationType.PerScenario on individual workers...")
        data['_PerScenario'] = {}
        for worker_name in manager.worker_names:
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenario",
                module_name=thisfile,
                invocation_type=InvocationType.PerScenario,
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data['_PerScenario'][worker_name] = results
        print("Running InvocationType.PerScenarioChained on individual workers...")
        data['_PerScenarioChained'] = {}
        results = (None,)
        for worker_name in manager.worker_names:
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenarioChained",
                module_name=thisfile,
                function_args=results,
                invocation_type=InvocationType.PerScenarioChained,
                async_call=async_call)
            if async_call:
                results = results.complete()
        data['_PerScenarioChained'] = results
        print("Running InvocationType.PerScenarioChained (no args) on individual workers...")
        data['_PerScenarioChained_noargs'] = {}
        for worker_name in manager.worker_names:
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenarioChained_noargs",
                module_name=thisfile,
                invocation_type=InvocationType.PerScenarioChained,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data['_PerScenarioChained_noargs'][worker_name] = results

        print("Running InvocationType.OnScenario on individual workers...")
        data['_OnScenario'] = {}
        for worker_name in manager.worker_names:
            assert len(manager.get_scenarios_for_worker(worker_name)) > 0
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenario",
                module_name=thisfile,
                invocation_type=InvocationType.OnScenario(manager.get_scenarios_for_worker(worker_name)[0]),
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data['_OnScenario'][worker_name] = results
        print("Running InvocationType.OnScenarios on individual workers...")
        data['_OnScenarios'] = {}
        for worker_name in manager.worker_names:
            assert len(manager.get_scenarios_for_worker(worker_name)) > 0
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenario",
                module_name=thisfile,
                invocation_type=InvocationType.OnScenarios(manager.get_scenarios_for_worker(worker_name)),
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
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
                module_name=thisfile,
                function_args=results,
                invocation_type=InvocationType.OnScenariosChained(manager.get_scenarios_for_worker(worker_name)),
                oneway_call=oneway_call,
                async_call=async_call)
            if async_call:
                results = results.complete()
        data['_OnScenariosChained'] = results
        print("Running InvocationType.OnScenariosChained (no args) on individual workers...")
        data['_OnScenariosChained_noargs'] = {}
        for worker_name in manager.worker_names:
            assert len(manager.get_scenarios_for_worker(worker_name)) > 0
            results = manager.invoke_function_on_worker(
                worker_name,
                "_PerScenarioChained_noargs",
                module_name=thisfile,
                invocation_type=InvocationType.OnScenariosChained(manager.get_scenarios_for_worker(worker_name)),
                oneway_call=oneway_call,
                async_call=async_call)
            if not delay:
                if async_call:
                    results = results.complete()
            data['_OnScenariosChained_noargs'][worker_name] = results

        if manager.scenario_tree.contains_bundles():
            print("Running InvocationType.PerBundle on individual workers...")
            data['_PerBundle'] = {}
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundle",
                    module_name=thisfile,
                    invocation_type=InvocationType.PerBundle,
                    oneway_call=oneway_call,
                    async_call=async_call)
                if not delay:
                    if async_call:
                        results = results.complete()
                data['_PerBundle'][worker_name] = results
            print("Running InvocationType.PerBundleChained on individual workers...")
            data['_PerBundleChained'] = {}
            results = (None,)
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained",
                    module_name=thisfile,
                    function_args=results,
                    invocation_type=InvocationType.PerBundleChained,
                    async_call=async_call)
                if async_call:
                    results = results.complete()
            data['_PerBundleChained'] = results
            print("Running InvocationType.PerBundleChained (no args) on individual workers...")
            data['_PerBundleChained_noargs'] = {}
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained_noargs",
                    module_name=thisfile,
                    invocation_type=InvocationType.PerBundleChained,
                    async_call=async_call)
                if not delay:
                    if async_call:
                        results = results.complete()
                data['_PerBundleChained_noargs'][worker_name] = results

            print("Running InvocationType.OnBundle on individual workers...")
            data['_OnBundle'] = {}
            for worker_name in manager.worker_names:
                assert len(manager.get_scenarios_for_worker(worker_name)) > 0
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundle",
                    module_name=thisfile,
                    invocation_type=InvocationType.OnBundle(manager.get_bundles_for_worker(worker_name)[0]),
                    oneway_call=oneway_call,
                    async_call=async_call)
                if not delay:
                    if async_call:
                        results = results.complete()
                data['_OnBundle'][worker_name] = results
            print("Running InvocationType.OnBundles on individual workers...")
            data['_OnBundles'] = {}
            for worker_name in manager.worker_names:
                assert len(manager.get_scenarios_for_worker(worker_name)) > 0
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundle",
                    module_name=thisfile,
                    invocation_type=InvocationType.OnBundles(manager.get_bundles_for_worker(worker_name)),
                    oneway_call=oneway_call,
                    async_call=async_call)
                if not delay:
                    if async_call:
                        results = results.complete()
                data['_OnBundles'][worker_name] = results
            print("Running InvocationType.OnBundlesChained on individual workers...")
            data['_OnBundlesChained'] = {}
            results = (None,)
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained",
                    module_name=thisfile,
                    function_args=results,
                    invocation_type=InvocationType.OnBundlesChained(manager.get_bundles_for_worker(worker_name)),
                    async_call=async_call)
                if async_call:
                    results = results.complete()
            data['_OnBundlesChained'] = results
            print("Running InvocationType.OnBundlesChained (no args) on individual workers...")
            data['_OnBundlesChained_noargs'] = {}
            for worker_name in manager.worker_names:
                results = manager.invoke_function_on_worker(
                    worker_name,
                    "_PerBundleChained_noargs",
                    module_name=thisfile,
                    invocation_type=InvocationType.OnBundlesChained(manager.get_bundles_for_worker(worker_name)),
                    async_call=async_call)
                if not delay:
                    if async_call:
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
                    if async_call:
                        for worker_name in results:
                            results[worker_name] = results[worker_name].complete()
            if oneway_call:
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

        # bad oneway_call and async_call combo
        with self.assertRaises(ValueError):
            manager.invoke_method(
                "junk",
                method_args=(None,),
                method_kwds={'a': None},
                oneway_call=True,
                async_call=True)
        # bad paused state
        if isinstance(manager, ScenarioTreeManagerClientPyro) and \
           (not async_call) and (not oneway_call) and (not delay):
            self.assertEqual(manager._transmission_paused, False)
            manager.pause_transmit()
            self.assertEqual(manager._transmission_paused, True)
            with self.assertRaises(ValueError):
                manager.invoke_method(
                    "junk",
                    method_args=(None,),
                    method_kwds={'a': None},
                    oneway_call=oneway_call,
                    async_call=async_call)
            manager.unpause_transmit()
            self.assertEqual(manager._transmission_paused, False)

        result = manager.invoke_method(
            "junk",
            method_args=(None,),
            method_kwds={'a': None},
            oneway_call=oneway_call,
            async_call=async_call)
        if async_call:
            result = result.complete()
        if oneway_call:
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

        results = {}
        for worker_name in manager.worker_names:
            # bad oneway_call and async_call combo
            with self.assertRaises(ValueError):
                manager.invoke_method_on_worker(worker_name,
                                                "junk",
                                                method_args=(None,),
                                                method_kwds={'a': None},
                                                oneway_call=True,
                                                async_call=True)
            results[worker_name] = \
                manager.invoke_method_on_worker(worker_name,
                                                "junk",
                                                method_args=(None,),
                                                method_kwds={'a': None},
                                                oneway_call=oneway_call,
                                                async_call=async_call)
        if async_call:
            results = dict((worker_name, results[worker_name].complete())
                           for worker_name in results)

        if oneway_call:
            for worker_name in results:
                self.assertEqual(id(results[worker_name]), id(None))
        else:
            self.assertEqual(sorted(results.keys()),
                             sorted(manager.worker_names))
            for worker_name in results:
                self.assertEqual(results[worker_name], ((None,), {'a': None}))

    @unittest.nottest
    def _scenarios_test(self,
                        async_call=False,
                        oneway_call=False,
                        delay=False):
        self._setup(self.options)
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async_call=async_call,
                                     oneway_call=oneway_call,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles1_test(self,
                       async_call=False,
                       oneway_call=False,
                       delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict1
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async_call=async_call,
                                     oneway_call=oneway_call,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles2_test(self,
                       async_call=False,
                       oneway_call=False,
                       delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict2
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async_call=async_call,
                                     oneway_call=oneway_call,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles3_test(self,
                       async_call=False,
                       oneway_call=False,
                       delay=False):
        options = PySPConfigBlock()
        self._setup(self.options)
        self.options.scenario_bundle_specification = self._bundle_dict3
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async_call=async_call,
                                     oneway_call=oneway_call,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    def test_scenarios(self):
        self._scenarios_test(async_call=False,
                             oneway_call=False,
                             delay=False)
    def test_scenarios_async_call(self):
        self._scenarios_test(async_call=True,
                             oneway_call=False,
                             delay=False)
    def test_scenarios_async_call_delay(self):
        self._scenarios_test(async_call=True,
                             oneway_call=False,
                             delay=True)

    def test_bundles1(self):
        self._bundles1_test(async_call=False,
                            oneway_call=False,
                            delay=False)
    def test_bundles1_async_call(self):
        self._bundles1_test(async_call=True,
                            oneway_call=False,
                            delay=False)
    def test_bundles1_async_call_delay(self):
        self._bundles1_test(async_call=True,
                            oneway_call=False,
                            delay=True)

    def test_bundles2(self):
        self._bundles2_test(async_call=False,
                            oneway_call=False,
                            delay=False)
    def test_bundles2_async_call(self):
        self._bundles2_test(async_call=True,
                            oneway_call=False,
                            delay=False)
    def test_bundles2_async_call_delay(self):
        self._bundles2_test(async_call=True,
                            oneway_call=False,
                            delay=True)

    def test_bundles3(self):
        self._bundles3_test(async_call=False,
                            oneway_call=False,
                            delay=False)
    def test_bundles3_async_call(self):
        self._bundles3_test(async_call=True,
                            oneway_call=False,
                            delay=False)
    def test_bundles3_async_call_delay(self):
        self._bundles3_test(async_call=True,
                            oneway_call=False,
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

class TestScenarioTreeManagerClientSerial(
        unittest.TestCase,
        _ScenarioTreeManagerTesterBase):

    cls = _ScenarioTreeManagerClientTestSerial

    def setUp(self):
        self.options = PySPConfigBlock()
        self.async_call = False
        self.oneway_call = False
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
    for i, proc in enumerate(_taskworker_processes):
        _kill(proc)
        outname = os.path.join(thisdir,
                               "TestCapture_scenariotreeserver_" + \
                               str(i+1) + ".out")
        if os.path.exists(outname):
            try:
                os.remove(outname)
            except OSError:
                pass
    _taskworker_processes = []
    if os.path.exists(os.path.join(thisdir, "Pyro_NS_URI")):
        try:
            os.remove(os.path.join(thisdir, "Pyro_NS_URI"))
        except OSError:
            pass

def _setUpPyro():
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
            outname = os.path.join(thisdir,
                                   "TestCapture_scenariotreeserver_" + \
                                   str(i+1) + ".out")
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

class _ScenarioTreeManagerClientPyroTesterBase(_ScenarioTreeManagerTesterBase):

    cls = ScenarioTreeManagerClientPyro

    def setUp(self):
        _setUpPyro()
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
                                async_call=False,
                                oneway_call=False,
                                delay=False):
        self._setup(self.options, servers=1)
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async_call=async_call,
                                     oneway_call=oneway_call,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), False)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles1_1server_test(self,
                               async_call=False,
                               oneway_call=False,
                               delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict1
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager,
                                     async_call=async_call,
                                     oneway_call=oneway_call,
                                     delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles2_1server_test(self,
                               async_call=False,
                               oneway_call=False,
                               delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict2
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async_call=async_call, oneway_call=oneway_call, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    @unittest.nottest
    def _bundles3_1server_test(self,
                               async_call=False,
                               oneway_call=False,
                               delay=False):
        self._setup(self.options, servers=1)
        self.options.scenario_bundle_specification = self._bundle_dict3
        with self.cls(self.options, **_init_kwds) as manager:
            self._run_function_tests(manager, async_call=async_call, oneway_call=oneway_call, delay=delay)
            self.assertEqual(manager._scenario_tree.contains_bundles(), True)
        self.assertEqual(list(self.options.unused_user_values()), [])

    def test_scenarios_1server(self):
        self._scenarios_1server_test(async_call=False,
                                     oneway_call=False,
                                     delay=False)
    def test_scenarios_1server_async_call(self):
        self._scenarios_1server_test(async_call=True,
                                     oneway_call=False,
                                     delay=False)
    def test_scenarios_1server_async_call_delay(self):
        self._scenarios_1server_test(async_call=True,
                                     oneway_call=False,
                                     delay=True)

    def test_bundles1_1server(self):
        self._bundles1_1server_test(async_call=False,
                                    oneway_call=False,
                                    delay=False)
    def test_bundles1_1server_async_call(self):
        self._bundles1_1server_test(async_call=True,
                                    oneway_call=False,
                                    delay=False)
    def test_bundles1_1server_async_call_delay(self):
        self._bundles1_1server_test(async_call=True,
                                    oneway_call=False,
                                    delay=True)

    def test_bundles2_1server(self):
        self._bundles2_1server_test(async_call=False,
                                    oneway_call=False,
                                    delay=False)
    def test_bundles2_1server_async_call(self):
        self._bundles2_1server_test(async_call=True,
                                    oneway_call=False,
                                    delay=False)
    def test_bundles2_1server_async_call_delay(self):
        self._bundles2_1server_test(async_call=True,
                                    oneway_call=False,
                                    delay=True)

    def test_bundles3_1server(self):
        self._bundles3_1server_test(async_call=False,
                                    oneway_call=False,
                                    delay=False)
    def test_bundles3_1server_async_call(self):
        self._bundles3_1server_test(async_call=True,
                                    oneway_call=False,
                                    delay=False)
    def test_bundles3_1server_async_call_delay(self):
        self._bundles3_1server_test(async_call=True,
                                    oneway_call=False,
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
class TestScenarioTreeManagerClientPyro(
        unittest.TestCase,
        _ScenarioTreeManagerClientPyroTesterBase):

    def setUp(self):
        _ScenarioTreeManagerClientPyroTesterBase.setUp(self)
    def _setup(self, options, servers=None):
        _ScenarioTreeManagerClientPyroTesterBase._setup(self, options, servers=servers)
        options.pyro_handshake_at_startup = False

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

if __name__ == "__main__":
    unittest.main()
