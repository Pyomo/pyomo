#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeManagerSerial",
           "ScenarioTreeManagerSPPyro")

import os
import gc
import sys
import time
import contextlib
import copy

try:
    import pstats
    pstats_available=True
except ImportError:
    pstats_available=False
# for profiling
try:
    import cProfile as profile
except ImportError:
    import profile
try:
    from guppy import hpy
    guppy_available = True
except ImportError:
    guppy_available = False

from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyutilib.pyro import shutdown_pyro_components
from pyutilib.services import TempfileManager

from pyomo.util import pyomo_command
from pyomo.core.base import maximize, minimize
from pyomo.opt.parallel import SolverManagerFactory

from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.scenariotree import ScenarioTreeInstanceFactory
import pyomo.pysp.scenariotree.scenariotreeserverutils

class _ScenarioTreeManagerBase(object):

    @staticmethod
    def register_options(options):
        #
        # scenario instance construction
        #
        safe_register_common_option(options,
                                    "model_location")
        safe_register_common_option(options,
                                    "scenario_tree_location")
        safe_register_common_option(options,
                                    "objective_sense_stage_based")
        safe_register_common_option(options,
                                    "boundsetter_callback")
        safe_register_common_option(options,
                                    "aggregategetter_callback")
        #
        # scenario tree generation
        #
        safe_register_common_option(options,
                                    "scenario_tree_random_seed")
        safe_register_common_option(options,
                                    "scenario_tree_downsample_fraction")
        safe_register_common_option(options,
                                    "scenario_bundle_specification")
        safe_register_common_option(options,
                                    "create_random_bundles")
        safe_register_common_option(options,
                                    "scenario_tree_manager")
        #
        # various
        #
        safe_register_common_option(options,
                                    "output_times")
        safe_register_common_option(options,
                                    "verbose")
        safe_register_common_option(options,
                                    "profile_memory")

    def __init__(self, options):

        init_start_time = time.time()

        self._options = options
        self._scenario_instance_factory = None
        self._scenario_tree = None
        self._objective_sense = None
        # bundle info
        self._bundle_scenario_instance_map = {}
        # For the users to modify as they please in the aggregate
        # callback as long as the data placed on it can be serialized
        # by Pyro
        self._aggregate_user_data = {}
        # callback info
        self._callback_function = {}
        self._callback_mapped_module_name = {}
        self._aggregate_getter = None
        self._aggregate_getter_name = None
        self._bound_setter = None
        self._bound_setter_name = None

        self._import_callbacks()
        self._generate_scenario_tree()
        assert self._scenario_tree is not None

        #
        # derived methods
        #

        self._init()

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

        if self._options.output_times:
            print("Overall initialization time=%.2f seconds"
                  % (time.time() - init_start_time))

        # gather and report memory statistics (for leak
        # detection purposes) if specified.
        if self._options.profile_memory:
            if guppy_available:
                print(hpy().heap())
            else:
                print("Guppy module is unavailable for "
                      "memory profiling")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._scenario_tree = None
        self._instances = None
        self._sppyro_worker_jobs_map = {}
        self._sppyro_job_worker_map = {}
        self._initialized = False

    def _import_callbacks(self):

        renamed = {}
        renamed["pysp_aggregategetter_callback"] = \
            "ph_aggregategetter_callback"
        renamed["pysp_phrhosetter_callback"] = \
            "ph_rhosetter_callback"
        renamed["pysp_boundsetter_callback"] = \
            "ph_boundsetter_callback"
        for module_name, attr_name, callback_name in (
                (self._options.aggregategetter_callback,
                 "_aggregate_getter",
                 "pysp_aggregategetter_callback"),
                (self._options.boundsetter_callback,
                 "_bound_setter",
                 "pysp_boundsetter_callback")):

            assert callback_name in renamed.keys()
            deprecated_callback_name = renamed[callback_name]
            if module_name is not None:
                sys_modules_key, module = \
                    load_external_module(module_name)
                callback = None
                for oname, obj in inspect.getmembers(module):
                    if oname == callback_name:
                        callback = obj
                        break
                if callback is None:
                    for oname, obj in inspect.getmembers(module):
                        if oname == deprecated_callback_name:
                            callback = obj
                            break
                    if callback is None:
                        raise ImportError(
                            "PySP callback with name '%s' could "
                            "not be found in module file: %s"
                            % (deprecated_callback_name, module_name))
                    if callback is None:
                        raise ImportError(
                            "PySP callback with name '%s' could "
                            "not be found in module file: %s"
                            % (callback_name, module_name))
                    else:
                        logger.warning(
                            "DEPRECATED: Callback with name '%s' "
                            "has been renamed '%s'"
                            % (deprecated_callback_name,
                               callback_name))
                        callback_name = deprecated_callback_name

                self._callback_function[sys_modules_key] = callback
                setattr(self, attr_name, sys_modules_key)
                setattr(self, attr_name+"_name", callback_name)
                self._callback_mapped_module_name\
                    [sys_modules_key] = module_name

    def _generate_scenario_tree(self):

        start_time = time.time()
        if self._options.verbose:
            print("Importing model and scenario tree files")

        scenario_instance_factory = \
            ScenarioTreeInstanceFactory(
                self._options.model_location,
                self._options.scenario_tree_location,
                self._options.verbose)

        if self._options.verbose or self._options.output_times:
            print("Time to import model and scenario tree "
                  "structure files=%.2f seconds"
                  %(time.time() - start_time))

        try:

            self._scenario_tree = \
                scenario_instance_factory.\
                generate_scenario_tree(
                    downsample_fraction=self._options.scenario_tree_downsample_fraction,
                    bundles_file=self._options.scenario_bundle_specification,
                    random_bundles=self._options.create_random_bundles,
                    random_seed=self._options.scenario_tree_random_seed)

            # print the input tree for validation/information
            # purposes.
            if self._options.verbose:
                self._scenario_tree.pprint()

            # validate the tree prior to doing anything serious
            if not self._scenario_tree.validate():
                raise RuntimeError("Scenario tree is invalid")
            else:
                if self._options.verbose:
                    print("Scenario tree is valid!")

        except:
            print("Failed to generate scenario tree")
            raise
        finally:
            scenario_instance_factory.close()

    #
    # Methods defined by derived class
    #

    def _init(self):
        raise NotImplementedError("This method is abstract")

class ScenarioTreeManagerSerial(_ScenarioTreeManagerBase):

    @staticmethod
    def register_options(options):
        _ScenarioTreeManagerBase.register_options(options)

    def __init__(self, options):
        # scenario instance models
        self._instances = None
        # bundle instance models
        self._bundle_binding_instance_map = {}

        # this calls _init()
        super(ScenarioTreeManagerSerial, self).\
            __init__(options)

    def close(self):
        super(ScenarioTreeManagerSerial, self).close()
        self._instance = None
        self._bundle_binding_instance_map = {}

    #
    # Abstract methods
    #

    def _init(self):
        assert self._scenario_tree is not None
        if self._options.verbose:
            print("Initializing ScenarioTreeManagerSerial")
            print("")

        #
        # Build scenario instances
        #

        build_start_time = time.time()

        if self._options.verbose:
            print("Constructing scenario tree instances")

        self._instances = \
            self._scenario_tree._scenario_instance_factory.\
            construct_instances_for_scenario_tree(
                self._scenario_tree,
                report_timing=self._options.output_times)

        if self._options.verbose or \
           self._options.output_times:
            print("Time to construct scenario instances="
                  "%.2f seconds"
                  % (time.time() - build_start_time))

        if self._options.verbose:
            print("Linking instances into scenario tree")

        build_start_time = time.time()

        # with the scenario instances now available, link the
        # referenced objects directly into the scenario tree.
        self._scenario_tree.linkInInstances(
            self._instances,
            objective_sense=self._options.objective_sense_stage_based,
            create_variable_ids=True)

        if self._options.verbose or self._options.output_times:
            print("Time link scenario tree with instances="
                  "%.2f seconds"
                  % (time.time() - build_start_time))

        if self._scenario_tree.contains_bundles():
            build_start_time = time.time()

            if self._options.verbose:
                print("Forming binding instances for "
                      "all scenario bundles")

            self._bundle_binding_instance_map.clear()
            self._bundle_scenario_instance_map.clear()

            if not self._scenario_tree.contains_bundles():
                raise RuntimeError(
                    "Failed to create binding instances for "
                    "scenario bundles - no bundles are defined!")

            for scenario_bundle in self._scenario_tree.\
                _scenario_bundles:

                if self._options.verbose:
                    print("Creating binding instance for "
                          "scenario bundle=%s"
                          % (scenario_bundle._name))

                self._bundle_scenario_instance_map\
                    [scenario_bundle._name] = {}

                for scenario_name in scenario_bundle._scenario_names:
                    self._bundle_scenario_instance_map\
                        [scenario_bundle._name][scenario_name] = \
                            self._instances[scenario_name]

                # IMPORTANT: The bundle variable IDs must be idential
                #            to those in the parent scenario tree -
                #            this is critical for storing results,
                #            which occurs at the full-scale scenario
                #            tree.

                scenario_bundle._scenario_tree.linkInInstances(
                    self._instances,
                    create_variable_ids=False,
                    master_scenario_tree=self._scenario_tree,
                    initialize_solution_data=False)

                bundle_ef_instance = create_ef_instance(
                    scenario_bundle._scenario_tree,
                    ef_instance_name=scenario_bundle._name,
                    verbose_output=self._verbose)

                self._bundle_binding_instance_map\
                    [scenario_bundle._name] = \
                        bundle_ef_instance

            if self._output_times:
                print("Scenario bundle construction time="
                      "%.2f seconds"
                      % (time.time() - build_start_time))

        if self._options.verbose:
            print("ScenarioTreeManagerSerial is successfully "
                  "initialized")

        if self._options.aggregategetter_callback is not None:
            # Run the user script to collect aggregate scenario data
            if self._options.verbose:
                print("Executing user aggregate getter "
                      "callback function")
            for scenario in self._scenario_tree._scenarios:
                result = self._callback_function[self._aggregate_getter](
                    self,
                    self._scenario_tree,
                    scenario,
                    self._aggregate_user_data)
                assert len(result) == 1
                self._aggregate_user_data = result[0]

        if self._options.boundsetter_callback is not None:
            # run the user script to initialize variable bounds
            if self._options.verbose:
                print("Executing user bound setter "
                      "callback function")
            for scenario in self._scenario_tree._scenarios:
                self._callback_function[self._bound_setter](
                    self,
                    self._scenario_tree,
                    scenario)

class ScenarioTreeManagerSPPyro(_ScenarioTreeManagerBase):

    @staticmethod
    def register_options(options):
        _ScenarioTreeManagerBase.register_options(options)
        safe_register_common_option(options,
                                    "pyro_hostname")
        safe_register_common_option(options,
                                    "handshake_with_sppyro")
        safe_register_common_option(options,
                                    "sppyro_required_workers")
        safe_register_common_option(options,
                                    "sppyro_find_workers_timeout")
        safe_register_common_option(options,
                                    "shutdown_pyro")

    def __init__(self, options):
        # distributed worker information
        self._sppyro_worker_jobs_map = {}
        self._sppyro_job_worker_map = {}
        # for now this is a SolverManager but we don't treat
        # it like one
        self._solver_manager = None

        # this calls _init()
        super(ScenarioTreeManagerSPPyro, self).\
            __init__(options)

    def close(self):
        super(ScenarioTreeManagerSPPyro, self).close()
        pyomo.pysp.scenariotree.scenariotreeserverutils.\
            release_scenariotreeservers(self)
        if self._solver_manager is not None:
            self._solver_manager.release_workers()
            self._solver_manager.deactivate()

    #
    # Abstract methods
    #

    def _init(self):
        assert self._scenario_tree is not None
        if self._options.verbose:
            print("Initializing ScenarioTreeManagerSPPyro")
            print("")

        # construct the solver manager.
        self._solver_manager = SolverManagerFactory(
            'phpyro',
            host=self._options.pyro_hostname)

        assert self._solver_manager is not None

        if self._scenario_tree.contains_bundles():
            num_jobs = len(self._scenario_tree._scenario_bundles)
            if self._options.verbose:
                print("Bundle solver jobs available: %s"
                      % (str(num_jobs)))
        else:
            num_jobs = len(self._scenario_tree._scenarios)
            if self._options.verbose:
                print("Scenario solver jobs available: %s"
                      % (str(num_jobs)))

        if self._options.verbose:
            print("Acquiring SPPyro scenario tree servers")

        workers_expected = self._options.sppyro_required_workers
        if (workers_expected is None):
            workers_expected = num_jobs

        timeout = self._options.sppyro_find_workers_timeout if \
                  (self._options.sppyro_required_workers is None) else \
                  None

        self._solver_manager.acquire_workers(workers_expected,
                                             timeout)

        if self._options.verbose:
            print("Scenario tree servers acquired")

        initialization_action_handles = []

        if self._options.verbose:
            print("Broadcasting requests to initialize "
                  "distributed scenario tree workers")

        initialization_action_handles.extend(
            pyomo.pysp.scenariotree.scenariotreeserverutils.\
            initialize_scenariotree_workers(self))

        if self._options.verbose:
            print("Distributed scenario tree initialization "
                  "requests successfully transmitted")

        # run the user script to collect aggregate scenario data. This
        # can slow down initialization as syncronization across all
        # scenario tree servers is required following serial
        # executation
        if self._options.aggregategetter_callback is not None:
            if self._options.verbose:
                print("Executing user aggregate getter "
                      "callback function on scenario tree servers")

            callback_name = "pysp_aggregategetter_callback"
            # Transmit invocation to scenario tree servers
            if self._scenario_tree.contains_bundles():
                for scenario_bundle in self._scenario_tree._scenario_bundles:
                    ah = pyomo.pysp.scenariotree.scenariotreeserverutils.\
                         transmit_external_function_invocation_to_worker(
                             self,
                             scenario_bundle._name,
                             self._mapped_module_name[self._aggregate_getter],
                             self._aggregate_getter_name,
                             invocation_type=(pyomo.pysp.scenariotree.\
                                              scenariotreeserverutils.InvocationType.\
                                              PerScenarioChainedInvocation),
                             return_action_handle=True,
                             function_args=(self._aggregate_user_data,))
                    while(1):
                        action_handle = self._solver_manager.wait_any()
                        if action_handle in initialization_action_handles:
                            initialization_action_handles.remove(action_handle)
                            self._solver_manager.get_results(action_handle)
                        elif action_handle == ah:
                            result = self._solver_manager.get_results(action_handle)
                            break
                    assert len(result) == 1
                    self._aggregate_user_data = result[0]

            else:
                for scenario in self._scenario_tree._scenarios:
                    ah = pyomo.pysp.scenariotree.scenariotreeserverutils.\
                         transmit_external_function_invocation_to_worker(
                             self,
                             scenario._name,
                             self._mapped_module_name[self._aggregate_getter],
                             self._aggregate_getter_name,
                             invocation_type=(pyomo.pysp.scenariotree.\
                                              scenariotreeserverutils.InvocationType.\
                                              SingleInvocation),
                             return_action_handle=True,
                             function_args=(self._aggregate_user_data,))
                    while(1):
                        action_handle = self._solver_manager.wait_any()
                        if action_handle in initialization_action_handles:
                            initialization_action_handles.remove(action_handle)
                            self._solver_manager.get_results(action_handle)
                        elif action_handle == ah:
                            result = self._solver_manager.get_results(action_handle)
                            break
                    assert len(result) == 1
                    self._aggregate_user_data = result[0]

            # Transmit aggregate state to scenario tree servers
            if self._options.verbose:
                print("Broadcasting final aggregate data "
                      "to scenario tree servers")
            initialization_action_handles.extend(
                pyomo.pysp.scenariotree.\
                scenariotreeserverutils.transmit_external_function_invocation(
                    self,
                    "pyomo.pysp.scenariotree.scenariotreemanager",
                    "assign_aggregate_data",
                    invocation_type=(pyomo.pysp.scenariotree.\
                                     scenariotreeserverutils.InvocationType.\
                                     SingleInvocation),
                    return_action_handles=True,
                    function_args=(self._aggregate_user_data,)))

        # run the user script to initialize variable bounds
        if self._options.boundsetter_callback is not None:

            if self._options.verbose:
                print("Executing user bound setter "
                      "callback function on scenario tree servers")

            # Transmit invocation to scenario tree servers
            if self._scenario_tree.contains_bundles():
                for scenario_bundle in self._scenario_tree._scenario_bundles:
                    initialization_action_handles.append(
                        pyomo.pysp.scenariotree.scenariotreeserverutils.\
                        transmit_external_function_invocation_to_worker(
                            self,
                            scenario_bundle._name,
                            self._mapped_module_name[self._bound_setter],
                            self._bound_setter_name,
                            invocation_type=(pyomo.pysp.scenariotree.\
                                             scenariotreeserverutils.InvocationType.\
                                             PerScenarioInvocation),
                            return_action_handle=True))
            else:
                for scenario in self._scenario_tree._scenarios:
                    initialization_action_handles.append(
                        pyomo.pysp.scenariotree.scenariotreeserverutils.\
                        transmit_external_function_invocation_to_worker(
                            self,
                            scenario._name,
                            self._mapped_module_name[self._bound_setter],
                            self._bound_setter_name,
                            invocation_type=(pyomo.pysp.scenariotree.\
                                             scenariotreeserverutils.InvocationType.\
                                             SingleInvocation),
                            return_action_handle=True))

        #
        # gather scenario tree data
        #

        if self._options.verbose:
            print("Broadcasting requests to collect scenario tree "
                  "instance data from scenario tree servers")

        pyomo.pysp.scenariotree.scenariotreeserverutils.\
            gather_scenario_tree_data(self,
                                      initialization_action_handles)
        assert len(initialization_action_handles) == 0

        if self._options.verbose:
            print("Scenario tree instance data successfully "
                  "collected")

        if self._options.verbose:
            print("Broadcasting scenario tree id mapping"
                  "to scenario tree servers")

        pyomo.pysp.scenariotree.\
            scenariotreeserverutils.transmit_scenario_tree_ids(self)

        if self._options.verbose:
            print("Scenario tree ids successfully sent")

        if self._options.verbose:
            print("ScenarioTreeManagerSPPyro is successfully "
                  "initialized")
