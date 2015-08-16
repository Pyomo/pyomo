#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeManagerSerial",
           "ScenarioTreeManagerSPPyroBasic",
           "ScenarioTreeManagerSPPyro")

# TODO: options ordering during help output

import sys
import time
import itertools
import inspect
import logging
import traceback

from pyutilib.pyro import shutdown_pyro_components
from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.util.misc import load_external_module
from pyomo.pysp.scenariotree import ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.sppyro_action_manager \
    import SPPyroAsyncActionManager
from pyomo.pysp.scenariotree.scenariotreeserver \
    import SPPyroScenarioTreeServer
from pyomo.pysp.scenariotree.scenariotreeserverutils \
    import (WorkerInitType,
            InvocationType)
from pyomo.pysp.ef import create_ef_instance

from six import iteritems

try:
    from guppy import hpy
    guppy_available = True
except ImportError:
    guppy_available = False

logger = logging.getLogger('pyomo.pysp')

#
# Code that is common to the serial manager object
# and the sppyro worker object, but not the sppyro
# manager object
#

class _ScenarioTreeWorkerImpl(PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the _ScenarioTreeWorkerImpl class")

    #
    # various
    #
    safe_register_common_option(_registered_options,
                                "output_times")
    safe_register_common_option(_registered_options,
                                "verbose")

    def __init__(self, *args, **kwds):
        super(_ScenarioTreeWorkerImpl, self).__init__(*args, **kwds)

        # scenario instance models
        self._instances = None
        # bundle instance models
        self._bundle_binding_instance_map = {}

    #
    # Creates the binding instance for the bundle and
    # stores it in _bundle_extensive_form_map
    #


    def add_bundle(self, bundle_name, scenario_list):

        if self._scenario_tree.contains_bundle(bundle_name):
            raise ValueError(
                "Unable to create bundle with name %s. A bundle "
                "with that name already exists on the scenario tree"
                % (bundle_name))

        for scenario_name in scenario_list:

            if scenario_name in self._scenario_to_bundle_map:
                raise ValueError(
                    "Unable to form binding instance for bundle %s. "
                    "Scenario %s already belongs to bundle %s."
                    % (bundle_name,
                       scenario_name,
                       self._scenario_to_bundle_map[scenario_name]))

            self._scenario_to_bundle_map[scenario_name] = bundle_name

        self._scenario_tree.add_bundle(bundle_name, scenario_list)

        self._form_bundle_binding_instance(bundle_name)

    def _form_bundle_binding_instance(self, bundle_name):

        if self._options.verbose:
            print("Forming binding instance for scenario bundle %s"
                  % (bundle_name))

        start_time = time.time()

        if not self._scenario_tree.contains_bundle(bundle_name):
            raise RuntimeError(
                "Failed to create binding instances for scenario "
                "bundle - no scenario bundle with name %s exists."
                % (bundle_name))

        assert bundle_name not in self._bundle_binding_instance_map

        bundle = self._scenario_tree.get_bundle(bundle_name)

        for scenario_name in bundle._scenario_names:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            assert scenario_name in self._scenario_to_bundle_map
            assert self._scenario_to_bundle_map[scenario_name] == bundle_name
            assert scenario._instance is not None
            assert scenario._instance is self._instances[scenario_name]
            assert scenario._instance.parent_block() is None
            self._scenario_to_bundle_map[scenario_name] = bundle_name

        # IMPORTANT: The bundle variable IDs must be idential to
        #            those in the parent scenario tree - this is
        #            critical for storing results, which occurs at
        #            the full-scale scenario tree.

        bundle._scenario_tree.linkInInstances(
            self._instances,
            create_variable_ids=False,
            master_scenario_tree=self._scenario_tree,
            initialize_solution_data=False)

        bundle_ef_instance = create_ef_instance(
            bundle._scenario_tree,
            ef_instance_name=bundle._name,
            verbose_output=self._options.verbose)

        self._bundle_binding_instance_map[bundle._name] = \
            bundle_ef_instance

        end_time = time.time()
        if self._options.output_times:
            print("Time construct binding instance for scenario bundle "
                  "%s=%.2f seconds" % (bundle_name, end_time - start_time))

    def remove_bundle(self, bundle_name):

        if not self._scenario_tree.contains_bundle(bundle_name):
            raise ValueError(
                "Unable to remove bundle with name %s. A bundle "
                "with that name does not exist on the scenario tree"
                % (bundle_name))

        self._remove_bundle_impl(bundle_name)

        bundle = self._scenario_tree.get_bundle(bundle_name)
        for scenario_name in bundle._scenario_names:

            del self._scenario_to_bundle_map[scenario_name]

        self._scenario_tree.remove_bundle(bundle_name)

    def _remove_bundle_impl(self, bundle_name):

        if self._options.verbose:
            print("Destroying binding instance for scenario bundle %s"
                  % (bundle_name))

        if not self._scenario_tree.contains_bundle(bundle_name):
            raise RuntimeError(
                "Failed to destory binding instance for scenario "
                "bundle - no scenario bundle with name %s exists."
                % (bundle_name))

        assert bundle_name in self._bundle_binding_instance_map

        bundle_ef_instance = \
            self._bundle_binding_instance_map[bundle_name]

        bundle = self._scenario_tree.get_bundle(bundle_name)

        for scenario_name in bundle._scenario_names:
            scenario = self._scenario_tree.get_scenario(scenario_name)
            bundle_ef_instance.del_component(scenario._instance)
            scenario._instance_objective.activate()

        del self._bundle_binding_instance_map[bundle_name]

    #
    # Abstract methods for _ScenarioTreeManager
    #

    def _close_impl(self):
        # copy the list of bundle names as the next loop will modify
        # the scenario_tree._scenario_bundles list
        bundle_names = \
            [bundle._name for bundle in self._scenario_tree._scenario_bundles]
        for bundle_name in bundle_names:
            self.remove_bundle(bundle_name)
        assert not self._scenario_tree.contains_bundles()
        self._instances = None
        self._bundle_binding_instance_map = None

    #
    # Methods defined by derived class: None
    #

#
# Code that is common to the serial manager object
# and the sppyro manager object, but not the sppyro
# worker object
#

class _ScenarioTreeManagerImpl(PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the _ScenarioTreeManagerImpl class")

    #
    # scenario instance construction
    #
    safe_register_common_option(_registered_options,
                                "model_location")
    safe_register_common_option(_registered_options,
                                "scenario_tree_location")
    safe_register_common_option(_registered_options,
                                "objective_sense_stage_based")
    safe_register_common_option(_registered_options,
                                "postinit_callback_location")
    safe_register_common_option(_registered_options,
                                "aggregategetter_callback_location")

    #
    # scenario tree generation
    #
    safe_register_common_option(_registered_options,
                                "scenario_tree_random_seed")
    safe_register_common_option(_registered_options,
                                "scenario_tree_downsample_fraction")
    safe_register_common_option(_registered_options,
                                "scenario_bundle_specification")
    safe_register_common_option(_registered_options,
                                "create_random_bundles")

    #
    # various
    #
    safe_register_common_option(_registered_options,
                                "output_times")
    safe_register_common_option(_registered_options,
                                "verbose")
    safe_register_common_option(_registered_options,
                                "profile_memory")

    def __init__(self, *args, **kwds):
        super(_ScenarioTreeManagerImpl, self).__init__(*args, **kwds)

        # callback info
        self._callback_function = {}
        self._callback_mapped_module_name = {}
        self._aggregategetter_keys = []
        self._aggregategetter_names = []
        self._postinit_keys = []
        self._postinit_names = []

        self._generate_scenario_tree()
        self._import_callbacks()

    def _generate_scenario_tree(self):

        start_time = time.time()
        if self._options.verbose:
            print("Importing model and scenario tree files")

        scenario_instance_factory = \
            ScenarioTreeInstanceFactory(
                self._options.model_location,
                scenario_tree_location=self._options.scenario_tree_location,
                verbose=self._options.verbose)

        if self._options.verbose or self._options.output_times:
            print("Time to import model and scenario tree "
                  "structure files=%.2f seconds"
                  %(time.time() - start_time))

        try:

            self._scenario_tree = \
                scenario_instance_factory.\
                generate_scenario_tree(
                    downsample_fraction=\
                       self._options.scenario_tree_downsample_fraction,
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

    def _import_callbacks(self):

        renamed = {}
        renamed["pysp_aggregategetter_callback"] = \
            "ph_aggregategetter_callback"
        renamed["pysp_postinit_callback"] = \
            "ph_boundsetter_callback"
        for module_names, attr_name, callback_name in (
                (self._options.aggregategetter_callback_location,
                 "_aggregategetter",
                 "pysp_aggregategetter_callback"),
                (self._options.postinit_callback_location,
                 "_postinit",
                 "pysp_postinit_callback")):

            assert callback_name in renamed.keys()
            deprecated_callback_name = renamed[callback_name]
            for module_name in module_names:
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
                getattr(self, attr_name+"_keys").append(sys_modules_key)
                getattr(self, attr_name+"_names").append(callback_name)
                self._callback_mapped_module_name\
                    [sys_modules_key] = module_name

    #
    # Interface
    #

    def initialize(self):

        init_start_time = time.time()
        action_handles = None
        try:
            ############# derived method
            action_handles = self._init()
            #############
        except:
            if not self._inside_with_block:
                print("Exception encountered. Scenario tree manager attempting to shut down.")
                print("Original Exception:")
                traceback.print_exception(*sys.exc_info())
                self.close()
            raise

# TODO: move to solver manager
#        self._objective_sense = \
#            self._scenario_tree._scenarios[0]._objective_sense
#        assert all(_s._objective_sense == self._objective_sense
#                   for _s in self._scenario_tree._scenarios)

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

        return action_handles

    #
    # Methods defined by derived class:
    #

    def _init(self):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

#
# A base  that is common to both the serial and distributed master
# manager object.
#

class _ScenarioTreeManager(PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the _ScenarioTreeManager class")

    def __init__(self, *args, **kwds):
        super(_ScenarioTreeManager, self).__init__(*args, **kwds)

        init_start_time = time.time()
        self._scenario_tree = None
        self._objective_sense = None
        # bundle info
        self._scenario_to_bundle_map = {}
        # For the users to modify as they please in the aggregate
        # callback as long as the data placed on it can be serialized
        # by Pyro
        self._aggregate_user_data = {}
        # set to true with the __enter__ method is called
        self._inside_with_block = False

    #
    # Methods defined by derived class:
    #

    def _close_impl(self, bundle_name, scenario_list):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    #
    # Interface:
    #

    def __enter__(self):
        self._inside_with_block = True
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            print("Exception encountered. Scenario tree manager attempting to shut down.")
            print("Original Exception:")
            traceback.print_exception(*args)
        self.close()

    def close(self):
        self._close_impl()
        self._scenario_tree = None
        self._objective_sense = None
        self._scenario_to_bundle_map = {}
        self._aggregate_user_data = {}

class ScenarioTreeManagerSerial(_ScenarioTreeManagerImpl,
                                _ScenarioTreeWorkerImpl,
                                _ScenarioTreeManager,
                                PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeManagerSerial class")

    #
    # scenario instance construction
    #
    safe_register_common_option(_registered_options,
                                "output_instance_construction_time")
    safe_register_common_option(_registered_options,
                                "compile_scenario_instances")

    def __init__(self, *args, **kwds):
        super(ScenarioTreeManagerSerial, self).__init__(*args, **kwds)

    #
    # Abstract methods for _ScenarioTreeManagerImpl:
    #

    def _init(self):
        assert self._scenario_tree is not None
        if self._options.verbose:
            print("Initializing ScenarioTreeManagerSerial with options:")
            self.display_options()

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
                output_instance_construction_time=\
                   self._options.output_instance_construction_time,
                profile_memory=self._options.profile_memory,
                compile_scenario_instances=self._options.compile_scenario_instances)

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
                  "%.2f seconds" % (time.time() - build_start_time))

        #
        # Create bundle instances if needed
        #
        if self._scenario_tree.contains_bundles():
            start_time = time.time()
            if self._options.verbose:
                print("Construction extensive form instances for all bundles.")

            for bundle in self._scenario_tree._scenario_bundles:
                for scenario_name in bundle._scenario_names:
                    if scenario_name in self._scenario_to_bundle_map:
                        raise ValueError(
                            "Unable to form binding instance for bundle %s. "
                            "Scenario %s already belongs to bundle %s."
                            % (bundle_name,
                               scenario_name,
                               self._scenario_to_bundle_map[scenario_name]))
                    self._scenario_to_bundle_map[scenario_name] = bundle._name
                self._form_bundle_binding_instance(bundle._name)

            end_time = time.time()
            if self._options.output_times:
                print("Scenario bundle construction time=%.2f seconds"
                      % (end_time - start_time))

        if self._options.verbose:
            print("ScenarioTreeManagerSerial is successfully "
                  "initialized")

        if len(self._options.aggregategetter_callback_location):
            # Run the user script to collect aggregate scenario data
            for callback_module_key in self._aggregategetter_keys:
                if self._options.verbose:
                    print("Executing user defined aggregategetter callback function "
                          "defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))
                for scenario in self._scenario_tree._scenarios:
                    result = self._callback_function[callback_module_key](
                        self,
                        self._scenario_tree,
                        scenario,
                        self._aggregate_user_data)
                    assert len(result) == 1
                    self._aggregate_user_data.update(result[0])

        if len(self._options.postinit_callback_location):
            # run the user script to initialize variable bounds
            for callback_module_key in self._postinit_keys:
                if self._options.verbose:
                    print("Executing user defined posinit callback function "
                          "defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))
                for scenario in self._scenario_tree._scenarios:
                    self._callback_function[callback_module_key](
                        self,
                        self._scenario_tree,
                        scenario)

        return None

    #
    # Abstract methods for _ScenarioTreeWorkerImpl: None
    #

    #
    # Abstract methods for _ScenarioTreeManager: None required
    #

class ScenarioTreeManagerSPPyroBasic(_ScenarioTreeManagerImpl,
                                     _ScenarioTreeManager,
                                     PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeManagerSPPyroBasic class")

    safe_register_common_option(_registered_options,
                                "pyro_hostname")
    safe_register_common_option(_registered_options,
                                "handshake_with_sppyro")
    safe_register_common_option(_registered_options,
                                "shutdown_pyro")
    SPPyroScenarioTreeServer.register_options(_registered_options)

    def __init__(self, *args, **kwds):
        # distributed worker information
        self._sppyro_server_workers_map = {}
        self._sppyro_worker_server_map = {}
        self._action_manager = None
        self._transmission_paused = False
        super(ScenarioTreeManagerSPPyroBasic, self).__init__(*args, **kwds)

    #
    # Abstract methods for _ScenarioTreeManager:
    #

    def _close_impl(self):
        if self._options.verbose:
            print("Closing ScenarioTreeManagerSPPyro")
        if self._action_manager is not None:
            self.release_scenariotreeservers()
        if self._options.shutdown_pyro:
            print("Shutting down Pyro components.")
            shutdown_pyro_components(num_retries=0)

    #
    # Abstract methods for _ScenarioTreeManagerImpl: None
    #

    def _init(self):
        assert self._scenario_tree is not None
        if self._options.verbose:
            print("Initializing ScenarioTreeManagerSPPyroBasic with options:")
            self.display_options()
            print("")

    #
    # Extended the manager interface for SPPyro
    #

    def acquire_scenariotreeservers(self, num_servers, timeout=None):
        """Acquire a pool of scenario tree servers and initialize the action manager."""
        assert self._action_manager is None
        self._action_manager = SPPyroAsyncActionManager(
            verbose=self._options.verbose,
            host=self._options.pyro_hostname)
        self._action_manager.acquire_servers(num_servers, timeout)
        # extract server options
        server_options = SPPyroScenarioTreeServer.\
                             extract_user_options_to_dict(self._options)
        # override these options just in case this instance factory
        # extracted from an archive
        server_options['model_location'] = \
            self._scenario_tree._scenario_instance_factory._model_filename
        server_options['scenario_tree_location'] = \
            self._scenario_tree._scenario_instance_factory._scenario_tree_filename

        # transmit setup requests
        action_handles = []
        self.pause_transmit()
        for server_name in self._action_manager.server_pool:
            # This will make sure we don't come across any lingering
            # task results from a previous run that ended badly
            self._action_manager.client.clear_queue(override_type=server_name)
            action_handles.append(
                self._action_manager.queue(
                    server_name,
                    action="SPPyroScenarioTreeServer_setup",
                    options=server_options,
                    generate_response=True))
            self._sppyro_server_workers_map[server_name] = []
        self.unpause_transmit()
        self._action_manager.wait_all(action_handles)

        return len(self._action_manager.server_pool)

    def release_scenariotreeservers(self):
        """Release the pool of scenario tree servers and destroy the action manager."""
        assert self._action_manager is not None
        if self._options.verbose:
            print("Releasing %s scenario tree servers"
                  % (len(self._action_manager.server_pool)))

        if self._transmission_paused:
            print("Unpausing pyro transmissions in preparation for "
                  "releasing scenario tree servers")
            self.unpause_transmit()
        # copy the keys since the remove_worker function is modifying
        # the dict
        for worker_name in list(self._sppyro_worker_server_map.keys()):
            self.remove_worker(worker_name)
        self._action_manager.close()
        self._action_manager = None
        self._sppyro_server_workers_map = {}
        self._sppyro_worker_server_map = {}

    def pause_transmit(self):
        """Pause transmission of action requests. Return whether
        transmission was already paused."""
        assert self._action_manager is not None
        self._action_manager.begin_bulk()
        was_paused = self._transmission_paused
        self._transmission_paused = True
        return was_paused

    def unpause_transmit(self):
        """Unpause transmission of action requests and bulk transmit
        anything queued."""
        assert self._action_manager is not None
        self._action_manager.end_bulk()
        self._transmission_paused = False

    def complete_actions(self, action_handles, ignore_others=False):
        assert self._action_manager is not None
        if action_handles is None:
            return None
        results = {}
        while len(results) < len(action_handles):
            ah = self._action_manager.wait_any()
            if ah in action_handles:
                results[ah.id] = self._action_manager.get_results(ah)
            elif not ignore_others:
                raise ValueError(
                    "Encountered unexpected action handle id=%s when "
                    "completing scenario tree manager async action list.")
        return results

    def add_worker(self,
                   worker_name,
                   init_type,
                   init_data,
                   worker_options,
                   worker_registered_name='ScenarioTreeWorkerBasic',
                   return_action_handle=False,
                   server_name=None):
        assert self._action_manager is not None

        if server_name is None:
            # Find a server that currently owns the fewest workers
            server_name = \
                min(self._action_manager.server_pool,
                    key=lambda k: len(self._sppyro_server_workers_map.get(k,[])))

        if self._options.verbose:
            print("Initializing worker with name %s on scenario tree server %s"
                  % (worker_name, server_name))

        generate_response = \
            self._options.handshake_with_sppyro or return_action_handle

        if self._transmission_paused:
            if self._options.handshake_with_sppyro and \
               (not return_action_handle):
                raise ValueError(
                    "Unable to add worker. "
                    "Pyro transmissions are currently paused but the "
                    "handshake_with_sppyro option is currently set to True and "
                    "return_action_handle is False. These settings require action "
                    "handles be collected within this method. Pyro transmissions must "
                    "be unpaused in order for this to take place.")

        if isinstance(worker_options, ConfigBlock):
            worker_class = get_registered_worker_type(worker_registered_name)
            try:
                worker_options = worker_class.\
                                 extract_user_options_to_dict(worker_options)
            except KeyError:
                raise KeyError(
                    "Unable to serialize options for registered worker name %s "
                    "(class=%s). The worker_options did not seem to match the "
                    "registered options on the worker class. Did you forget to "
                    "register them? Message: %s" % (worker_registered_name,
                                                    worker_type.__name__,
                                                    str(sys.exc_info()[1])))

        action_handle = self._action_manager.queue(
            server_name,
            action="SPPyroScenarioTreeServer_initialize",
            worker_type=worker_registered_name,
            worker_name=worker_name,
            init_type=init_type.key,
            init_data=init_data,
            options=worker_options,
            generate_response=generate_response)

        if generate_response and (not return_action_handle):
            self._action_manager.wait_all([action_handle])

        self._sppyro_server_workers_map[server_name].append(worker_name)
        self._sppyro_worker_server_map[worker_name] = server_name

        return action_handle if (return_action_handle) else None

    def remove_worker(self, worker_name):
        assert self._action_manager is not None
        server_name = self.get_server_for_worker(worker_name)
        self._action_manager.queue(
            server_name,
            action="SPPyroScenarioTreeServer_release",
            worker_name=worker_name,
            generate_response=False)
        self._sppyro_server_workers_map[server_name].remove(worker_name)
        del self._sppyro_worker_server_map[worker_name]

    def get_server_for_worker(self, worker_name):
        try:
            return self._sppyro_worker_server_map[worker_name]
        except KeyError:
            raise KeyError(
                "Scenario tree worker with name %s does not exist on "
                "any scenario tree servers" % (worker_name))

    #
    # Invoke an external function passing the scenario tree worker
    # as the first argument. The remaining arguments and how this
    # function is invoked is controlled by the invocation_type keyword.
    # By default, a single invocation takes place
    #

    def transmit_external_function_invocation_to_worker(
            self,
            worker_name,
            module_name,
            function_name,
            invocation_type=InvocationType.SingleInvocation,
            return_action_handle=False,
            function_args=None,
            function_kwds=None):

        assert self._action_manager is not None
        if self._options.verbose:
            print("Transmitting external function invocation request to "
                  "scenario tree worker with name %s" % (worker_name))

        generate_response = \
            self._options.handshake_with_sppyro or return_action_handle

        if self._transmission_paused:
            if self._options.handshake_with_sppyro and \
               (not return_action_handle):
                raise ValueError(
                    "Unable to transmit external function invocation. "
                    "Pyro transmissions are currently paused but the "
                    "handshake_with_spyro option is currently set to True and "
                    "return_action_handle is False. These settings require action "
                    "handles be collected within this method. Pyro transmissions must "
                    "be unpaused in order for this to take place.")

        action_handle = self._action_manager.queue(
            self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action="invoke_external_function",
            generate_response=generate_response,
            args=(invocation_type.key,
                  module_name,
                  function_name,
                  function_args,
                  function_kwds),
            kwds={})

        if generate_response and (not return_action_handle):
            self._action_manager.wait_all([action_handle])

        return action_handle if (return_action_handle) else None

    #
    # Invoke an external function passing each scenario tree worker
    # as the first argument. The remaining arguments and how this
    # function is invoked is controlled by the invocation_type keyword.
    # By default, a single invocation takes place
    #

    def transmit_external_function_invocation(
            self,
            module_name,
            function_name,
            invocation_type=InvocationType.SingleInvocation,
            return_action_handles=False,
            function_args=None,
            function_kwds=None):

        start_time = time.time()

        if self._options.verbose:
            print("Transmitting external function invocation request "
                  "to scenario tree workers")

        generate_response = \
            self._options.handshake_with_sppyro or return_action_handles

        was_paused = self.pause_transmit()
        if was_paused:
            if self._options.handshake_with_sppyro and \
               (not return_action_handles):
                raise ValueError(
                    "Unable to transmit external function invocation. "
                    "Pyro transmissions are currently paused but the "
                    "handshake_with_spyro option is currently set to True and "
                    "return_action_handles is False. These settings require action "
                    "handles be collected within this method. Pyro transmissions must "
                    "be unpaused in order for this to take place.")

        action_handles = []
        for worker_name in self._sppyro_worker_server_map:

            action_handles.append(
                self._action_manager.queue(
                    self.get_server_for_worker(worker_name),
                    worker_name=worker_name,
                    action="invoke_external_function",
                    generate_response=generate_response,
                    args=(invocation_type.key,
                          module_name,
                          function_name,
                          function_args,
                          function_kwds),
                    kwds={}))

        if not was_paused:
            self.unpause_transmit()

        if generate_response and (not return_action_handles):
            self._action_manager.wait_all(action_handles)

        end_time = time.time()

        if self._options.output_times:
            print("External function invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return action_handles if (return_action_handles) else None

    #
    # Invoke a method on a scenario tree worker
    #

    def transmit_method_invocation_to_worker(
            self,
            worker_name,
            method_name,
            return_action_handle=False,
            method_args=None,
            method_kwds=None):

        if self._options.verbose:
            print("Transmitting method invocation request to "
                  "scenario tree worker with name %s" % (worker_name))

        generate_response = \
            self._options.handshake_with_sppyro or return_action_handle

        if self._transmission_paused:
            if self._options.handshake_with_sppyro and \
               (not return_action_handle):
                raise ValueError(
                    "Unable to transmit method invocation. "
                    "Pyro transmissions are currently paused but the "
                    "handshake_with_spyro option is currently set to True and "
                    "return_action_handle is False. These settings require action "
                    "handles be collected within this method. Pyro transmissions must "
                    "be unpaused in order for this to take place.")

        action_handle = self._action_manager.queue(
            self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action=method_name,
            generate_response=generate_response,
            args=method_args if (method_args is not None) else (),
            kwds=method_kwds if (method_kwds is not None) else {})

        if generate_response and (not return_action_handle):
            self._action_manager.wait_all([action_handle])

        return action_handle if (return_action_handle) else None

    #
    # Invoke a method on the respective scenario tree workers
    #

    def transmit_method_invocation(
            self,
            method_name,
            return_action_handles=False,
            method_args=None,
            method_kwds=None):
        assert self._action_manager is not None
        start_time = time.time()

        if self._options.verbose:
            print("Transmitting method invocation request "
                  "to scenario tree workers")

        action_handles = []

        generate_response = \
            self._options.handshake_with_sppyro or return_action_handles

        was_paused = self.pause_transmit()
        if was_paused:
            if self._options.handshake_with_sppyro and \
               (not return_action_handles):
                raise ValueError(
                    "Unable to transmit method invocation. "
                    "Pyro transmissions are currently paused but the "
                    "handshake_with_spyro option is currently set to True and "
                    "return_action_handles is False. These settings require action "
                    "handles be collected within this method. Pyro transmissions must "
                    "be unpaused in order for this to take place.")

        action_handles = []
        for worker_name in self._sppyro_worker_server_map:

            action_handles.append(
                self._action_manager.queue(
                    self.get_server_for_worker(worker_name),
                    worker_name=worker_name,
                    action=method_name,
                    generate_response=generate_response,
                    args=method_args if (method_args is not None) else (),
                    kwds=method_kwds if (method_kwds is not None) else {}))

        if not was_paused:
            self.unpause_transmit()

        if generate_response and (not return_action_handles):
            self._action_manager.wait_all(action_handles)

        end_time = time.time()

        if self._options.output_times:
            print("Method invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return action_handles if (return_action_handles) else None

#
# This class extends the initialization process of ScenarioTreeManagerSPPyroBasic
# so that scenario tree servers are automatically acquired and assigned worker processes
# for that manage scenarios / bundles.
#

class ScenarioTreeManagerSPPyro(ScenarioTreeManagerSPPyroBasic,
                                PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeManagerSPPyro class")
    safe_register_common_option(_registered_options,
                                "sppyro_required_servers")
    safe_register_common_option(_registered_options,
                                "sppyro_find_servers_timeout")
    safe_register_common_option(_registered_options,
                                "sppyro_serial_workers")

    def __init__(self, *args, **kwds):
        self._scenario_to_worker_map = {}
        self._bundle_to_worker_map = {}
        self._worker_registered_name = kwds.pop('registered_worker_name',
                                                'ScenarioTreeWorkerBasic')
        super(ScenarioTreeManagerSPPyro, self).__init__(*args, **kwds)

    def _initialize_scenariotree_workers(self):

        start_time = time.time()

        if self._options.verbose:
            print("Transmitting scenario tree worker initializations")

        if len(self._action_manager.server_pool) == 0:
            raise RuntimeError(
                "No scenario tree server processes have been acquired!")

        if self._scenario_tree.contains_bundles():
            jobs = [(bundle._name,
                     WorkerInitType.ScenarioBundle,
                     bundle._scenario_names)
                    for bundle in self._scenario_tree._scenario_bundles]
        else:
            jobs = [(scenario._name, WorkerInitType.Scenario, scenario._name)
                    for scenario in self._scenario_tree._scenarios]

        assert len(self._sppyro_server_workers_map) == \
            len(self._action_manager.server_pool)
        assert len(self._sppyro_worker_server_map) == 0
        scenario_instance_factory = \
            self._scenario_tree._scenario_instance_factory

        worker_type = SPPyroScenarioTreeServer.\
                      get_registered_worker_type(self._worker_registered_name)
        worker_options = None
        try:
            worker_options = worker_type.\
                             extract_user_options_to_dict(self._options)
        except KeyError:
            raise KeyError(
                "Unable to extract options for registered worker name %s (class=%s). "
                "Did you forget to register the worker options into the options "
                "object passed into this class? Message: %s"
                  % (self._worker_registered_name,
                     worker_type.__name__,
                     str(sys.exc_info()[1])))

        assert worker_options is not None
        worker_initializations = dict((server_name, []) for server_name
                                      in self._action_manager.server_pool)
        for server_name in itertools.cycle(self._action_manager.server_pool):
            if len(jobs) == 0:
                break
            worker_initializations[server_name].append(jobs.pop())

        assert not self._transmission_paused
        if not self._options.handshake_with_sppyro:
            self.pause_transmit()
        initialization_action_handles = []
        for cntr, server_name in enumerate(worker_initializations):

            if self._options.sppyro_serial_workers:

                #
                # Multiple workers per server
                #

                for worker_name, init_type, init_data \
                       in worker_initializations[server_name]:

                    self.add_worker(
                        worker_name,
                        init_type,
                        init_data,
                        worker_options,
                        worker_registered_name=self._worker_registered_name,
                        server_name=server_name)

                    if init_type == WorkerInitType.ScenarioBundle:
                        self._bundle_to_worker_map[worker_name] = worker_name
                        assert self._scenario_tree.contains_bundle(worker_name)
                        for scenario_name in init_data:
                            self._scenario_to_worker_map[scenario_name] = worker_name
                    else:
                        assert init_type == WorkerInitType.Scenario
                        assert self._scenario_tree.contains_scenario(worker_name)
                        self._scenario_to_worker_map[worker_name] = worker_name

            else:

                #
                # One worker per server
                #

                init_type = worker_initializations[server_name][0][1]
                assert all(init_type == _init_type for _,_init_type,_ \
                           in worker_initializations[server_name])
                if init_type == WorkerInitType.ScenarioBundle:
                    init_type = WorkerInitType.ScenarioBundleList
                    worker_name = 'Worker_BundleGroup'+str(cntr)
                    init_data = {}
                    for bcnt, (_,_,data) in \
                           enumerate(worker_initializations[server_name]):
                        init_data['Bundle'+str(bcnt)] = data
                else:
                    assert init_type == WorkerInitType.Scenario
                    init_type = WorkerInitType.ScenarioList
                    worker_name = 'Worker_ScenarioGroup'+str(cntr)
                    init_data = [data for _,_,data \
                                 in worker_initializations[server_name]]

                action_handle = self.add_worker(
                    worker_name,
                    init_type,
                    init_data,
                    worker_options,
                    worker_registered_name=self._worker_registered_name,
                    return_action_handle=True,
                    server_name=server_name)

                if self._options.handshake_with_sppyro:
                    self._action_manager.wait_all([action_handle])
                else:
                    initialization_action_handles.append(action_handle)

                if init_type == WorkerInitType.ScenarioBundleList:
                    self._bundle_to_worker_map[worker_name] = worker_name
                    assert not self._scenario_tree.contains_bundle(worker_name)
                    for scenario_name in init_data:
                        self._scenario_to_worker_map[scenario_name] = worker_name
                else:
                    assert init_type == WorkerInitType.ScenarioList
                    assert not self._scenario_tree.contains_scenario(worker_name)
                    for scenario_name in init_data:
                        self._scenario_to_worker_map[scenario_name] = worker_name

        if not self._options.handshake_with_sppyro:
            self.unpause_transmit()

        end_time = time.time()

        if self._options.output_times:
            print("Initialization transmission time=%.2f seconds"
                  % (end_time - start_time))

        return initialization_action_handles

    #
    # Override abstract methods for _ScenarioTreeManagerImpl
    # that were implemented by ScenarioTreeManagerSPPyroBasic
    #

    def _init(self):
        assert self._scenario_tree is not None
        if self._options.verbose:
            print("Initializing ScenarioTreeManagerSPPyro with options:")
            self.display_options()
            print("")

        if self._scenario_tree.contains_bundles():
            num_jobs = len(self._scenario_tree._scenario_bundles)
            if self._options.verbose:
                print("Bundle jobs available: %s"
                      % (str(num_jobs)))
        else:
            num_jobs = len(self._scenario_tree._scenarios)
            if self._options.verbose:
                print("Scenario jobs available: %s"
                      % (str(num_jobs)))

        servers_expected = self._options.sppyro_required_servers
        if (servers_expected is None):
            servers_expected = num_jobs

        timeout = self._options.sppyro_find_servers_timeout if \
                  (self._options.sppyro_required_servers is None) else \
                  None

        if self._options.verbose:
            if servers_expected is None:
                assert timeout is not None
                print("Using timeout of %s seconds to aquire up to "
                      "%s servers" % (timeout, num_jobs))
            else:
                print("Waiting to acquire exactly %s servers to distribute "
                      "work over %s jobs" % (servers_expected, num_jobs))

        self.acquire_scenariotreeservers(servers_expected, timeout=timeout)

        if self._options.verbose:
            print("Broadcasting requests to initialize workers "
                  "on scenario tree servers")

        initialization_action_handles = self._initialize_scenariotree_workers()

        if self._options.verbose:
            print("Distributed scenario tree initialization "
                  "requests successfully transmitted")

        worker_names = sorted(self._sppyro_worker_server_map)

        # run the user script to collect aggregate scenario data. This
        # can slow down initialization as syncronization across all
        # scenario tree servers is required following serial
        # executation
        if len(self._options.aggregategetter_callback_location):
            assert not self._transmission_paused
            for callback_module_key, callback_name in zip(self._aggregategetter_keys,
                                                          self._aggregategetter_names):
                if self._options.verbose:
                    print("Transmitting invocation of user defined aggregategetter "
                          "callback function defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))

                for worker_name in worker_names:

                   action_handle = self.transmit_external_function_invocation_to_worker(
                       worker_name,
                       self._callback_mapped_module_name[callback_module_key],
                       callback_name,
                       invocation_type=InvocationType.PerScenarioChainedInvocation,
                       return_action_handle=True,
                       function_args=(self._aggregate_user_data,))
                   result = None
                   while (1):
                       ah = self._action_manager.wait_any()
                       if ah == action_handle:
                           result = self._action_manager.get_results(ah)
                           break
                       else:
                           assert ah in initialization_action_handles
                           initialization_action_handles.remove(ah)
                   assert len(result) == 1
                   self._aggregate_user_data = result[0]

            # Transmit aggregate state to scenario tree servers
            if self._options.verbose:
                print("Broadcasting final aggregate data "
                      "to scenario tree servers")

            ahs = self.transmit_method_invocation(
                "assign_aggregate_user_data",
                method_args=(self._aggregate_user_data,),
                return_action_handles=self._options.handshake_with_sppyro)
            if self._options.handshake_with_sppyro:
                initialization_action_handles.extend(ahs)

        # run the user script to initialize variable bounds
        if len(self._options.postinit_callback_location):

            for callback_module_key, callback_name in zip(self._postinit_keys,
                                                          self._postinit_names):
                if self._options.verbose:
                    print("Transmitting invocation of user defined postinit "
                          "callback function defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))

                # Transmit invocation to scenario tree workers
                ahs = self.transmit_external_function_invocation(
                    self._callback_mapped_module_name[callback_module_key],
                    callback_name,
                    invocation_type=InvocationType.PerScenarioInvocation,
                    return_action_handles=self._options.handshake_with_sppyro)
                if self._options.handshake_with_sppyro:
                    initialization_action_handles.extend(ahs)

        return initialization_action_handles

    #
    # Extended the manager interface for SPPyro
    #

    def get_worker_for_scenario(self, scenario_name):
        assert self._scenario_tree.contains_scenario(scenario_name)
        return self._scenario_to_worker_map[scenario_name]

    def get_server_for_scenario(self, scenario_name):
        return self.get_server_for_worker(
            self.get_worker_for_scenario(scenario_name))

    def get_worker_for_bundle(self, bundle_name):
        assert self._scenario_tree.contains_bundle(bundle_name)
        return self._bundle_to_worker_map[bundle_name]

    def get_server_for_bundle(self, bundle_name):
        return self.get_server_for_worker(
            self.get_worker_for_bundle(bundle_name))
