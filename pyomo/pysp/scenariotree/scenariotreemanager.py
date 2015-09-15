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
from collections import defaultdict

import pyutilib.misc
from pyutilib.pyro import shutdown_pyro_components
from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyomo.opt.parallel.manager import ActionHandle
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.util.misc import load_external_module
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.sppyro_action_manager \
    import SPPyroAsyncActionManager
from pyomo.pysp.scenariotree.scenariotreeserver \
    import SPPyroScenarioTreeServer
from pyomo.pysp.scenariotree.scenariotreeserverutils \
    import (ScenarioWorkerInit,
            BundleWorkerInit,
            WorkerInit,
            WorkerInitType,
            InvocationType,
            _map_deprecated_invocation_type)
from pyomo.pysp.ef import create_ef_instance

from six import iteritems, itervalues, StringIO
from six.moves import xrange

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

    def _invoke_external_function_impl(
            self,
            module_name,
            function_name,
            invocation_type=InvocationType.Single,
            function_args=None,
            function_kwds=None):
        invocation_type = _map_deprecated_invocation_type(invocation_type)

        this_module = pyutilib.misc.import_file(module_name)

        module_attrname = function_name
        subname = None
        if not hasattr(this_module, module_attrname):
            if "." in module_attrname:
                module_attrname, subname = function_name.split(".",1)
            if not hasattr(this_module, module_attrname):
                raise RuntimeError(
                    "Function="+function_name+" is not present "
                    "in module="+module_name)

        call_objects = None
        if invocation_type == InvocationType.Single:
            pass
        elif (invocation_type == InvocationType.PerBundle) or \
             (invocation_type == InvocationType.PerBundleChained):
            if not self._scenario_tree.contains_bundles():
                raise ValueError(
                    "Received request for bundle invocation type "
                    "but the scenario tree does not contain bundles.")
            call_objects = self._scenario_tree.bundles
        elif (invocation_type == InvocationType.PerScenario) or \
             (invocation_type == InvocationType.PerScenarioChained):
            call_objects = self._scenario_tree.scenarios
        else:
            raise ValueError("Unexpected function invocation type '%s'. "
                             "Expected one of %s"
                             % (invocation_type,
                                [str(v) for v in InvocationType._values]))

        function = getattr(this_module, module_attrname)
        if subname is not None:
            function = getattr(function, subname)

        if function_kwds is None:
            function_kwds = {}
        if function_args is None:
            function_args = ()

        result = None
        if (invocation_type == InvocationType.Single):

            if function_args is None:
                function_args = ()
            result = function(self,
                              self._scenario_tree,
                              *function_args,
                              **function_kwds)
        elif (invocation_type == InvocationType.PerBundleChained) or \
             (invocation_type == InvocationType.PerScenarioChained):

            if (function_args is None) or \
               (type(function_args) is not tuple) or \
               (len(function_args) == 0):
                raise ValueError("Function invocation type %s must be executed "
                                 "with function_args keyword set to non-empty "
                                 "tuple type. Invalid value: %s"
                                 % (invocation_type.key, function_args))

            result = function_args
            for call_object in call_objects:
                result = function(self,
                                  self._scenario_tree,
                                  call_object,
                                  *result,
                                  **function_kwds)
        else:

            if function_args is None:
                function_args = ()
            result = dict((call_object._name, function(self,
                                                       self._scenario_tree,
                                                       call_object,
                                                       *function_args,
                                                       **function_kwds))
                          for call_object in call_objects)

        return result

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

    #
    # Note: These Async objects can be cleaned up.
    #       This is a first draft.
    #
    class Async(object):
        def complete(self):
            raise NotImplementedError(type(self).__name__+": This method is abstract")

    class AsyncResult(Async):

        __slots__ = ('_manager',
                     '_result',
                     '_action_handle_data',
                     '_invocation_type',
                     '_map_result')

        def __init__(self,
                     manager,
                     result=None,
                     action_handle_data=None,
                     map_result=None):
            assert manager is not None
            if result is not None:
                assert action_handle_data is None
            if map_result is not None:
                assert result is None
                assert action_handle_data is not None
            self._manager = manager
            self._action_handle_data = action_handle_data
            self._result = result
            self._map_result = map_result

        def complete(self):

            if self._result is not None:
                if isinstance(self._result,
                              _ScenarioTreeManagerImpl.Async):
                    self._result = self._result.complete()
                return self._result

            if self._action_handle_data is None:
                assert self._result is None
                return None

            result = None
            if isinstance(self._action_handle_data, ActionHandle):
                result = self._manager._action_manager.wait_for(
                    self._action_handle_data)
                if self._map_result is not None:
                    result = self._map_result(self._action_handle_data, result)
            else:
                ah_to_result = self._manager._action_manager.wait_all(
                    self._action_handle_data)
                if self._map_result is not None:
                    result = self._map_result(ah_to_result)
                else:
                    result = dict((self._action_handle_data[ah], ah_to_result[ah])
                                  for ah in ah_to_result)
            self._result = result
            return self._result

    class AsyncResultChain(Async):
        __slots__ = ("_results", "_return_index")

        def __init__(self, results, return_index=-1):
            self._results = results
            self._return_index = return_index

        def complete(self):
            for i in xrange(len(self._results)):
                assert isinstance(self._results[i],
                                  _ScenarioTreeManagerImpl.Async)
                self._results[i] = self._results[i].complete()
            return self._results[self._return_index]

    class AsyncResultCallback(Async):
        __slots__ = ("_result", "_done")

        def __init__(self, result):
            self._result = result
            self._done = False

        def complete(self):
            if not self._done:
                self._result = self._result()
                self._done = True
            return self._result

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
                    bundles=self._options.scenario_bundle_specification,
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
                module, sys_modules_key = \
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

    def initialize(self, async=False):

        init_start_time = time.time()
        result = None
        try:
            if self._options.verbose:
                print("Initializing %s with options:"
                      % (type(self).__name__))
                self.display_options()
                print("")
            ############# derived method
            async_handle = self._init()
            if async:
                result = async_handle
            else:
                result = async_handle.complete()
            #############
            if self._options.verbose:
                print("%s is successfully initialized"
                      % (type(self).__name__))

        except:
            if not self._inside_with_block:
                print("Exception encountered. Scenario tree manager "
                      "attempting to shut down.")
                print("Original Exception:")
                traceback.print_exception(*sys.exc_info())
                self.close()
            raise

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

        return result

    #
    # Abstract Interface
    #

    def _init(self):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def worker_names(self):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def get_worker_for_scenario(self, scenario_name):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def get_server_for_scenario(self, scenario_name):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def get_worker_for_bundle(self, bundle_name):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def get_server_for_bundle(self, bundle_name):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def invoke_external_function_on_worker(*args, **kwds):
        """Invoke an external function on a scenario tree worker,
        passing the scenario tree worker as the first argument, and
        the worker scenario tree as the second argument. The remaining
        arguments and how this function is invoked is controlled by
        the invocation_type keyword. By default, a single invocation
        takes place."""
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def invoke_method_on_worker(*args, **kwds):
        """Invoke a method on a scenario tree worker."""
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
        # bundle info
        self._scenario_to_bundle_map = {}
        # For the users to modify as they please in the aggregate
        # callback as long as the data placed on it can be serialized
        # by Pyro
        self._aggregate_user_data = {}
        # set to true with the __enter__ method is called
        self._inside_with_block = False

    #
    # Interface:
    #

    @property
    def scenario_tree(self):
        return self._scenario_tree

    def __enter__(self):
        self._inside_with_block = True
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            sys.stderr.write("Exception encountered. Scenario tree manager attempting "
                             "to shut down.\n")
            tmp = StringIO()
            _args = list(args) + [None, tmp]
            traceback.print_exception(*_args)
            try:
                self.close()
            except:
                sys.stderr.write("Exception encountered during emergency scenario "
                                 "tree manager shutdown. Printing original exception "
                                 "here:\n")
                sys.stderr.write(tmp.getvalue())
                raise
        else:
            self.close()

    def close(self):
        self._close_impl()
        self._scenario_tree = None
        self._scenario_to_bundle_map = {}
        self._aggregate_user_data = {}

    #
    # Abstract Interface:
    #

    def _close_impl(self, bundle_name, scenario_list):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def invoke_external_function(self, *args, **kwds):
        """Invoke an external function all scenario tree workers,
        passing the scenario tree worker as the first argument, and
        the worker scenario tree as the second argument. The remaining
        arguments and how this function is invoked is controlled by
        the invocation_type keyword. By default, a single invocation
        takes place."""
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def invoke_method(self, *args, **kwds):
        """Invoke a method on all scenario tree workers."""
        raise NotImplementedError(type(self).__name__+": This method is abstract")

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
        self._worker_name = 'ScenarioTreeManagerSerial:MainWorker'
        super(ScenarioTreeManagerSerial, self).__init__(*args, **kwds)

    #
    # Abstract methods for _ScenarioTreeManagerImpl:
    #

    def _init(self):
        assert self._scenario_tree is not None

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

        return self.AsyncResult(
            self, result={self._worker_name: True})

    def worker_names(self):
        return (self._worker_name,)

    def get_worker_for_scenario(self, scenario_name):
        assert self._scenario_tree.contains_scenario(scenario_name)
        return self._worker_name

    def get_worker_for_bundle(self, bundle_name):
        assert self._scenario_tree.contains_bundle(bundle_name)
        return self._worker_name

    def invoke_external_function_on_worker(
            self,
            worker_name,
            module_name,
            function_name,
            invocation_type=InvocationType.Single,
            function_args=None,
            function_kwds=None,
            oneway=False,
            async=False):
        """Invoke an external function on a scenario tree worker,
        passing the scenario tree worker as the first argument, and
        the worker scenario tree as the second argument. The remaining
        arguments and how this function is invoked is controlled by
        the invocation_type keyword. By default, a single invocation
        takes place."""
        invocation_type = _map_deprecated_invocation_type(invocation_type)

        assert worker_name == self._worker_name
        start_time = time.time()

        if self._options.verbose:
            print("Invoking external function=%s in module=%s "
                  "on worker=%s"
                  % (function_name, module_name, worker_name))

        result = self._invoke_external_function_impl(module_name,
                                                     function_name,
                                                     invocation_type=invocation_type,
                                                     function_args=function_args,
                                                     function_kwds=function_kwds)

        if oneway:
            result = None
        if async:
            result = self.AsyncResult(self, result=result)

        end_time = time.time()
        if self._options.output_times:
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def invoke_method_on_worker(
            self,
            worker_name,
            method_name,
            method_args=None,
            method_kwds=None,
            oneway=False,
            async=False):
        """Invoke a method on a scenario tree worker."""

        assert worker_name == self._worker_name
        start_time = time.time()

        if self._options.verbose:
            print("Invoking method=%s on worker=%s"
                  % (method_name, self._worker_name))

        result = getattr(self, method_name)(*method_args, **method_kwds)

        if oneway:
            result = None
        if async:
            result = self.AsyncResult(self, result=result)

        end_time = time.time()
        if self._options.output_times:
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    #
    # Abstract methods for _ScenarioTreeWorkerImpl: None
    #

    #
    # Abstract methods for _ScenarioTreeManager:
    #

    def invoke_external_function(self,
                                 module_name,
                                 function_name,
                                 invocation_type=InvocationType.Single,
                                 function_args=None,
                                 function_kwds=None,
                                 worker_names=None,
                                 oneway=False,
                                 async=False):
        """Invoke an external function all scenario tree workers,
        passing the scenario tree worker as the first argument, and
        the worker scenario tree as the second argument. The remaining
        arguments and how this function is invoked is controlled by
        the invocation_type keyword. By default, a single invocation
        takes place."""
        invocation_type = _map_deprecated_invocation_type(invocation_type)

        if worker_names is not None:
            assert all(worker_name == self._worker_name
                       for worker_name in worker_names)

        result = self.invoke_external_function_on_worker(
            self._worker_name,
            module_name,
            function_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds,
            oneway=oneway,
            async=False)

        if not oneway:
            if invocation_type == InvocationType.Single:
                result = {self._worker_name: result}
        if async:
            result = self.AsyncResult(self, result=result)

        return result

    def invoke_method(
            self,
            method_name,
            method_args=None,
            method_kwds=None,
            worker_names=None,
            oneway=False,
            async=False):
        """Invoke a method on all scenario tree workers."""

        if worker_names is not None:
            assert all(worker_name == self._worker_name
                       for worker_name in worker_names)

        result = self.invoke_method_on_worker(
            self._worker_name,
            method_name,
            method_args=method_args,
            method_kwds=method_kwds,
            oneway=oneway,
            async=False)

        if not oneway:
            result = {self._worker_name: result}
        if async:
            result = self.AsyncResult(self, result=result)

        return result

class ScenarioTreeManagerSPPyroBasic(_ScenarioTreeManagerImpl,
                                     _ScenarioTreeManager,
                                     PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeManagerSPPyroBasic class")

    safe_register_common_option(_registered_options,
                                "pyro_hostname")
    safe_register_common_option(_registered_options,
                                "shutdown_pyro")
    safe_register_common_option(_registered_options,
                                "shutdown_sppyro_servers")
    SPPyroScenarioTreeServer.register_options(_registered_options)

    def __init__(self, *args, **kwds):
        # distributed worker information
        self._sppyro_server_workers_map = {}
        self._sppyro_worker_server_map = {}
        # the same as the .keys() of the above map
        # but won't suffer from stochastic iteration
        # order python dictionaries
        self._sppyro_worker_list = []
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

    def invoke_external_function_on_worker(
            self,
            worker_name,
            module_name,
            function_name,
            invocation_type=InvocationType.Single,
            function_args=None,
            function_kwds=None,
            oneway=False,
            async=False):
        """Invoke an external function on a scenario tree worker,
        passing the scenario tree worker as the first argument, and
        the worker scenario tree as the second argument. The remaining
        arguments and how this function is invoked is controlled by
        the invocation_type keyword. By default, a single invocation
        takes place."""
        invocation_type = _map_deprecated_invocation_type(invocation_type)

        assert self._action_manager is not None
        assert worker_name in self._sppyro_worker_list
        start_time = time.time()

        if self._options.verbose:
            print("Invoking external function=%s in module=%s "
                  "on worker=%s"
                  % (function_name, module_name, worker_name))

        action_handle = self._invoke_external_function_on_worker(
            worker_name,
            module_name,
            function_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds,
            oneway=oneway)

        if oneway:
            action_handle = None

        result = self.AsyncResult(
            self, action_handle_data=action_handle)

        if not async:
            result = result.complete()

        end_time = time.time()
        if self._options.output_times:
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    #
    # Invoke a method on a scenario tree worker
    #

    def invoke_method_on_worker(
            self,
            worker_name,
            method_name,
            method_args=None,
            method_kwds=None,
            oneway=False,
            async=False):
        """Invoke a method on a scenario tree worker."""

        assert self._action_manager is not None
        assert worker_name in self._sppyro_worker_list
        start_time = time.time()

        if self._options.verbose:
            print("Invoking method=%s on worker=%s"
                  % (method_name, worker_name))

        action_handle = self._invoke_method_on_worker(
            worker_name,
            method_name,
            method_args=method_args,
            method_kwds=method_kwds,
            oneway=oneway)

        if oneway:
            action_handle = None

        result = self.AsyncResult(
            self, action_handle_data=action_handle)

        if not async:
            result = result.complete()

        end_time = time.time()
        if self._options.output_times:
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    #
    # Abstract methods for _ScenarioTreeManagerImpl: None
    #

    def _init(self):
        assert self._scenario_tree is not None
        return self.AsyncResult(
            self, result=True)

    #
    # Extended the manager interface for SPPyro
    #

    def acquire_scenariotreeservers(self, num_servers, timeout=None):
        """Acquire a pool of scenario tree servers and initialize the
        action manager."""

        assert self._action_manager is None
        self._action_manager = SPPyroAsyncActionManager(
            verbose=self._options.verbose,
            host=self._options.pyro_hostname)
        self._action_manager.acquire_servers(num_servers, timeout=timeout)
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
        """Release the pool of scenario tree servers and destroy the
        action manager."""

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

        generate_response = None
        action_name = None
        if self._options.shutdown_sppyro_servers:
            action_name = 'SPPyroScenarioTreeServer_shutdown'
            generate_response = False
        else:
            action_name = 'SPPyroScenarioTreeServer_reset'
            generate_response = True

        # transmit reset or shutdown requests
        action_handles = []
        self.pause_transmit()
        for server_name in self._action_manager.server_pool:
            action_handles.append(self._action_manager.queue(
                server_name,
                action=action_name,
                generate_response=generate_response))
        self.unpause_transmit()
        if generate_response:
            self._action_manager.wait_all(action_handles)

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

    def add_worker(self,
                   worker_name,
                   worker_init,
                   worker_options,
                   worker_registered_name,
                   server_name=None,
                   oneway=False):

        assert self._action_manager is not None

        if server_name is None:
            # Find a server that currently owns the fewest workers
            server_name = \
                min(self._action_manager.server_pool,
                    key=lambda k: len(self._sppyro_server_workers_map.get(k,[])))

        if self._options.verbose:
            print("Initializing worker with name %s on scenario tree server %s"
                  % (worker_name, server_name))

        if isinstance(worker_options, ConfigBlock):
            worker_class = SPPyroScenarioTreeServer.\
                           get_registered_worker_type(worker_registered_name)
            try:
                worker_options = worker_class.\
                                 extract_user_options_to_dict(worker_options,
                                                              sparse=True)
            except KeyError:
                raise KeyError(
                    "Unable to serialize options for registered worker name %s "
                    "(class=%s). The worker_options did not seem to match the "
                    "registered options on the worker class. Did you forget to "
                    "register them? Message: %s" % (worker_registered_name,
                                                    worker_type.__name__,
                                                    str(sys.exc_info()[1])))

        if type(worker_init) is not WorkerInit:
            raise TypeError("worker_init argument has invalid type %s. "
                            "Must be of type %s" % (type(worker_init),
                                                    WorkerInit))

        # replace enum with the string name to avoid
        # serialization issues with default Pyro4 serializers.
        worker_init = WorkerInit(type_=worker_init.type_.key,
                                 names=worker_init.names,
                                 data=worker_init.data)

        action_handle = self._action_manager.queue(
            server_name,
            action="SPPyroScenarioTreeServer_initialize",
            worker_type=worker_registered_name,
            worker_name=worker_name,
            worker_init=worker_init,
            options=worker_options,
            generate_response=not oneway)

        self._sppyro_server_workers_map[server_name].append(worker_name)
        self._sppyro_worker_server_map[worker_name] = server_name
        self._sppyro_worker_list.append(worker_name)

        return action_handle

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
        self._sppyro_worker_list.remove(worker_name)

    def get_server_for_worker(self, worker_name):
        try:
            return self._sppyro_worker_server_map[worker_name]
        except KeyError:
            raise KeyError(
                "Scenario tree worker with name %s does not exist on "
                "any scenario tree servers" % (worker_name))

    def _invoke_external_function_on_worker(
            self,
            worker_name,
            module_name,
            function_name,
            invocation_type=InvocationType.Single,
            function_args=None,
            function_kwds=None,
            oneway=False):
        invocation_type = _map_deprecated_invocation_type(invocation_type)
        return self._action_manager.queue(
            self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action="invoke_external_function",
            generate_response=not oneway,
            args=(module_name,
                  function_name),
            kwds={'invocation_type': invocation_type.key,
                  'function_args': function_args,
                  'function_kwds': function_kwds})

    def _invoke_method_on_worker(
            self,
            worker_name,
            method_name,
            method_args=None,
            method_kwds=None,
            oneway=False):

        return self._action_manager.queue(
            self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action=method_name,
            generate_response=not oneway,
            args=method_args if (method_args is not None) else (),
            kwds=method_kwds if (method_kwds is not None) else {})

#
# This class extends the initialization process of
# ScenarioTreeManagerSPPyroBasic so that scenario tree servers are
# automatically acquired and assigned worker instantiations that
# manage all scenarios / bundles.
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
                                "sppyro_multiple_server_workers")
    safe_register_common_option(_registered_options,
                                "sppyro_handshake_at_startup")

    default_registered_worker_name = 'ScenarioTreeWorkerBasic'

    def __init__(self, *args, **kwds):
        self._scenario_to_worker_map = {}
        self._bundle_to_worker_map = {}
        self._worker_registered_name = kwds.pop('registered_worker_name',
                                                self.default_registered_worker_name)
        super(ScenarioTreeManagerSPPyro, self).__init__(*args, **kwds)

    #
    # Override the PySPConfiguredObject register_options implementation so
    # that the default behavior will be to register this classes default
    # worker type options along with the options for this class
    #

    @classmethod
    def register_options(cls, *args, **kwds):
        """Cls.register_options(
              [options],
              registered_worker_name=Cls.default_registered_worker_name) -> options.
        Fills an options block will all registered options for this
        class. The optional argument 'options' can be a previously
        existing options block, which would be both updated and
        returned by this function.

        The optional flag 'registered_worker_name' can be used to
        control the worker type whose options will be additionaly
        registered with this classes options.  This flag can be set to
        None, implying that no additional worker options should be
        registered."""

        registered_worker_name = \
            kwds.pop('registered_worker_name',
                     ScenarioTreeManagerSPPyro.default_registered_worker_name)
        options = super(ScenarioTreeManagerSPPyro, cls).\
                  register_options(*args, **kwds)
        if registered_worker_name is not None:
            worker_type = SPPyroScenarioTreeServer.\
                          get_registered_worker_type(registered_worker_name)
            worker_type.register_options(options)
        return options

    def _initialize_scenariotree_workers(self):

        start_time = time.time()

        if self._options.verbose:
            print("Transmitting scenario tree worker initializations")

        if len(self._action_manager.server_pool) == 0:
            raise RuntimeError(
                "No scenario tree server processes have been acquired!")

        if self._scenario_tree.contains_bundles():
            jobs = [BundleWorkerInit(bundle.name,
                                     bundle.scenario_names)
                    for bundle in reversed(self._scenario_tree.bundles)]
        else:
            jobs = [ScenarioWorkerInit(scenario.name)
                    for scenario in reversed(self._scenario_tree.scenarios)]

        assert len(self._sppyro_server_workers_map) == \
            len(self._action_manager.server_pool)
        assert len(self._sppyro_worker_server_map) == 0
        assert len(self._sppyro_worker_list) == 0
        scenario_instance_factory = \
            self._scenario_tree._scenario_instance_factory

        worker_type = SPPyroScenarioTreeServer.\
                      get_registered_worker_type(self._worker_registered_name)
        worker_options = None
        try:
            worker_options = worker_type.\
                             extract_user_options_to_dict(self._options, sparse=True)
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
        # The first loop it just to get the counts
        tmp = defaultdict(int)
        cnt = 0
        for server_name in itertools.cycle(self._action_manager.server_pool):
            if len(jobs) == cnt:
                break
            tmp[server_name] += 1
            cnt += 1
        # We do this in two loops so the scenario / bundle assignment looks
        # contiguous by names listed on the scenario tree
        assert len(tmp) == len(self._action_manager.server_pool)
        for server_name in tmp:
            assert tmp[server_name] > 0
            for _i in xrange(tmp[server_name]):
                worker_initializations[server_name].append(jobs.pop())

        assert not self._transmission_paused
        if not self._options.sppyro_handshake_at_startup:
            self.pause_transmit()
        action_handle_data = {}
        for cntr, server_name in enumerate(worker_initializations):

            if self._options.sppyro_multiple_server_workers:

                #
                # Multiple workers per server
                #

                for worker_init in worker_initializations[server_name]:
                    assert type(worker_init.names) is tuple
                    assert len(worker_init.names) == 1
                    object_name = worker_init.names[0]
                    worker_name = server_name+":Worker_"+str(object_name)
                    action_handle = self.add_worker(
                        worker_name,
                        worker_init,
                        worker_options,
                        self._worker_registered_name,
                        server_name=server_name)

                    if self._options.sppyro_handshake_at_startup:
                        action_handle_data[worker_name] =  \
                            self.AsyncResult(
                                self, action_handle_data=action_handle).complete()
                    else:
                        action_handle_data[action_handle] = worker_name

                    if worker_init.type_ == WorkerInitType.Bundles:
                        assert self._scenario_tree.contains_bundle(object_name)
                        self._bundle_to_worker_map[object_name] = worker_name
                        assert type(worker_init.data) is dict
                        assert len(worker_init.data) == 1
                        assert len(worker_init.data[object_name]) > 0
                        for scenario_name in worker_init.data[object_name]:
                            self._scenario_to_worker_map[scenario_name] = worker_name
                    else:
                        assert worker_init.type_ == WorkerInitType.Scenarios
                        assert self._scenario_tree.contains_scenario(object_name)
                        self._scenario_to_worker_map[object_name] = worker_name

            else:

                #
                # One worker per server
                #

                init_type = worker_initializations[server_name][0].type_
                assert all(init_type == _worker_init.type_ for _worker_init
                           in worker_initializations[server_name])
                assert all(type(_worker_init.names) is tuple
                           for _worker_init in worker_initializations[server_name])
                assert all(len(_worker_init.names) == 1
                           for _worker_init in worker_initializations[server_name])
                worker_name = None
                if init_type == WorkerInitType.Bundles:
                    worker_name = server_name+":Worker_BundleGroup"+str(cntr)
                    worker_init = BundleWorkerInit(
                        [_worker_init.names[0] for _worker_init
                         in worker_initializations[server_name]],
                        dict((_worker_init.names[0],
                              _worker_init.data[_worker_init.names[0]])
                             for _worker_init in worker_initializations[server_name]))
                else:
                    assert init_type == WorkerInitType.Scenarios
                    worker_name = server_name+":Worker_ScenarioGroup"+str(cntr)
                    worker_init = ScenarioWorkerInit(
                        [_worker_init.names[0] for _worker_init
                         in worker_initializations[server_name]])

                action_handle = self.add_worker(
                    worker_name,
                    worker_init,
                    worker_options,
                    self._worker_registered_name,
                    server_name=server_name)

                if self._options.sppyro_handshake_at_startup:
                    action_handle_data[worker_name] =  \
                        self.AsyncResult(
                            self, action_handle_data=action_handle).complete()
                else:
                    action_handle_data[action_handle] = worker_name

                if worker_init.type_ == WorkerInitType.Bundles:
                    for bundle_name in worker_init.names:
                        assert self._scenario_tree.contains_bundle(bundle_name)
                        self._bundle_to_worker_map[bundle_name] = worker_name
                        for scenario_name in worker_init.data[bundle_name]:
                            assert self._scenario_tree.contains_scenario(scenario_name)
                            self._scenario_to_worker_map[scenario_name] = worker_name
                else:
                    assert worker_init.type_ == WorkerInitType.Scenarios
                    for scenario_name in worker_init.names:
                        assert self._scenario_tree.contains_scenario(scenario_name)
                        self._scenario_to_worker_map[scenario_name] = worker_name

        if not self._options.sppyro_handshake_at_startup:
            self.unpause_transmit()

        end_time = time.time()

        if self._options.output_times:
            print("Initialization transmission time=%.2f seconds"
                  % (end_time - start_time))

        if self._options.sppyro_handshake_at_startup:
            return self.AsyncResult(
                self, result=action_handle_data)
        else:
            return self.AsyncResult(
                self, action_handle_data=action_handle_data)

    #
    # Abstract methods for _ScenarioTreeManagerImpl:
    #

    #
    # Override abstract methods for _ScenarioTreeManagerImpl
    # that were implemented by ScenarioTreeManagerSPPyroBasic
    #
    def _init(self):
        assert self._scenario_tree is not None

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

        servers_required = self._options.sppyro_required_servers
        if servers_required == 0:
            servers_required = num_jobs
        elif servers_required > num_jobs:
            if servers_required > num_jobs:
                print("Value assigned to sppyro_required_servers option (%s) "
                      "is greater than the number of available jobs (%s). "
                      "Limiting the number of servers to acquire to %s"
                      % (servers_required, num_jobs, num_jobs))
            servers_required = num_jobs

        timeout = self._options.sppyro_find_servers_timeout if \
                  (self._options.sppyro_required_servers == 0) else \
                  None

        if self._options.verbose:
            if servers_required == 0:
                assert timeout is not None
                print("Using timeout of %s seconds to aquire up to "
                      "%s servers" % (timeout, num_jobs))
            else:
                print("Waiting to acquire exactly %s servers to distribute "
                      "work over %s jobs" % (servers_required, num_jobs))

        self.acquire_scenariotreeservers(servers_required, timeout=timeout)

        if self._options.verbose:
            print("Broadcasting requests to initialize workers "
                  "on scenario tree servers")

        initialization_handle = self._initialize_scenariotree_workers()

        worker_names = sorted(self._sppyro_worker_server_map)

        # run the user script to collect aggregate scenario data. This
        # can slow down initialization as syncronization across all
        # scenario tree servers is required following serial
        # execution
        if len(self._options.aggregategetter_callback_location):
            assert not self._transmission_paused
            for callback_module_key, callback_name in zip(self._aggregategetter_keys,
                                                          self._aggregategetter_names):
                if self._options.verbose:
                    print("Transmitting invocation of user defined aggregategetter "
                          "callback function defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))

                result = self.invoke_external_function(
                    self._callback_mapped_module_name[callback_module_key],
                    callback_name,
                    invocation_type=InvocationType.PerScenarioChained,
                    function_args=(self._aggregate_user_data,))
                self._aggregate_user_data = result[0]

            # Transmit aggregate state to scenario tree servers
            if self._options.verbose:
                print("Broadcasting final aggregate data "
                      "to scenario tree servers")

            self.invoke_method(
                "assign_data",
                method_args=("_aggregate_user_data", self._aggregate_user_data,),
                oneway=not self._options.sppyro_handshake_at_startup)

        # run the user script to initialize variable bounds
        if len(self._options.postinit_callback_location):

            for callback_module_key, callback_name in zip(self._postinit_keys,
                                                          self._postinit_names):
                if self._options.verbose:
                    print("Transmitting invocation of user defined postinit "
                          "callback function defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))

                # Transmit invocation to scenario tree workers
                self.invoke_external_function(
                    self._callback_mapped_module_name[callback_module_key],
                    callback_name,
                    invocation_type=InvocationType.PerScenario,
                    oneway=not self._options.sppyro_handshake_at_startup)

        return initialization_handle

    def worker_names(self):
        return self._sppyro_worker_list

    def get_worker_for_scenario(self, scenario_name):
        assert self._scenario_tree.contains_scenario(scenario_name)
        return self._scenario_to_worker_map[scenario_name]

    def get_worker_for_bundle(self, bundle_name):
        assert self._scenario_tree.contains_bundle(bundle_name)
        return self._bundle_to_worker_map[bundle_name]

    def invoke_external_function(
            self,
            module_name,
            function_name,
            invocation_type=InvocationType.Single,
            function_args=None,
            function_kwds=None,
            worker_names=None,
            oneway=False,
            async=False):
        """Invoke an external function all scenario tree workers,
        passing the scenario tree worker as the first argument, and
        the worker scenario tree as the second argument. The remaining
        arguments and how this function is invoked is controlled by
        the invocation_type keyword. By default, a single invocation
        takes place."""
        invocation_type = _map_deprecated_invocation_type(invocation_type)

        start_time = time.time()
        assert self._action_manager is not None
        invocation_type = _map_deprecated_invocation_type(invocation_type)
        if self._options.verbose:
            print("Transmitting external function invocation request "
                  "to scenario tree workers")

        if self._transmission_paused:
            if (not async) and (not oneway):
                raise ValueError(
                    "Unable to perform external function invocations. "
                    "Pyro transmissions are currently paused, but the "
                    "function invocation is not one-way and not asynchronous."
                    "This implies action handles be collected within "
                    "this method. Pyro transmissions must be un-paused in order "
                    "for this to take place.")

        if worker_names is None:
            worker_names = self._sppyro_worker_list

        action_handle_data = None
        map_result = None
        if (invocation_type == InvocationType.Single) or \
           (invocation_type == InvocationType.PerBundle) or \
           (invocation_type == InvocationType.PerScenario):

            was_paused = self.pause_transmit()
            action_handle_data = {}
            for worker_name in worker_names:
                action_handle_data[self._invoke_external_function_on_worker(
                    worker_name,
                    module_name,
                    function_name,
                    invocation_type=invocation_type,
                    function_args=function_args,
                    function_kwds=function_kwds,
                    oneway=oneway)] = worker_name

            if invocation_type != InvocationType.Single:
                map_result = lambda ah_to_result: \
                             dict((key, result[key])
                                  for result in itervalues(ah_to_result)
                                  for key in result)

            if not was_paused:
                self.unpause_transmit()

        elif (invocation_type == InvocationType.PerBundleChained) or \
             (invocation_type == InvocationType.PerScenarioChained):

            if self._transmission_paused:
                raise ValueError("Chained invocation type %s cannot be executed "
                                 "when Pyro transmission is paused"
                                 % (invocation_type.key))

            if (function_args is None) or \
               (type(function_args) is not tuple) or \
               (len(function_args) == 0):
                raise ValueError("Function invocation type %s must be executed "
                                 "with function_args keyword set to non-empty "
                                 "tuple type" % (invocation_type.key))

            result = function_args
            for worker_name in worker_names[:-1]:

                result = self.AsyncResult(
                    self,
                    action_handle_data=self._invoke_external_function_on_worker(
                        worker_name,
                        module_name,
                        function_name,
                        invocation_type=invocation_type,
                        function_args=result,
                        function_kwds=function_kwds,
                        oneway=False)).complete()

            action_handle_data = self._invoke_external_function_on_worker(
                worker_names[-1],
                module_name,
                function_name,
                invocation_type=invocation_type,
                function_args=result,
                function_kwds=function_kwds,
                oneway=oneway)

        else:
            raise ValueError("Unexpected function invocation type '%s'. "
                             "Expected one of %s"
                             % (invocation_type,
                                [str(v) for v in InvocationType._values]))

        if oneway:
            action_handle_data = None
            map_result = None

        result = self.AsyncResult(
            self,
            action_handle_data=action_handle_data,
            map_result=map_result)

        if not async:
            result = result.complete()

        end_time = time.time()

        if self._options.output_times:
            print("External function invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return result

    def invoke_method(
            self,
            method_name,
            method_args=None,
            method_kwds=None,
            worker_names=None,
            oneway=False,
            async=False):
        """Invoke a method on all scenario tree workers."""

        start_time = time.time()
        assert self._action_manager is not None

        if self._options.verbose:
            print("Transmitting method invocation request "
                  "to scenario tree workers")

        if self._transmission_paused:
            if (not async) and (not oneway):
                raise ValueError(
                    "Unable to perform method invocations. "
                    "Pyro transmissions are currently paused, but the "
                    "method invocation is not one-way and not asynchronous."
                    "This implies action handles be collected within "
                    "this method. Pyro transmissions must be un-paused in order "
                    "for this to take place.")

        if worker_names is None:
            worker_names = self._sppyro_worker_list

        was_paused = self.pause_transmit()

        action_handle_data = dict(
            (self._action_manager.queue(
                self.get_server_for_worker(worker_name),
                worker_name=worker_name,
                action=method_name,
                generate_response=not oneway,
                args=method_args if (method_args is not None) else (),
                kwds=method_kwds if (method_kwds is not None) else {}),
             worker_name) for worker_name in worker_names)

        if not was_paused:
            self.unpause_transmit()

        if oneway:
            action_handle_data = None

        result = self.AsyncResult(
            self,
            action_handle_data=action_handle_data)

        if not async:
            result = result.complete()

        end_time = time.time()

        if self._options.output_times:
            print("Method invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return result

    #
    # Extended Interface for SPPyro
    #

    def get_server_for_scenario(self, scenario_name):
        return self.get_server_for_worker(
            self.get_worker_for_scenario(scenario_name))

    def get_server_for_bundle(self, bundle_name):
        return self.get_server_for_worker(
            self.get_worker_for_bundle(bundle_name))
