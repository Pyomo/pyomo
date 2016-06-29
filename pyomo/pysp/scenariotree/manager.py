#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("InvocationType",
           "ScenarioTreeManagerClientSerial",
           "ScenarioTreeManagerClientPyro")

import sys
import time
import itertools
import inspect
import logging
import traceback
from collections import defaultdict

import pyutilib.misc
import pyutilib.enum
from pyutilib.pyro import shutdown_pyro_components
from pyomo.opt.parallel.manager import ActionHandle
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_declare_common_option,
                                    _domain_must_be_str,
                                    _domain_tuple_of_str)
from pyomo.pysp.util.misc import (load_external_module,
                                  _EnumValueWithData)
from pyomo.pysp.scenariotree.instance_factory import \
    ScenarioTreeInstanceFactory
from pyomo.pysp.scenariotree.action_manager_pyro \
    import ScenarioTreeActionManagerPyro
from pyomo.pysp.scenariotree.server_pyro \
    import ScenarioTreeServerPyro
from pyomo.pysp.scenariotree.server_pyro_utils \
    import (ScenarioWorkerInit,
            BundleWorkerInit,
            WorkerInit,
            WorkerInitType)
from pyomo.pysp.ef import create_ef_instance

from six import iteritems, itervalues, StringIO
from six.moves import xrange

try:
    from guppy import hpy
    guppy_available = True
except ImportError:
    guppy_available = False

logger = logging.getLogger('pyomo.pysp')

_invocation_type_enum_list = []
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 0, 'Single'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 1, 'PerScenario'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 2, 'PerScenarioChained'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 3, 'PerBundle'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 4, 'PerBundleChained'))

##### These values are DEPRECATED
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 5, 'SingleInvocation'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 6, 'PerScenarioInvocation'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 7, 'PerScenarioChainedInvocation'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 8, 'PerBundleInvocation'))
_invocation_type_enum_list.append(
    pyutilib.enum.EnumValue('InvocationType', 9, 'PerBundleChainedInvocation'))
#####

# These are enum values that carry data with them
_invocation_type_enum_list.append(
    _EnumValueWithData(_domain_must_be_str,
                       'InvocationType', 10, 'OnScenario'))
_invocation_type_enum_list.append(
    _EnumValueWithData(_domain_tuple_of_str,
                       'InvocationType', 11, 'OnScenarios'))
_invocation_type_enum_list.append(
    _EnumValueWithData(_domain_must_be_str,
                       'InvocationType', 12, 'OnBundle'))
_invocation_type_enum_list.append(
    _EnumValueWithData(_domain_tuple_of_str,
                       'InvocationType', 13, 'OnBundles'))
_invocation_type_enum_list.append(
    _EnumValueWithData(_domain_tuple_of_str,
                       'InvocationType', 14, 'OnScenariosChained'))
_invocation_type_enum_list.append(
    _EnumValueWithData(_domain_tuple_of_str,
                       'InvocationType', 15, 'OnBundlesChained'))

class _InvocationTypeDocumentedEnum(pyutilib.enum.Enum):
    """Controls execution of function invocations with a scenario tree manager.

    In all cases, the function must accept the process-local scenario
    tree worker as the first argument. Whether or not additional
    arguments are required, depends on the invocation type. For the
    'Single' invocation type, no additional arguments are required.
    Otherwise, the function signature is required to accept a second
    argument representing the worker-local scenario or scenario
    bundle object.

    It is implied that the function invocation takes place on the
    scenario tree worker(s), which is(are) not necessarily the same as
    the scenario tree manager whose method is provided with the
    invocation type. For instance, Pyro-based scenario tree managers
    (e.g., ScenarioTreeManagerClientPyro) must transmit these method
    invocations to their respective scenario tree workers which live
    in separate processes. Any scenario tree worker is itself an
    instance of a ScenarioTreeManager so the same invocation rules
    apply when using this interface in a worker-local context. The
    ScenarioTreeManagerClientSerial implementation is its own scenario
    tree worker, so all function invocations take place locally and on
    the same object whose method is invoked.

    If the worker name is not provided (e.g., when the
    'invoke_function' method is used), then the following behavior is
    implied for each invocation type:

       - Single:
            The function is executed once per scenario tree
            worker. Return value will be in the form of a dict mapping
            worker name to function return value.

       - PerScenario:
            The function is executed once per scenario in the scenario
            tree. Return value will be in the form of a dict mapping
            scenario name to return value.

       - PerScenarioChained:
            The function is executed once per scenario in the scenario
            tree in a sequential call chain. The result from each
            function call is passed into the next function call in
            *arg form after the scenario tree worker and scenario
            arguments (unless no additional function arguments were
            initially provided).  Return value is in the form of a
            tuple (or None), and represents the return value from the
            final call in the chain.

       - PerBundle:
            Identical to the PerScenario invocation type except by
            bundle.

       - PerBundleChained:
            Identical to the PerScenarioChained invocation type except
            by bundle.

     * NOTE: The remaining invocation types listed below should
             initialized with any relevant data before being passed
             into methods that use them. This is done using the
             __call__ method, which returns a matching invocation type
             loaded with the provided data.

             Examples:
                  InvocationType.OnScenario('Scenario1')
                  InvocationType.OnScenarios(['Scenario1', 'Scenario2'])

       - OnScenario(<scenario-name>):
            The function is executed on the named scenario and its
            associated scenario tree worker. Return value corresponds
            exactly to the function return value.

       - OnScenarios([<scenario-names>]):
            The function is executed on the named scenarios and their
            associated scenario tree worker(s). Return value will be
            in the form of a dict mapping scenario name to return
            value.

       - OnScenariosChained([<scenario-names>]):
            Same as PerScenarioChained only executed over the given
            subset of scenarios named. Invocation order is guaranteed
            to correspond exactly to the iteration order of the given
            scenario names.

       - OnBundle(<bundle-name>):
            Identical to the OnScenario invocation type except with a
            bundle.

       - OnBundles([<bundle-names>]):
            Identical to the OnScenarios invocation type except by
            bundle.

       - OnBundlesChained([<bundle-names>]):
            Identical to the OnScenariosChained invocation type except
            by bundle.

    If the scenario tree worker name is provided (e.g., when the
    'invoke_function_on_worker' method is used), then the following
    behaviors change:

       - Single:
            The return value corresponds exactly to the function
            return value (rather than a dict mapping worker_name to
            return value).

       - Per*:
            Function execution takes place only over the scenarios /
            bundles managed by the named scenario tree worker.

       - On*:
            Not necessarily designed for this context, but execution
            behavior remains the same. An exception will be raised if
            the named scenario(s) / bundles(s) are not directly
            managed by the named scenario tree worker.

    """

InvocationType = _InvocationTypeDocumentedEnum(*_invocation_type_enum_list)

_deprecated_invocation_types = \
    {InvocationType.SingleInvocation: InvocationType.Single,
     InvocationType.PerScenarioInvocation: InvocationType.PerScenario,
     InvocationType.PerScenarioChainedInvocation: InvocationType.PerScenarioChained,
     InvocationType.PerBundleInvocation: InvocationType.PerBundle,
     InvocationType.PerBundleChainedInvocation: InvocationType.PerBundleChained}
def _map_deprecated_invocation_type(invocation_type):
    if invocation_type in _deprecated_invocation_types:
        logger.warning("DEPRECATED: %s has been renamed to %s"
                       % (invocation_type, _deprecated_invocation_types[invocation_type]))
        invocation_type = _deprecated_invocation_types[invocation_type]
    return invocation_type

#
# A base class and interface that is common to all scenario tree
# client and worker managers.
#

class ScenarioTreeManager(PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManager class")

    #
    # Note: These Async objects can be cleaned up.
    #       This is a first draft.
    #
    class Async(object):
        def complete(self):
            raise NotImplementedError(type(self).__name__+": This method is abstract")

    class AsyncResult(Async):

        __slots__ = ('_action_manager',
                     '_result',
                     '_action_handle_data',
                     '_invocation_type',
                     '_map_result')

        def __init__(self,
                     action_manager,
                     result=None,
                     action_handle_data=None,
                     map_result=None):
            if result is not None:
                assert action_handle_data is None
            if action_handle_data is not None:
                assert action_manager is not None
            if map_result is not None:
                assert result is None
                assert action_handle_data is not None
            self._action_manager = action_manager
            self._action_handle_data = action_handle_data
            self._result = result
            self._map_result = map_result

        def complete(self):

            if self._result is not None:
                if isinstance(self._result,
                              ScenarioTreeManager.Async):
                    self._result = self._result.complete()
                return self._result

            if self._action_handle_data is None:
                assert self._result is None
                return None

            result = None
            if isinstance(self._action_handle_data, ActionHandle):
                result = self._action_manager.wait_for(
                    self._action_handle_data)
                if self._map_result is not None:
                    result = self._map_result(self._action_handle_data, result)
            else:
                ah_to_result = {}
                ahs = set(self._action_handle_data)
                while len(ahs) > 0:
                    ah = self._action_manager.wait_any(ahs)
                    ah_to_result[ah] = self._action_manager.get_results(ah)
                    ahs.remove(ah)
                #self._action_manager.wait_all(self._action_handle_data)
                #ah_to_result = dict((ah, self._action_manager.get_results(ah))
                #                    for ah in self._action_handle_data)
                if self._map_result is not None:
                    result = self._map_result(ah_to_result)
                else:
                    result = dict((self._action_handle_data[ah], ah_to_result[ah])
                                  for ah in ah_to_result)
            self._result = result
            return self._result

    # This class ensures that a chain of asynchronous
    # actions are completed in order
    class AsyncResultChain(Async):
        __slots__ = ("_results", "_return_index")

        def __init__(self, results, return_index=-1):
            self._results = results
            self._return_index = return_index

        def complete(self):
            for i in xrange(len(self._results)):
                assert isinstance(self._results[i],
                                  ScenarioTreeManager.Async)
                self._results[i] = self._results[i].complete()
            if self._return_index is not None:
                return self._results[self._return_index]
            return None

    # This class returns the result of a callback function
    # when completing an asynchronous action
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
        if self.__class__ is ScenarioTreeManager:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)

        super(ScenarioTreeManager, self).__init__(*args, **kwds)

        init_start_time = time.time()
        self._error_shutdown = False
        self._scenario_tree = None
        # bundle info
        self._scenario_to_bundle_map = {}
        # For the users to modify as they please in the aggregate
        # callback as long as the data placed on it can be serialized
        # by Pyro
        self._aggregate_user_data = {}
        # set to true with the __enter__ method is called
        self._inside_with_block = False
        self._initialized = False

    def _add_bundle(self, bundle_name, scenario_list):

        for scenario_name in scenario_list:

            if scenario_name in self._scenario_to_bundle_map:
                raise ValueError(
                    "Unable to form binding instance for bundle %s. "
                    "Scenario %s already belongs to bundle %s."
                    % (bundle_name,
                       scenario_name,
                       self._scenario_to_bundle_map[scenario_name]))

            self._scenario_to_bundle_map[scenario_name] = bundle_name

    #
    # Interface:
    #

    @property
    def scenario_tree(self):
        return self._scenario_tree

    @property
    def initialized(self):
        return self._initialized

    def initialize(self, *args, **kwds):
        """Initialize the scenario tree manager.

        A scenario tree manager must be initialized before using it.
        """

        init_start_time = time.time()
        result = None
        try:
            if self._options.verbose:
                print("Initializing %s with options:"
                      % (type(self).__name__))
                self.display_options()
                print("")
            ############# derived method
            result = self._init(*args, **kwds)
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

        if self._options.output_times or \
           self._options.verbose:
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

        self._initialized = True

        return result

    def __enter__(self):
        self._inside_with_block = True
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            sys.stderr.write("Exception encountered. Scenario tree manager "
                             "attempting to shut down.\n")
            tmp = StringIO()
            _args = list(args) + [None, tmp]
            traceback.print_exception(*_args)
            self._error_shutdown = True
            try:
                self.close()
            except:
                logger.error("Exception encountered during emergency scenario "
                             "tree manager shutdown. Printing original exception "
                             "here:\n")
                logger.error(tmp.getvalue())
                raise
        else:
            self.close()

    def close(self):
        """Close the scenario tree manager and any associated objects."""
        if self._options.verbose:
            print("Closing "+str(self.__class__.__name__))
        self._close_impl()
        if hasattr(self._scenario_tree, "_scenario_instance_factory"):
            self._scenario_tree._scenario_instance_factory.close()
        self._scenario_tree = None
        self._scenario_to_bundle_map = {}
        self._aggregate_user_data = {}

    def add_bundle(self, bundle_name, scenario_list):
        """Add a scenario bundle to this scenario tree manager and the
        scenario tree that it manages."""
        if self._options.verbose:
            print("Adding scenario bundle with name %s"
                  % (bundle_name))

        if self._scenario_tree.contains_bundle(bundle_name):
            raise ValueError(
                "Unable to create bundle with name %s. A bundle "
                "with that name already exists on the scenario tree"
                % (bundle_name))

        self._scenario_tree.add_bundle(bundle_name, scenario_list)
        self._add_bundle(bundle_name, scenario_list)
        self._add_bundle_impl(bundle_name, scenario_list)

    def remove_bundle(self, bundle_name):
        """Remove a bundle from this scenario tree manager and the
        scenario tree that it manages."""
        if self._options.verbose:
            print("Removing scenario bundle with name %s"
                  % (bundle_name))

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

    def invoke_function(self,
                        function_name,
                        module_name,
                        invocation_type=InvocationType.Single,
                        function_args=(),
                        function_kwds=None,
                        async=False,
                        oneway=False):
        """Invokes a function on scenario tree constructs managed by
           this scenario tree manager. The function must always accept
           at least one argument, which is the process-local scenario
           tree worker object (may or may not be this object).

        Args:
            function_name:
                 The name of the function to be invoked.
            module_name:
                 The name / location of the module containing the
                 function.
            invocation_type:
                 Controls how the function is invoked. Refer to the
                 doc string for pyomo.pysp.scenariotree.manager.InvocationType
                 for more information.
            function_args:
                 Extra arguments passed to the function when it is
                 invoked. These will always be placed after the
                 initial process-local scenario tree worker object as
                 well as any additional arguments governed by the
                 invocation type.
            function_kwds:
                 Additional keywords to pass to the function when it
                 is invoked.
            async:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.
            oneway:
                 When set to True, it will be assumed no return value
                 is expected from this function (async is
                 implied). Setting both async and oneway to True will
                 result in an exception being raised.

            *Note: The 'oneway' and 'async' keywords are valid for all
                   scenario tree manager implementations. However,
                   they are designed for use with Pyro-based
                   implementations. Their existence in other
                   implementations is not meant to guarantee
                   asynchronicity, but rather to provide a consistent
                   interface for code to be written around.

        Returns:
            If 'oneway' is True, this function will always return
            None. Otherwise, the return value type is governed by the
            'invocation_type' keyword, which will be nested inside an
            asynchronous object if 'async' is set to True.
        """
        if async and oneway:
            raise ValueError("async oneway calls do not make sense")
        invocation_type = _map_deprecated_invocation_type(invocation_type)
        return self._invoke_function_impl(function_name,
                                          module_name,
                                          invocation_type=invocation_type,
                                          function_args=function_args,
                                          function_kwds=function_kwds,
                                          async=async,
                                          oneway=oneway)

    def invoke_method(self,
                      method_name,
                      method_args=(),
                      method_kwds=None,
                      async=False,
                      oneway=False):
        """Invokes a method on a scenario tree constructs managed
           by this scenario tree manager client. This may or may not
           take place on this client itself.

        Args:
            method_name:
                 The name of the method to be invoked.
            method_args:
                 Arguments passed to the method when it is invoked.
            method_kwds:
                 Keywords to pass to the method when it is invoked.
            async:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.
            oneway:
                 When set to True, it will be assumed no return value
                 is expected from this method (async is
                 implied). Setting both async and oneway to True will
                 result in an exception being raised.

            *Note: The 'oneway' and 'async' keywords are valid for all
                   scenario tree manager client
                   implementations. However, they are designed for use
                   with Pyro-based implementations. Their existence in
                   other implementations is not meant to guarantee
                   asynchronicity, but rather to provide a consistent
                   interface for code to be written around.

        Returns:
            If 'oneway' is True, this function will always return
            None. Otherwise, the return corresponds exactly to the
            method's return value, which will be nested inside an
            asynchronous object if 'async' is set to True.
        """
        if async and oneway:
            raise ValueError("async oneway calls do not make sense")
        return self._invoke_method_impl(method_name,
                                        method_args=method_args,
                                        method_kwds=method_kwds,
                                        async=async,
                                        oneway=oneway)


    #
    # Methods defined by derived class that are not
    # part of the user interface
    #

    def _init(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _close_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _invoke_function_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _invoke_method_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _add_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _remove_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

#
# A base class and interface that is common to client-side scenario
# tree manager implementations (e.g, both the Pyro and Serial
# versions).
#

class ScenarioTreeManagerClient(ScenarioTreeManager,
                                PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerClient class")

    #
    # scenario instance construction
    #
    safe_declare_common_option(_declared_options,
                               "model_location")
    safe_declare_common_option(_declared_options,
                               "scenario_tree_location")
    safe_declare_common_option(_declared_options,
                               "objective_sense_stage_based")
    safe_declare_common_option(_declared_options,
                               "postinit_callback_location")
    safe_declare_common_option(_declared_options,
                               "aggregategetter_callback_location")

    #
    # scenario tree generation
    #
    safe_declare_common_option(_declared_options,
                               "scenario_tree_random_seed")
    safe_declare_common_option(_declared_options,
                               "scenario_tree_downsample_fraction")
    safe_declare_common_option(_declared_options,
                               "scenario_bundle_specification")
    safe_declare_common_option(_declared_options,
                               "create_random_bundles")

    #
    # various
    #
    safe_declare_common_option(_declared_options,
                               "output_times")
    safe_declare_common_option(_declared_options,
                               "verbose")
    safe_declare_common_option(_declared_options,
                               "profile_memory")

    def __init__(self, *args, **kwds):
        if self.__class__ is ScenarioTreeManagerClient:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)
        super(ScenarioTreeManagerClient, self).__init__(*args, **kwds)

        # callback info
        self._scenario_tree = None
        self._callback_function = {}
        self._callback_mapped_module_name = {}
        self._aggregategetter_keys = []
        self._aggregategetter_names = []
        self._postinit_keys = []
        self._postinit_names = []
        self._modules_imported = {}
        self._generate_scenario_tree()
        self._import_callbacks()

    def _generate_scenario_tree(self):

        start_time = time.time()
        if self._options.verbose:
            print("Importing model and scenario tree files")

        scenario_instance_factory = \
            ScenarioTreeInstanceFactory(
                self._options.model_location,
                self._options.scenario_tree_location)

        #
        # Try to prevent unnecessarily re-importing the model module
        # if other callbacks are in the same location. Doing so might
        # have serious consequences.
        #
        if scenario_instance_factory._model_module is not None:
            self._modules_imported[scenario_instance_factory.\
                                   _model_filename] = \
                scenario_instance_factory._model_module
        if scenario_instance_factory._scenario_tree_module is not None:
            self._modules_imported[scenario_instance_factory.\
                                   _scenario_tree_filename] = \
                scenario_instance_factory._scenario_tree_module

        if self._options.output_times or \
           self._options.verbose:
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
                    random_seed=self._options.scenario_tree_random_seed,
                    verbose=self._options.verbose)

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
            scenario_instance_factory.close()
            raise

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
                if module_name in self._modules_imported:
                    module = self._modules_imported[module_name]
                    sys_modules_key = module_name
                else:
                    module, sys_modules_key = \
                        load_external_module(module_name, clear_cache=True)
                    self._modules_imported[module_name] = module
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

    # override initialize on ScenarioTreeManager for documentation purposes
    def initialize(self, async=False):
        """Initialize the scenario tree manager client.

        Args:
            async:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.

        Returns:
            A dictionary mapping scenario tree worker names to their
            initial return value (True is most cases). If 'async' is
            set to True, this return value will be nested inside an
            asynchronous object.

        *NOTE: Calling complete() on an asynchronous object returned
               from this method may causes changes in the object whose
               method is called. One should avoid using the client
               until initialization is complete.
        """
        return super(ScenarioTreeManagerClient, self).initialize(async=async)

    def invoke_function_on_worker(self,
                                  worker_name,
                                  function_name,
                                  module_name,
                                  invocation_type=InvocationType.Single,
                                  function_args=(),
                                  function_kwds=None,
                                  async=False,
                                  oneway=False):
        """Invokes a function on a scenario tree worker managed
           by this scenario tree manager client. The function must
           always accept at least one argument, which is the
           process-local scenario tree worker object (may or may not
           be this object).

        Args:
            worker_name:
                 The name of the scenario tree worker. The list of worker
                 names can be found at client.worker_names.
            function_name:
                 The name of the function to be invoked.
            module_name:
                 The name / location of the module containing the
                 function.
            invocation_type:
                 Controls how the function is invoked. Refer to the
                 doc string for pyomo.pysp.scenariotree.manager.InvocationType
                 for more information.
            function_args:
                 Extra arguments passed to the function when it is
                 invoked. These will always be placed after the
                 initial process-local scenario tree worker object as
                 well as any additional arguments governed by the
                 invocation type.
            function_kwds:
                 Additional keywords to pass to the function when it
                 is invoked.
            async:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.
            oneway:
                 When set to True, it will be assumed no return value
                 is expected from this function (async is
                 implied). Setting both async and oneway to True will
                 result in an exception being raised.

            *Note: The 'oneway' and 'async' keywords are valid for all
                   scenario tree manager implementations. However,
                   they are designed for use with Pyro-based
                   implementations. Their existence in other
                   implementations is not meant to guarantee
                   asynchronicity, but rather to provide a consistent
                   interface for code to be written around.

        Returns:
            If 'oneway' is True, this function will always return
            None. Otherwise, the return value type is governed by the
            'invocation_type' keyword, which will be nested inside an
            asynchronous object if 'async' is set to True.
        """
        if async and oneway:
            raise ValueError("async oneway calls do not make sense")
        invocation_type = _map_deprecated_invocation_type(invocation_type)
        return self._invoke_function_on_worker_impl(worker_name,
                                                    function_name,
                                                    module_name,
                                                    invocation_type=invocation_type,
                                                    function_args=function_args,
                                                    function_kwds=function_kwds,
                                                    async=async,
                                                    oneway=oneway)

    def invoke_method_on_worker(self,
                                worker_name,
                                method_name,
                                method_args=(),
                                method_kwds=None,
                                async=False,
                                oneway=False):
        """Invokes a method on a scenario tree worker managed
           by this scenario tree manager client. The worker
           may or may not be this client.

        Args:
            worker_name:
                 The name of the scenario tree worker. The list of worker
                 names can be found at client.worker_names.
            method_name:
                 The name of the worker method to be invoked.
            method_args:
                 Arguments passed to the method when it is invoked.
            method_kwds:
                 Keywords to pass to the method when it is invoked.
            async:
                 When set to True, the return value will be an
                 asynchronous object. Invocation results can be
                 obtained at any point by calling the complete()
                 method on this object, which will block until all
                 associated action handles are collected.
            oneway:
                 When set to True, it will be assumed no return value
                 is expected from this method (async is
                 implied). Setting both async and oneway to True will
                 result in an exception being raised.

            *Note: The 'oneway' and 'async' keywords are valid for all
                   scenario tree manager client
                   implementations. However, they are designed for use
                   with Pyro-based implementations. Their existence in
                   other implementations is not meant to guarantee
                   asynchronicity, but rather to provide a consistent
                   interface for code to be written around.

        Returns:
            If 'oneway' is True, this function will always return
            None. Otherwise, the return corresponds exactly to the
            method's return value, which will be nested inside an
            asynchronous object if 'async' is set to True.
        """
        if async and oneway:
            raise ValueError("async oneway calls do not make sense")
        return self._invoke_method_on_worker_impl(worker_name,
                                                  method_name,
                                                  method_args=method_args,
                                                  method_kwds=method_kwds,
                                                  async=async,
                                                  oneway=oneway)

    @property
    def worker_names(self):
        """The list of worker names managed by this client."""
        return self._worker_names_impl()

    def get_worker_for_scenario(self, scenario_name):
        """Get the worker name assigned to the scenario with the given name."""
        if not self._scenario_tree.contains_scenario(scenario_name):
            raise KeyError("Scenario with name %s does not exist "
                           "in the scenario tree" % (scenario_name))
        return self._get_worker_for_scenario_impl(scenario_name)

    def get_worker_for_bundle(self, bundle_name):
        """Get the worker name assigned to the bundle with the given name."""
        if not self._scenario_tree.contains_bundle(bundle_name):
            raise KeyError("Bundle with name %s does not exist "
                           "in the scenario tree" % (bundle_name))
        return self._get_worker_for_bundle_impl(bundle_name)

    def get_scenarios_for_worker(self, worker_name):
        """Get the list of scenario names assigned to the worker with
        the given name."""
        if worker_name not in self.worker_names:
            raise KeyError("Worker with name %s does not exist under "
                           "in this client" % (worker_name))
        return self._get_scenarios_for_worker_impl(worker_name)

    def get_bundles_for_worker(self, worker_name):
        """Get the list of bundle names assigned to the worker with
        the given name."""
        if worker_name not in self.worker_names:
            raise KeyError("Worker with name %s does not exist under "
                           "in this client" % (worker_name))
        return self._get_bundles_for_worker_impl(worker_name)

    #
    # Partially implement _init for ScenarioTreeManager
    # subclasses are now expected to generate a (possibly
    # dummy) async object during _init_client
    #
    def _init(self, async=False):
        async_handle = self._init_client()
        if async:
            result = async_handle
        else:
            result = async_handle.complete()
        return result

    #
    # Methods defined by derived class that are not
    # part of the user interface
    #

    def _init_client(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _invoke_function_on_worker_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _invoke_method_on_worker_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _worker_names_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_worker_for_scenario_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_worker_for_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_scenarios_for_worker_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_bundles_for_worker_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

#
# A partial implementation of the ScenarioTreeManager
# interface that is common to both the Serial scenario
# tree manager as well as the Pyro workers used by the
# Pyro scenario tree manager.
#

class _ScenarioTreeManagerWorker(PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "_ScenarioTreeManagerWorker class")

    #
    # various
    #
    safe_declare_common_option(_declared_options,
                               "output_times")
    safe_declare_common_option(_declared_options,
                               "verbose")

    def __init__(self, *args, **kwds):
        if self.__class__ is _ScenarioTreeManagerWorker:
            raise NotImplementedError(
                "%s is an abstract class for subclassing" % self.__class__)
        super(_ScenarioTreeManagerWorker, self).__init__(*args, **kwds)

        # scenario instance models
        self._instances = None
        # bundle instance models
        self._bundle_binding_instance_map = {}
        self._modules_imported = {}

    def _invoke_function_by_worker(self,
                                   function_name,
                                   module_name,
                                   invocation_type=InvocationType.Single,
                                   function_args=(),
                                   function_kwds=None):

        if function_kwds is None:
            function_kwds = {}

        if module_name in self._modules_imported:
            this_module = self._modules_imported[module_name]
        elif module_name in sys.modules:
            this_module = sys.modules[module_name]
        else:
            this_module = pyutilib.misc.import_file(module_name,
                                                    clear_cache=True)
            self._modules_imported[module_name] = this_module
            self._modules_imported[this_module.__file__] = this_module
            if this_module.__file__.endswith(".pyc"):
                self._modules_imported[this_module.__file__[:-1]] = \
                    this_module

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
        elif (invocation_type == InvocationType.PerScenario) or \
             (invocation_type == InvocationType.PerScenarioChained):
            call_objects = self._scenario_tree.scenarios
        elif (invocation_type == InvocationType.OnScenario):
            call_objects = [self._scenario_tree.get_scenario(invocation_type.data)]
        elif (invocation_type == InvocationType.OnScenarios) or \
             (invocation_type == InvocationType.OnScenariosChained):
            assert len(invocation_type.data) != 0
            call_objects = [self._scenario_tree.get_scenario(scenario_name)
                            for scenario_name in invocation_type.data]
        elif (invocation_type == InvocationType.PerBundle) or \
             (invocation_type == InvocationType.PerBundleChained):
            if not self._scenario_tree.contains_bundles():
                raise ValueError(
                    "Received request for bundle invocation type "
                    "but the scenario tree does not contain bundles.")
            call_objects = self._scenario_tree.bundles
        elif (invocation_type == InvocationType.OnBundle):
            call_objects = [self._scenario_tree.get_bundle(invocation_type.data)]
        elif (invocation_type == InvocationType.OnBundles) or \
             (invocation_type == InvocationType.OnBundlesChained):
            if not self._scenario_tree.contains_bundles():
                raise ValueError(
                    "Received request for bundle invocation type "
                    "but the scenario tree does not contain bundles.")
            assert len(invocation_type.data) != 0
            call_objects = [self._scenario_tree.get_bundle(bundle_name)
                            for bundle_name in invocation_type.data]
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

        result = None
        if (invocation_type == InvocationType.Single):

            result = function(self,
                              *function_args,
                              **function_kwds)

        elif (invocation_type == InvocationType.OnScenario) or \
             (invocation_type == InvocationType.OnBundle):

            assert len(call_objects) == 1
            result = function(self,
                              call_objects[0],
                              *function_args,
                              **function_kwds)

        elif (invocation_type == InvocationType.PerScenarioChained) or \
             (invocation_type == InvocationType.OnScenariosChained) or \
             (invocation_type == InvocationType.PerBundleChained) or \
             (invocation_type == InvocationType.OnBundlesChained):

            if len(function_args) > 0:
                result = function_args
                for call_object in call_objects:
                    result = function(self,
                                      call_object,
                                      *result,
                                      **function_kwds)
            else:
                result = None
                for call_object in call_objects:
                    result = function(self,
                                      call_object,
                                      **function_kwds)
        else:

            result = dict((call_object.name, function(self,
                                                       call_object,
                                                       *function_args,
                                                       **function_kwds))
                          for call_object in call_objects)

        return result

    #
    # Abstract methods for ScenarioTreeManager:
    #

    def _close_impl(self):
        # copy the list of bundle names as the next loop will modify
        # the scenario_tree._scenario_bundles list
        if self._scenario_tree is not None:
            bundle_names = \
                [bundle.name for bundle in self._scenario_tree._scenario_bundles]
            for bundle_name in bundle_names:
                self.remove_bundle(bundle_name)
            assert not self._scenario_tree.contains_bundles()
        self._instances = None
        self._bundle_binding_instance_map = None

    def _invoke_function_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _invoke_method_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _add_bundle_impl(self, bundle_name, scenario_list):

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
            ef_instance_name=bundle.name,
            verbose_output=self._options.verbose)

        self._bundle_binding_instance_map[bundle.name] = \
            bundle_ef_instance

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Time construct binding instance for scenario bundle "
                  "%s=%.2f seconds" % (bundle_name, end_time - start_time))

    def _remove_bundle_impl(self, bundle_name):

        assert self._scenario_tree.contains_bundle(bundle_name)
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
# The Serial scenario tree manager class. This is a full
# implementation of the ScenarioTreeManager, ScenarioTreeManagerClient
# and _ScenarioTreeManagerWorker interfaces
#

class ScenarioTreeManagerClientSerial(_ScenarioTreeManagerWorker,
                                      ScenarioTreeManagerClient,
                                      PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerClientSerial class")

    #
    # scenario instance construction
    #
    safe_declare_common_option(_declared_options,
                               "output_instance_construction_time")
    safe_declare_common_option(_declared_options,
                               "compile_scenario_instances")

    def __init__(self, *args, **kwds):
        self._worker_name = 'ScenarioTreeManagerClientSerial:MainWorker'
        # good to have to keep deterministic ordering in code
        # rather than loop over the keys of the map on the
        # scenario tree
        self._scenario_names = []
        self._bundle_names = []
        super(ScenarioTreeManagerClientSerial, self).__init__(*args, **kwds)

    #
    # Abstract methods for ScenarioTreeManagerClient:
    #

    def _init_client(self):
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
                compile_scenario_instances=self._options.compile_scenario_instances,
                verbose=self._options.verbose)

        if self._options.output_times or \
           self._options.verbose:
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
        self._scenario_names = [_scenario.name for _scenario in
                                self._scenario_tree._scenarios]
        if self._options.output_times or \
           self._options.verbose:
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
                self._add_bundle(bundle.name, bundle._scenario_names)
                self._add_bundle_impl(bundle.name, bundle._scenario_names)

            end_time = time.time()
            if self._options.output_times or \
               self._options.verbose:
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
                        scenario)

        return self.AsyncResult(
            None, result={self._worker_name: True})

    def _invoke_function_on_worker_impl(self,
                                        worker_name,
                                        function_name,
                                        module_name,
                                        invocation_type=InvocationType.Single,
                                        function_args=(),
                                        function_kwds=None,
                                        async=False,
                                        oneway=False):

        assert worker_name == self._worker_name
        start_time = time.time()

        if self._options.verbose:
            print("Invoking function=%s in module=%s "
                  "on worker=%s"
                  % (function_name, module_name, worker_name))

        result = self._invoke_function_by_worker(function_name,
                                                 module_name,
                                                 invocation_type=invocation_type,
                                                 function_args=function_args,
                                                 function_kwds=function_kwds)

        if oneway:
            result = None
        if async:
            result = self.AsyncResult(None, result=result)

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _invoke_method_on_worker_impl(self,
                                      worker_name,
                                      method_name,
                                      method_args=(),
                                      method_kwds=None,
                                      async=False,
                                      oneway=False):

        assert worker_name == self._worker_name
        start_time = time.time()

        if self._options.verbose:
            print("Invoking method=%s on worker=%s"
                  % (method_name, self._worker_name))

        if method_kwds is None:
            method_kwds = {}
        result = getattr(self, method_name)(*method_args, **method_kwds)

        if oneway:
            result = None
        if async:
            result = self.AsyncResult(None, result=result)

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _worker_names_impl(self):
        return (self._worker_name,)

    def _get_worker_for_scenario_impl(self, scenario_name):
        assert self._scenario_tree.contains_scenario(scenario_name)
        return self._worker_name

    def _get_worker_for_bundle_impl(self, bundle_name):
        assert self._scenario_tree.contains_bundle(bundle_name)
        return self._worker_name

    def _get_scenarios_for_worker_impl(self, worker_name):
        assert worker_name == self._worker_name
        return self._scenario_names

    def _get_bundles_for_worker_impl(self, worker_name):
        assert worker_name == self._worker_name
        return self._bundle_names

    #
    # Abstract methods for ScenarioTreeManager:
    #

    # implemented by _ScenarioTreeManagerWorker
    #def _close_impl(...)

    def _invoke_function_impl(self,
                              function_name,
                              module_name,
                              invocation_type=InvocationType.Single,
                              function_args=(),
                              function_kwds=None,
                              async=False,
                              oneway=False):
        assert not (async and oneway)

        result = self._invoke_function_on_worker_impl(
            self._worker_name,
            function_name,
            module_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds,
            async=False,
            oneway=oneway)

        if not oneway:
            if invocation_type == InvocationType.Single:
                result = {self._worker_name: result}
        if async:
            result = self.AsyncResult(None, result=result)

        return result

    def _invoke_method_impl(self,
                            method_name,
                            method_args=(),
                            method_kwds=None,
                            async=False,
                            oneway=False):
        assert not (async and oneway)

        result =  self._invoke_method_on_worker_impl(
            self._worker_name,
            method_name,
            method_args=method_args,
            method_kwds=method_kwds,
            async=False,
            oneway=oneway)

        if not oneway:
            result = {self._worker_name: result}
        if async:
            result = self.AsyncResult(None, result=result)

        return result

    # override what is implemented by _ScenarioTreeManagerWorker
    def _add_bundle_impl(self, bundle_name, scenario_list):
        super(ScenarioTreeManagerClientSerial, self).\
            _add_bundle_impl(bundle_name, scenario_list)
        assert bundle_name not in self._bundle_names
        self._bundle_names.append(bundle_name)

    # override what is implemented by _ScenarioTreeManagerWorker
    def _remove_bundle_impl(self, bundle_name):
        super(ScenarioTreeManagerClientSerial, self).\
            _remove_bundle_impl(bundle_name)
        assert bundle_name in self._bundle_names
        self._bundle_names.remove(bundle_name)

#
# A partial implementation of the ScenarioTreeManager and
# ScenarioTreeManagerClient interfaces for Pyro that may serve some
# future purpose where there is not a one-to-one mapping between
# worker objects and scenarios / bundles in the scenario tree.
#

class _ScenarioTreeManagerClientPyroAdvanced(ScenarioTreeManagerClient,
                                             PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "_ScenarioTreeManagerClientPyroAdvanced class")

    safe_declare_common_option(_declared_options,
                               "pyro_host")
    safe_declare_common_option(_declared_options,
                               "pyro_port")
    safe_declare_common_option(_declared_options,
                               "pyro_shutdown")
    safe_declare_common_option(_declared_options,
                               "pyro_shutdown_workers")
    ScenarioTreeServerPyro.register_options(_declared_options)

    def __init__(self, *args, **kwds):
        # distributed worker information
        self._pyro_server_workers_map = {}
        self._pyro_worker_server_map = {}
        # the same as the .keys() of the above map
        # but won't suffer from stochastic iteration
        # order python dictionaries
        self._pyro_worker_list = []
        self._pyro_worker_scenarios_map = {}
        self._pyro_worker_bundles_map = {}
        self._action_manager = None
        self._transmission_paused = False
        super(_ScenarioTreeManagerClientPyroAdvanced, self).__init__(*args, **kwds)

    def _invoke_function_on_worker_pyro(self,
                                        worker_name,
                                        function_name,
                                        module_name,
                                        invocation_type=InvocationType.Single,
                                        function_args=(),
                                        function_kwds=None,
                                        oneway=False):

        return self._action_manager.queue(
            queue_name=self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action="_invoke_function_impl",
            generate_response=not oneway,
            args=(function_name,
                  module_name),
            kwds={'invocation_type': (invocation_type.key,
                                      getattr(invocation_type, 'data', None)),
                  'function_args': function_args,
                  'function_kwds': function_kwds})

    def _invoke_method_on_worker_pyro(
            self,
            worker_name,
            method_name,
            method_args=(),
            method_kwds=None,
            oneway=False):

        return self._action_manager.queue(
            queue_name=self.get_server_for_worker(worker_name),
            worker_name=worker_name,
            action="_invoke_method_impl",
            generate_response=not oneway,
            args=(method_name,),
            kwds={'method_args': method_args,
                  'method_kwds': method_kwds})

    #
    # Abstract methods for ScenarioTreeManagerClient:
    #

    def _init_client(self):
        assert self._scenario_tree is not None
        return self.AsyncResult(None, result=True)

    def _invoke_function_on_worker_impl(self,
                                        worker_name,
                                        function_name,
                                        module_name,
                                        invocation_type=InvocationType.Single,
                                        function_args=(),
                                        function_kwds=None,
                                        async=False,
                                        oneway=False):
        assert not (async and oneway)
        assert self._action_manager is not None
        assert worker_name in self._pyro_worker_list
        start_time = time.time()

        if self._options.verbose:
            print("Invoking external function=%s in module=%s "
                  "on worker=%s"
                  % (function_name, module_name, worker_name))

        action_handle = self._invoke_function_on_worker_pyro(
            worker_name,
            function_name,
            module_name,
            invocation_type=invocation_type,
            function_args=function_args,
            function_kwds=function_kwds,
            oneway=oneway)

        if oneway:
            action_handle = None

        result = self.AsyncResult(
            self._action_manager, action_handle_data=action_handle)

        if not async:
            result = result.complete()

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _invoke_method_on_worker_impl(self,
                                      worker_name,
                                      method_name,
                                      method_args=(),
                                      method_kwds=None,
                                      async=False,
                                      oneway=False):

        assert self._action_manager is not None
        assert worker_name in self._pyro_worker_list
        start_time = time.time()

        if self._options.verbose:
            print("Invoking method=%s on worker=%s"
                  % (method_name, worker_name))

        action_handle = self._invoke_method_on_worker_pyro(
            worker_name,
            method_name,
            method_args=method_args,
            method_kwds=method_kwds,
            oneway=oneway)

        if oneway:
            action_handle = None

        result = self.AsyncResult(
            self._action_manager, action_handle_data=action_handle)

        if not async:
            result = result.complete()

        end_time = time.time()
        if self._options.output_times or \
           self._options.verbose:
            print("Method invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _worker_names_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_worker_for_scenario_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_worker_for_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _get_scenarios_for_worker_impl(self, worker_name):
        assert worker_name in self._pyro_worker_list
        return self._pyro_worker_scenarios_map[worker_name]

    def _get_bundles_for_worker_impl(self, worker_name):
        assert worker_name in self._pyro_worker_list
        return self._pyro_worker_bundles_map[worker_name]

    #
    # Abstract methods for ScenarioTreeManager:
    #

    def _close_impl(self):
        if self._action_manager is not None:
            if self._error_shutdown:
                self.release_scenariotreeservers(ignore_errors=2)
            else:
                self.release_scenariotreeservers()
        if self._options.pyro_shutdown:
            print("Shutting down Pyro components.")
            shutdown_pyro_components(
                host=self._options.pyro_host,
                port=self._options.pyro_port,
                num_retries=0,
                caller_name=self.__class__.__name__)

    def _invoke_function_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _invoke_method_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _add_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    def _remove_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    #
    # Extended interface for Pyro
    #

    def acquire_scenariotreeservers(self, num_servers, timeout=None):
        """Acquire a pool of scenario tree servers and initialize the
        action manager."""

        assert self._action_manager is None
        self._action_manager = ScenarioTreeActionManagerPyro(
            verbose=self._options.verbose,
            host=self._options.pyro_host,
            port=self._options.pyro_port)
        self._action_manager.acquire_servers(num_servers, timeout=timeout)
        # extract server options
        server_options = ScenarioTreeServerPyro.\
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
                    queue_name=server_name,
                    action="ScenarioTreeServerPyro_setup",
                    options=server_options,
                    generate_response=True))
            self._pyro_server_workers_map[server_name] = []
        self.unpause_transmit()
        self._action_manager.wait_all(action_handles)
        for ah in action_handles:
            self._action_manager.get_results(ah)

        return len(self._action_manager.server_pool)

    def release_scenariotreeservers(self, ignore_errors=False):
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
        for worker_name in list(self._pyro_worker_server_map.keys()):
            self.remove_worker(worker_name)

        generate_response = None
        action_name = None
        if self._options.pyro_shutdown_workers:
            action_name = 'ScenarioTreeServerPyro_shutdown'
            generate_response = False
        else:
            action_name = 'ScenarioTreeServerPyro_reset'
            generate_response = True

        # transmit reset or shutdown requests
        action_handles = []
        self.pause_transmit()

        self._action_manager.ignore_task_errors = ignore_errors
        for server_name in self._action_manager.server_pool:
            action_handles.append(self._action_manager.queue(
                queue_name=server_name,
                action=action_name,
                generate_response=generate_response))
        self.unpause_transmit()
        if generate_response:
            self._action_manager.wait_all(action_handles)
            for ah in action_handles:
                self._action_manager.get_results(ah)
        self._action_manager.close()
        self._action_manager = None
        self._pyro_server_workers_map = {}
        self._pyro_worker_server_map = {}

    def pause_transmit(self):
        """Pause transmission of action requests. Return whether
        transmission was already paused."""
        assert self._action_manager is not None
        self._action_manager.pause()
        was_paused = self._transmission_paused
        self._transmission_paused = True
        return was_paused

    def unpause_transmit(self):
        """Unpause transmission of action requests and bulk transmit
        anything queued."""
        assert self._action_manager is not None
        self._action_manager.unpause()
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
                    key=lambda k: len(self._pyro_server_workers_map.get(k,[])))

        if self._options.verbose:
            print("Initializing worker with name %s on scenario tree server %s"
                  % (worker_name, server_name))

        if isinstance(worker_options, PySPConfigBlock):
            worker_class = ScenarioTreeServerPyro.\
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
        _worker_init = WorkerInit(type_=worker_init.type_.key,
                                 names=worker_init.names,
                                 data=worker_init.data)

        action_handle = self._action_manager.queue(
            queue_name=server_name,
            action="ScenarioTreeServerPyro_initialize",
            worker_type=worker_registered_name,
            worker_name=worker_name,
            worker_init=_worker_init,
            options=worker_options,
            generate_response=not oneway)

        self._pyro_server_workers_map[server_name].append(worker_name)
        self._pyro_worker_server_map[worker_name] = server_name
        self._pyro_worker_list.append(worker_name)

        if worker_init.type_ == WorkerInitType.Scenarios:
            self._pyro_worker_scenarios_map[worker_name] = worker_init.names
        else:
            assert worker_init.type_ == WorkerInitType.Bundles
            self._pyro_worker_bundles_map[worker_name] = worker_init.names
            self._pyro_worker_scenarios_map[worker_name] = []
            for bundle_name in worker_init.names:
                self._pyro_worker_scenarios_map[worker_name].\
                    extend(worker_init.data[bundle_name])

        return action_handle

    def remove_worker(self, worker_name):
        assert self._action_manager is not None
        server_name = self.get_server_for_worker(worker_name)
        self._action_manager.queue(
            queue_name=server_name,
            action="ScenarioTreeServerPyro_release",
            worker_name=worker_name,
            generate_response=False)
        self._pyro_server_workers_map[server_name].remove(worker_name)
        del self._pyro_worker_server_map[worker_name]
        self._pyro_worker_list.remove(worker_name)

    def get_server_for_worker(self, worker_name):
        try:
            return self._pyro_worker_server_map[worker_name]
        except KeyError:
            raise KeyError(
                "Scenario tree worker with name %s does not exist on "
                "any scenario tree servers" % (worker_name))

#
# This class extends the initialization process of
# _ScenarioTreeManagerClientPyroAdvanced so that scenario tree servers are
# automatically acquired and assigned worker instantiations that
# manage all scenarios / bundles (thereby completing everything
# necessary to implement the ScenarioTreeManager and
# ScenarioTreeManagerClient interfaces).
#

class ScenarioTreeManagerClientPyro(_ScenarioTreeManagerClientPyroAdvanced,
                                    PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the ScenarioTreeManagerClientPyro class")
    safe_declare_common_option(_declared_options,
                               "pyro_required_scenariotreeservers")
    safe_declare_common_option(_declared_options,
                               "pyro_find_scenariotreeservers_timeout")
    safe_declare_common_option(_declared_options,
                               "pyro_multiple_scenariotreeserver_workers")
    safe_declare_common_option(_declared_options,
                               "pyro_handshake_at_startup")

    default_registered_worker_name = 'ScenarioTreeManagerWorkerPyro'

    def __init__(self, *args, **kwds):
        self._scenario_to_worker_map = {}
        self._bundle_to_worker_map = {}
        self._worker_registered_name = kwds.pop('registered_worker_name',
                                                self.default_registered_worker_name)
        super(ScenarioTreeManagerClientPyro, self).__init__(*args, **kwds)

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
                     cls.default_registered_worker_name)
        options = super(ScenarioTreeManagerClientPyro, cls).\
                  register_options(*args, **kwds)
        if registered_worker_name is not None:
            worker_type = ScenarioTreeServerPyro.\
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

        assert len(self._pyro_server_workers_map) == \
            len(self._action_manager.server_pool)
        assert len(self._pyro_worker_server_map) == 0
        assert len(self._pyro_worker_list) == 0
        scenario_instance_factory = \
            self._scenario_tree._scenario_instance_factory

        worker_type = ScenarioTreeServerPyro.\
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
        if not self._options.pyro_handshake_at_startup:
            self.pause_transmit()
        action_handle_data = {}
        for cntr, server_name in enumerate(worker_initializations):

            if self._options.pyro_multiple_scenariotreeserver_workers:

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

                    if self._options.pyro_handshake_at_startup:
                        action_handle_data[worker_name] =  \
                            self.AsyncResult(
                                self._action_manager,
                                action_handle_data=action_handle).complete()
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

                if self._options.pyro_handshake_at_startup:
                    action_handle_data[worker_name] =  \
                        self.AsyncResult(
                            self._action_manager,
                            action_handle_data=action_handle).complete()
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

        if not self._options.pyro_handshake_at_startup:
            self.unpause_transmit()

        end_time = time.time()

        if self._options.output_times or \
           self._options.verbose:
            print("Initialization transmission time=%.2f seconds"
                  % (end_time - start_time))

        if self._options.pyro_handshake_at_startup:
            return self.AsyncResult(None, result=action_handle_data)
        else:
            return self.AsyncResult(
                self._action_manager, action_handle_data=action_handle_data)

    #
    # Abstract methods for ScenarioTreeManagerClient:
    #

    # Override the implementation on _ScenarioTreeManagerClientPyroAdvanced
    def _init_client(self):
        assert self._scenario_tree is not None
        if self._scenario_tree.contains_bundles():
            for bundle in self._scenario_tree._scenario_bundles:
                self._add_bundle(bundle.name, bundle._scenario_names)
            num_jobs = len(self._scenario_tree._scenario_bundles)
            if self._options.verbose:
                print("Bundle jobs available: %s"
                      % (str(num_jobs)))
        else:
            num_jobs = len(self._scenario_tree._scenarios)
            if self._options.verbose:
                print("Scenario jobs available: %s"
                      % (str(num_jobs)))

        servers_required = self._options.pyro_required_scenariotreeservers
        if servers_required == 0:
            servers_required = num_jobs
        elif servers_required > num_jobs:
            if servers_required > num_jobs:
                print("Value assigned to pyro_required_scenariotreeservers option (%s) "
                      "is greater than the number of available jobs (%s). "
                      "Limiting the number of servers to acquire to %s"
                      % (servers_required, num_jobs, num_jobs))
            servers_required = num_jobs

        timeout = self._options.pyro_find_scenariotreeservers_timeout if \
                  (self._options.pyro_required_scenariotreeservers == 0) else \
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

        worker_names = sorted(self._pyro_worker_server_map)

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

                result = self.invoke_function(
                    callback_name,
                    self._callback_mapped_module_name[callback_module_key],
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
                oneway=not self._options.pyro_handshake_at_startup)

        # run the user script to initialize variable bounds
        if len(self._options.postinit_callback_location):

            for callback_module_key, callback_name in zip(self._postinit_keys,
                                                          self._postinit_names):
                if self._options.verbose:
                    print("Transmitting invocation of user defined postinit "
                          "callback function defined in module: %s"
                          % (self._callback_mapped_module_name[callback_module_key]))

                # Transmit invocation to scenario tree workers
                self.invoke_function(
                    callback_name,
                    self._callback_mapped_module_name[callback_module_key],
                    invocation_type=InvocationType.PerScenario,
                    oneway=not self._options.pyro_handshake_at_startup)

        return initialization_handle

    # implemented by _ScenarioTreeManagerClientPyroAdvanced
    #def _invoke_function_on_worker_impl(...)

    # implemented by _ScenarioTreeManagerClientPyroAdvanced
    #def _invoke_method_on_worker_impl(...)

    def _worker_names_impl(self):
        return self._pyro_worker_list

    def _get_worker_for_scenario_impl(self, scenario_name):
        return self._scenario_to_worker_map[scenario_name]

    def _get_worker_for_bundle_impl(self, bundle_name):
        return self._bundle_to_worker_map[bundle_name]

    #
    # Abstract Methods for ScenarioTreeManager:
    #

    # implemented by _ScenarioTreeManagerClientPyroAdvanced
    #def _close_impl(...)

    def _invoke_function_impl(
            self,
            function_name,
            module_name,
            invocation_type=InvocationType.Single,
            function_args=(),
            function_kwds=None,
            async=False,
            oneway=False):
        assert not (async and oneway)
        assert self._action_manager is not None
        start_time = time.time()

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

        action_handle_data = None
        map_result = None
        if (invocation_type == InvocationType.Single) or \
           (invocation_type == InvocationType.PerBundle) or \
           (invocation_type == InvocationType.PerScenario):

            was_paused = self.pause_transmit()
            action_handle_data = {}

            for worker_name in self._pyro_worker_list:
                action_handle_data[self._invoke_function_on_worker_pyro(
                    worker_name,
                    function_name,
                    module_name,
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

        elif (invocation_type == InvocationType.OnScenario):

            action_handle_data = self._invoke_function_on_worker_pyro(
                self.get_worker_for_scenario(invocation_type.data),
                function_name,
                module_name,
                invocation_type=invocation_type,
                function_args=function_args,
                function_kwds=function_kwds,
                oneway=oneway)

        elif (invocation_type == InvocationType.OnBundle):

            action_handle_data = self._invoke_function_on_worker_pyro(
                self.get_worker_for_bundle(invocation_type.data),
                function_name,
                module_name,
                invocation_type=invocation_type,
                function_args=function_args,
                function_kwds=function_kwds,
                oneway=oneway)

        elif (invocation_type == InvocationType.OnScenarios) or \
             (invocation_type == InvocationType.OnBundles):

            _get_worker_func = None
            _inocation_type = None
            if invocation_type == InvocationType.OnScenarios:
                _get_worker_func = self.get_worker_for_scenario
                _invocation_type = InvocationType.OnScenarios
            else:
                assert invocation_type == InvocationType.OnBundles
                _get_worker_func = self.get_worker_for_bundle
                _invocation_type = InvocationType.OnBundles

            worker_map = {}
            for object_name in invocation_type.data:
                worker_name = _get_worker_func(object_name)
                if worker_name not in worker_map:
                    worker_map[worker_name] = []
                worker_map[worker_name].append(object_name)

            was_paused = self.pause_transmit()
            action_handle_data = {}
            for worker_name in worker_map:
                action_handle_data[self._invoke_function_on_worker_pyro(
                    worker_name,
                    function_name,
                    module_name,
                    invocation_type=_invocation_type(worker_map[worker_name]),
                    function_args=function_args,
                    function_kwds=function_kwds,
                    oneway=oneway)] = worker_name

            map_result = lambda ah_to_result: \
                         dict((key, result[key])
                              for result in itervalues(ah_to_result)
                              for key in result)

            if not was_paused:
                self.unpause_transmit()

        elif (invocation_type == InvocationType.PerScenarioChained) or \
             (invocation_type == InvocationType.PerBundleChained):

            if self._transmission_paused:
                raise ValueError("Chained invocation type %s cannot be executed "
                                 "when Pyro transmission is paused"
                                 % (invocation_type))

            result = function_args
            for i in xrange(len(self._pyro_worker_list) - 1):
                worker_name = self._pyro_worker_list[i]
                result = self.AsyncResult(
                    self._action_manager,
                    action_handle_data=self._invoke_function_on_worker_pyro(
                        worker_name,
                        function_name,
                        module_name,
                        invocation_type=invocation_type,
                        function_args=result,
                        function_kwds=function_kwds,
                        oneway=False)).complete()
                if len(function_args) == 0:
                    result = ()

            action_handle_data = self._invoke_function_on_worker_pyro(
                self._pyro_worker_list[-1],
                function_name,
                module_name,
                invocation_type=invocation_type,
                function_args=result,
                function_kwds=function_kwds,
                oneway=oneway)

        elif (invocation_type == InvocationType.OnScenariosChained) or \
             (invocation_type == InvocationType.OnBundlesChained):

            if self._transmission_paused:
                raise ValueError("Chained invocation type %s cannot be executed "
                                 "when Pyro transmission is paused"
                                 % (invocation_type))

            _get_worker_func = None
            _inocation_type = None
            if invocation_type == InvocationType.OnScenariosChained:
                _get_worker_func = self.get_worker_for_scenario
                _invocation_type = InvocationType.OnScenariosChained
            else:
                assert invocation_type == InvocationType.OnBundlesChained
                _get_worker_func = self.get_worker_for_bundle
                _invocation_type = InvocationType.OnBundlesChained

            #
            # We guarantee to execute the chained call in the same
            # order as the list of names on the invocation_type, but
            # we try to be as efficient about this as possible. E.g.,
            # if the order of the chain allows for more than one piece
            # of it to be executed on the worker in a single call, we
            # take advantage of that.
            #
            assert len(invocation_type.data) > 0
            object_names = list(reversed(invocation_type.data))
            object_names_for_worker = []
            result = function_args
            while len(object_names) > 0:
                object_names_for_worker.append(object_names.pop())
                worker_name = _get_worker_func(object_names_for_worker[-1])
                if (len(object_names) == 0) or \
                   (worker_name != _get_worker_func(object_names[-1])):
                    action_handle_data=self._invoke_function_on_worker_pyro(
                        worker_name,
                        function_name,
                        module_name,
                        invocation_type=_invocation_type(object_names_for_worker),
                        function_args=result,
                        function_kwds=function_kwds,
                        oneway=False)
                    if len(object_names) != 0:
                        result = self.AsyncResult(
                            self._action_manager,
                            action_handle_data=action_handle_data).complete()
                    if len(function_args) == 0:
                        result = ()
                    object_names_for_worker = []

        else:
            raise ValueError("Unexpected function invocation type '%s'. "
                             "Expected one of %s"
                             % (invocation_type,
                                [str(v) for v in InvocationType._values]))

        if oneway:
            action_handle_data = None
            map_result = None

        result = self.AsyncResult(
            self._action_manager,
            action_handle_data=action_handle_data,
            map_result=map_result)

        if not async:
            result = result.complete()

        end_time = time.time()

        if self._options.output_times or \
           self._options.verbose:
            print("External function invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return result

    def _invoke_method_impl(
            self,
            method_name,
            method_args=(),
            method_kwds=None,
            async=False,
            oneway=False):
        assert not (async and oneway)
        assert self._action_manager is not None
        start_time = time.time()

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

        if method_kwds is None:
            method_kwds = {}

        was_paused = self.pause_transmit()
        action_handle_data = dict(
            (self._action_manager.queue(
                queue_name=self.get_server_for_worker(worker_name),
                worker_name=worker_name,
                action=method_name,
                generate_response=not oneway,
                args=method_args,
                kwds=method_kwds),
             worker_name) for worker_name in self._pyro_worker_list)
        if not was_paused:
            self.unpause_transmit()

        if oneway:
            action_handle_data = None

        result = self.AsyncResult(
            self._action_manager,
            action_handle_data=action_handle_data)

        if not async:
            result = result.complete()

        end_time = time.time()

        if self._options.output_times or \
           self._options.verbose:
            print("Method invocation request transmission "
                  "time=%.2f seconds" % (end_time - start_time))

        return result

    # TODO
    def _add_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    # TODO
    def _remove_bundle_impl(self, *args, **kwds):
        raise NotImplementedError(type(self).__name__+": This method is abstract")

    #
    # Extended Interface for Pyro
    #

    def get_server_for_scenario(self, scenario_name):
        return self.get_server_for_worker(
            self.get_worker_for_scenario(scenario_name))

    def get_server_for_bundle(self, bundle_name):
        return self.get_server_for_worker(
            self.get_worker_for_bundle(bundle_name))
