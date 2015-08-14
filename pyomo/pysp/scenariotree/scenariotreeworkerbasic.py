#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ScenarioTreeWorkerBasic",)

import time

import pyutilib.misc
from pyutilib.misc.config import ConfigBlock
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import safe_register_common_option
from pyomo.pysp.scenariotree.scenariotreemanager \
    import (_ScenarioTreeWorkerImpl,
            _ScenarioTreeManager)
from pyomo.pysp.scenariotree.scenariotreeserverutils \
    import (WorkerInitType,
            InvocationType)

from six import iteritems

class ScenarioTreeWorkerBasic(_ScenarioTreeWorkerImpl,
                              _ScenarioTreeManager,
                              PySPConfiguredObject):

    _registered_options = \
        ConfigBlock("Options registered for the ScenarioTreeWorker class")

    #
    # scenario instance construction
    #
    safe_register_common_option(_registered_options,
                                "objective_sense_stage_based")
    safe_register_common_option(_registered_options,
                                "compile_scenario_instances")
    safe_register_common_option(_registered_options,
                                "output_instance_construction_time")

    #
    # various
    #
    safe_register_common_option(_registered_options,
                                "verbose")
    safe_register_common_option(_registered_options,
                                "profile_memory")

    def __init__(self,
                 server_name,
                 full_scenario_tree,
                 worker_name,
                 init_type,
                 init_data,
                 *args,
                 **kwds):

        super(ScenarioTreeWorkerBasic, self).__init__(*args, **kwds)
        # pyutilib.Enum can not be serialized depending on the
        # serializer type used by Pyro, so we just send the
        # key name

        if self._options.verbose:
            print("Initializing ScenarioTreeWorkerBasic with options:")
            self.display_options()
            print("")

        # The name of the scenario tree server owning this worker
        self._server_name = server_name
        # So we have access to real scenario and bundle probabilities
        self._full_scenario_tree = full_scenario_tree
        self._worker_name = worker_name

        scenarios_to_construct = []
        if (init_type == WorkerInitType.ScenarioList) or \
           (init_type == WorkerInitType.ScenarioBundle):

            if init_type == WorkerInitType.ScenarioBundle:
                if self._options.verbose:
                    print("Constructing worker with name %s for bundled scenarios %s"
                          % (worker_name, str(init_data)))
            else:
                assert init_type == WorkerInitType.ScenarioList
                if self._options.verbose:
                    print("Constructing worker with name %s for scenarios list %s"
                          % (worker_name, str(init_data)))

            scenarios_to_construct.extend(init_data)

        elif (init_type == WorkerInitType.ScenarioBundleList):

            if self._options.verbose:
                print("Constructing worker with name %s for bundle list:"
                      % (worker_name))
                for bundle_name in init_data:
                    print("  - %s: %s" % (bundle_name, init_data[bundle_name]))

            for bundle_name in init_data:
                scenarios_to_construct.extend(init_data[bundle_name])

        else:
            assert init_type == WorkerInitType.Scenario
            if self._options.verbose:
                print("Constructing worker with name %s for scenario %s"
                      % (worker_name, str(init_data)))

            scenarios_to_construct.append(init_data)

        # compress the scenario tree to reflect those instances for
        # which this ph solver server is responsible for constructing.
        self._scenario_tree = \
            self._full_scenario_tree.make_compressed(scenarios_to_construct,
                                                     normalize=False)

        self._instances = \
            self._full_scenario_tree._scenario_instance_factory.\
            construct_instances_for_scenario_tree(
                self._scenario_tree,
                output_instance_construction_time=\
                   self._options.output_instance_construction_time,
                profile_memory=self._options.profile_memory,
                compile_scenario_instances=self._options.compile_scenario_instances)

        # with the scenario instances now available, have the scenario
        # tree compute the variable match indices at each node.
        self._scenario_tree.linkInInstances(
            self._instances,
            objective_sense=self._options.objective_sense_stage_based,
            create_variable_ids=True)

        self._objective_sense = \
            self._scenario_tree._scenarios[0]._objective_sense
        assert all(_s._objective_sense == self._objective_sense
                   for _s in self._scenario_tree._scenarios)

        #
        # Create bundle if needed
        #
        if init_type == WorkerInitType.ScenarioBundle:
            assert not self._scenario_tree.contains_bundle(worker_name)
            self.add_bundle(worker_name, init_data)
            assert self._scenario_tree.contains_bundle(worker_name)
        elif init_type == WorkerInitType.ScenarioBundleList:
            for bundle_name in init_data:
                assert not self._scenario_tree.contains_bundle(bundle_name)
                self.add_bundle(bundle_name, init_data[bundle_name])
                assert self._scenario_tree.contains_bundle(bundle_name)

    def assign_aggregate_user_data(self, aggregate_user_data):
        if self._options.verbose:
            print("Received request to invoke assign_aggregate_user_data "
                  "method on worker %s" % (self._worker_name))

        self._aggregate_user_data = aggregate_user_data

    #
    # Invoke the indicated function in the specified module.
    #

    def invoke_external_function(self,
                                 invocation_type,
                                 module_name,
                                 function_name,
                                 function_args,
                                 function_kwds):

        start_time = time.time()

        # pyutilib.Enum can not be serialized depending on the
        # serializer type used by Pyro, so we just send the
        # key name
        invocation_type = getattr(InvocationType, invocation_type)

        if self._options.verbose:
            print("Received request to invoke external function"
                  "="+function_name+" in module="+module_name)

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

        if function_args is None:
            function_args = ()
        if function_kwds is None:
            function_kwds = {}

        call_items = None
        if invocation_type == InvocationType.SingleInvocation:
            pass
        elif (invocation_type == InvocationType.PerBundleInvocation) or \
             (invocation_type == InvocationType.PerBundleChainedInvocation):
            if not self._scenario_tree.contains_bundles():
                raise ValueError(
                    "Received request for bundle invocation type "
                    "but the scenario tree does not contain bundles.")
            call_items = iteritems(self._scenario_tree._scenario_bundle_map)
        elif (invocation_type == InvocationType.PerScenarioInvocation) or \
             (invocation_type == InvocationType.PerScenarioChainedInvocation):
            call_items = iteritems(self._scenario_tree._scenario_map)
        elif (invocation_type == InvocationType.PerNodeInvocation) or \
             (invocation_type == InvocationType.PerNodeChainedInvocation):
            call_items = iteritems(self._scenario_tree._tree_node_map)
        else:
            raise ValueError("Unexpected function invocation type '%s'. "
                             "Expected one of %s"
                             % (invocation_type,
                                [str(v) for v in InvocationType._values]))

        function = getattr(this_module, module_attrname)
        if subname is not None:
            function = getattr(function, subname)

        result = None
        if (invocation_type == InvocationType.SingleInvocation):
            result = function(self,
                              self._scenario_tree,
                              *function_args,
                              **function_kwds)
        elif (invocation_type == InvocationType.PerBundleChainedInvocation) or \
             (invocation_type == InvocationType.PerScenarioChainedInvocation) or \
             (invocation_type == InvocationType.PerNodeChainedInvocation):
            result = function_args
            for call_name, call_object in call_items:
                result = function(self,
                                  self._scenario_tree,
                                  call_object,
                                  *result,
                                  **function_kwds)
        else:
            result = dict((call_name,function(self,
                                              self._scenario_tree,
                                              call_object,
                                              *function_args,
                                              **function_kwds))
                          for call_name, call_object in call_items)

        end_time = time.time()
        if self._options.output_times:
            print("External function invocation time=%.2f seconds"
                  % (end_time - start_time))

        return result

    def _close_impl(self):
        _ScenarioTreeWorkerImpl._close_impl(self)
        ignored_options = dict((_c._name, _c.value(False))
                               for _c in self._options.unused_user_values())
        if len(ignored_options):
            print("")
            print("*** WARNING: The following options were explicitly "
                  "set but never accessed by worker %s: "
                  % (self._worker_name))
            for name in ignored_options:
                print(" - %s: %s" % (name, ignored_options[name]))
            print("*** If you believe this is a bug, please report it "
                  "to the PySP developers.")
            print("")
